package com.extrahand.cards.core.table

import com.extrahand.cards.core.CardImage
import com.extrahand.cards.core.DatasetCodec
import com.extrahand.cards.core.KnnCardRecognizer
import com.extrahand.cards.core.KnnIndex
import com.extrahand.cards.core.LumaGridV1
import com.extrahand.cards.core.SampleRecord
import com.extrahand.cards.core.l2Normalize
import kotlin.math.sqrt

/**
 * Geometry of a corner window, as fractions of the card, and the grids sampled inside it.
 *
 * - [WIDE] is the luma-grid-v1 corner region (30/126 x 52/176, grids 16x24 + 8x12), so it can be sliced
 *   from stored vectors with no images at all.
 * - [NARROW] covers only the rank+suit strip along the card's long edge. In a fanned hand that strip stays
 *   visible while the rest of the corner is under the next card. Used whenever sample photos are available.
 */
data class CornerLayout(val fracW: Float, val fracH: Float, val lumaGx: Int, val lumaGy: Int, val redGx: Int, val redGy: Int) {
    val lumaDim get() = lumaGx * lumaGy
    val dim get() = lumaDim + redGx * redGy
    val wRed get() = 3f / sqrt((redGx * redGy).toFloat())

    companion object {
        val WIDE = CornerLayout(
            LumaGridV1.CORNER_W.toFloat() / LumaGridV1.CANON_W, LumaGridV1.CORNER_H.toFloat() / LumaGridV1.CANON_H,
            LumaGridV1.CORNER_GX, LumaGridV1.CORNER_GY, LumaGridV1.RED_GX, LumaGridV1.RED_GY,
        )
        val NARROW = CornerLayout(0.17f, 0.30f, 10, 26, 5, 13)
    }
}

/**
 * Matches a card's **corner index** (rank + suit in the corner) on its own, so cards that are mostly hidden
 * (fanned hands, spreads, piles) can still be read.
 *
 * Two ways to build one:
 * - [fromImages]: from the sample photos, rectified like table cards ([Rectifier]). Both corners of each
 *   photo are used, with the [CornerLayout.NARROW] window. This is the accurate path.
 * - [from]: from stored luma-grid-v1 vectors alone. Blocks 1 and 3 of each vector *are* the wide corner,
 *   so no images are needed. See docs/FEATURE_SPEC.md ("Corner view").
 *
 * Queries come from [describe], which samples the layout's grids from an oriented window of the frame.
 */
class CornerIndex private constructor(private val index: KnnIndex, val layout: CornerLayout) {
    private val dim = layout.dim

    /** Unit-length mean corner vector per label: a cheap first-pass screen before k-NN. */
    private val centroidLabels: Array<String>
    private val centroids: FloatArray

    init {
        val sums = LinkedHashMap<String, FloatArray>()
        index.forEach { _, label, v ->
            val s = sums.getOrPut(label) { FloatArray(dim) }
            for (i in 0 until dim) s[i] += v[i]
        }
        centroidLabels = sums.keys.toTypedArray()
        centroids = FloatArray(centroidLabels.size * dim)
        sums.values.forEachIndexed { c, s -> l2Normalize(s); System.arraycopy(s, 0, centroids, c * dim, dim) }
    }

    val size: Int get() = index.size
    val labels: Set<String> get() = centroidLabels.toSet()

    data class Screen(val label: String, val similarity: Float)

    /** Best centroid match (fast, approximate). Null if the index is empty. */
    fun screen(q: FloatArray): Screen? {
        if (centroidLabels.isEmpty()) return null
        var best = -1; var bestSim = Float.NEGATIVE_INFINITY
        for (c in centroidLabels.indices) {
            var d = 0f; val base = c * dim
            for (i in 0 until dim) d += centroids[base + i] * q[i]
            if (d > bestSim) { bestSim = d; best = c }
        }
        return Screen(centroidLabels[best], bestSim)
    }

    /** k-NN vote over individual samples. */
    fun classify(q: FloatArray, k: Int = 5): List<KnnIndex.LabelScore> = index.classify(q, k)

    // ---- per-sample matching ----
    private val n = index.size
    private val mat = FloatArray(n * dim)
    private val lab = arrayOfNulls<String>(n)

    init {
        var i = 0
        index.forEach { _, label, v -> System.arraycopy(v, 0, mat, i * dim, dim); lab[i] = label; i++ }
    }

    /**
     * Result of [match]: the label whose closest sample is most similar to the query, that similarity, and
     * [margin], the gap to the closest sample of any other label.
     */
    data class Match(val label: String, val similarity: Float, val margin: Float)

    /**
     * Compares against every sample, not centroids, because one label spans several deck designs (often
     * one photo per deck). Note: hubness-corrected scoring (CSLS) was tried on real photos and ranked true
     * index windows below stray ones, since a real index resembles other decks' indexes. Plain cosine is used.
     */
    fun match(q: FloatArray): Match? {
        if (n == 0) return null
        val bestByLabel = HashMap<String, Float>()
        for (i in 0 until n) {
            var d = 0f; val base = i * dim
            for (k in 0 until dim) d += mat[base + k] * q[k]
            val cur = bestByLabel[lab[i]!!]
            if (cur == null || d > cur) bestByLabel[lab[i]!!] = d
        }
        var top: String? = null; var topSim = Float.NEGATIVE_INFINITY; var second = Float.NEGATIVE_INFINITY
        for ((l, sim) in bestByLabel) {
            if (sim > topSim) { second = topSim; topSim = sim; top = l } else if (sim > second) second = sim
        }
        return Match(top!!, topSim, if (second == Float.NEGATIVE_INFINITY) 1f else topSim - second)
    }

    /** Descriptor of a window ([origin], window axes [ax], [ay]) under this index's layout. */
    fun describe(planes: FramePlanes, origin: Pt, ax: Pt, ay: Pt): FloatArray? = describe(planes, origin, ax, ay, layout)

    companion object {
        /** Minimum luma standard deviation (0..1) of a training corner window. */
        const val TRAIN_MIN_STD = 0.07f
        const val LUMA_DIM = LumaGridV1.CORNER_GX * LumaGridV1.CORNER_GY        // 384
        const val RED_DIM = LumaGridV1.RED_GX * LumaGridV1.RED_GY               // 96
        const val DIM = LUMA_DIM + RED_DIM                                      // 480 (wide layout)
        private const val RED_OFFSET_IN_FULL = LUMA_DIM + LumaGridV1.CARD_GX * LumaGridV1.CARD_GY // 1152

        /** Wide corner region as a fraction of the card: width and height. */
        const val CORNER_FRAC_W = LumaGridV1.CORNER_W.toFloat() / LumaGridV1.CANON_W  // 0.238
        const val CORNER_FRAC_H = LumaGridV1.CORNER_H.toFloat() / LumaGridV1.CANON_H  // 0.295

        /** The wide corner view of a full luma-grid-v1 vector (480 values, unit length). */
        fun cornerView(full: FloatArray): FloatArray {
            require(full.size == LumaGridV1.dim) { "expected a ${LumaGridV1.id} vector" }
            val v = FloatArray(DIM)
            System.arraycopy(full, 0, v, 0, LUMA_DIM)
            System.arraycopy(full, RED_OFFSET_IN_FULL, v, LUMA_DIM, RED_DIM)
            l2Normalize(v)
            return v
        }

        /** Wide-layout index from a live recognizer's stored vectors (no images needed). */
        fun from(recognizer: KnnCardRecognizer): CornerIndex {
            val idx = KnnIndex(DIM)
            recognizer.forEachSample { id, label, v -> idx.add(id, label, cornerView(v)) }
            return CornerIndex(idx, CornerLayout.WIDE)
        }

        /** Wide-layout index from decoded dataset samples. */
        fun from(samples: List<SampleRecord>): CornerIndex {
            val idx = KnnIndex(DIM)
            for (s in samples) idx.add(s.id, s.label, cornerView(DatasetCodec.decodeVector(s.f, LumaGridV1.dim)))
            return CornerIndex(idx, CornerLayout.WIDE)
        }

        /** Index from precomputed descriptors: (label, descriptor) pairs in [layout]. */
        fun fromDescriptors(entries: List<Pair<String, FloatArray>>, layout: CornerLayout): CornerIndex {
            val idx = KnnIndex(layout.dim)
            entries.forEachIndexed { i, (label, v) -> idx.add("d$i", label, v) }
            return CornerIndex(idx, layout)
        }

        /** Index from rectified sample photos: (label, image). Both index corners of every photo are added. */
        fun fromImages(samples: List<Pair<String, CardImage>>, layout: CornerLayout = CornerLayout.NARROW): CornerIndex =
            fromDescriptors(samples.flatMap { (label, img) -> cornerDescriptors(img, layout).map { label to it } }, layout)

        /** The top-left and (turned 180 degrees) bottom-right corner descriptors of an upright card image. */
        fun cornerDescriptors(img: CardImage, layout: CornerLayout = CornerLayout.NARROW): List<FloatArray> {
            val planes = FramePlanes.of(img)
            val w = img.width.toFloat(); val h = img.height.toFloat()
            val ax = Pt(w * layout.fracW, 0f); val ay = Pt(0f, h * layout.fracH)
            // Training corners must show real ink: a photo that cut the corner off gives a near-blank window,
            // and a blank window would match every card margin on the table.
            return listOfNotNull(
                describe(planes, Pt(-0.5f, -0.5f), ax, ay, layout, minStd = TRAIN_MIN_STD),
                describe(planes, Pt(w - 0.5f, h - 0.5f), -ax, -ay, layout, minStd = TRAIN_MIN_STD),
            )
        }

        /**
         * Corner descriptor of an oriented window of [planes]. [origin] is the card's outer (sharp) corner,
         * [ax] runs along the card's width and [ay] along its height, both already scaled to the window
         * (layout.fracW x card width, layout.fracH x card height). Returns null if the window is mostly outside
         * the frame or has no contrast at all.
         */
        fun describe(
            planes: FramePlanes, origin: Pt, ax: Pt, ay: Pt, layout: CornerLayout = CornerLayout.WIDE, minStd: Float = 0.02f,
        ): FloatArray? {
            val l = planes.sampleGrid(origin, ax, ay, layout.lumaGx, layout.lumaGy, red = false) ?: return null
            val r = planes.sampleGrid(origin, ax, ay, layout.redGx, layout.redGy, red = true) ?: return null
            val out = FloatArray(layout.dim)
            val mean = l.average().toFloat()
            var ss = 0.0
            for (x in l) ss += (x - mean) * (x - mean).toDouble()
            val norm = sqrt(ss).toFloat()
            // A flat patch (bare table, blank card middle) has nothing to match.
            if (norm / sqrt(l.size.toFloat()) < minStd) return null
            for (i in l.indices) out[i] = (l[i] - mean) / norm * LumaGridV1.W_CORNER
            for (i in r.indices) out[layout.lumaDim + i] = r[i] * layout.wRed
            l2Normalize(out)
            return out
        }
    }
}
