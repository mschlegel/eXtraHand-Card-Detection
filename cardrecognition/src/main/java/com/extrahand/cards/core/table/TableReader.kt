package com.extrahand.cards.core.table

import com.extrahand.cards.core.Candidate
import com.extrahand.cards.core.Card
import com.extrahand.cards.core.CardImage
import com.extrahand.cards.core.KnnCardRecognizer
import kotlin.math.PI
import kotlin.math.abs
import kotlin.math.max
import kotlin.math.min
import kotlin.math.roundToInt
import kotlin.math.sqrt

/** How a card in a [TableScene] was read. */
enum class ReadMethod {
    /** The whole card was visible and rectified, then matched against the full luma-grid-v1 vectors. */
    FULL_CARD,
    /** Only the corner index (rank + suit) was visible, as in fanned hands and spreads. */
    CORNER_INDEX,
}

data class SeenCard(
    val card: Card,
    /** k-NN vote share, 0..1. */
    val confidence: Float,
    /** Cosine similarity to the closest training sample. */
    val similarity: Float,
    val method: ReadMethod,
    /** Full-frame pixel coordinates: the whole card ([ReadMethod.FULL_CARD]) or the matched corner patch. */
    val quad: Quad,
    val alternatives: List<Candidate> = emptyList(),
)

/** Face-up cards that belong together: one fanned / spread / overlapping hand, or nearby loose cards. */
data class CardGroup(
    val cards: List<SeenCard>,
    val box: Box,
    /** True if the group was one connected blob of overlapping cards (a held or spread hand). */
    val overlapping: Boolean,
    /** Card-shaped bits of the group that could not be read (likely more cards than [cards]). */
    val unreadRegions: Int = 0,
)

enum class BackLayout {
    /** One card-sized face-down rectangle: a single card or a squared-up pile (e.g. the draw deck). */
    SINGLE_OR_PILE,
    /** Several overlapping face-down cards: a spread, a fanned hand held facing away, a messy pile. */
    SPREAD,
}

data class BackGroup(
    val layout: BackLayout,
    val box: Box,
    /** Rectified outline when [layout] is [BackLayout.SINGLE_OR_PILE]. */
    val quad: Quad?,
    /** At least this many cards (area based; a pile's thickness is not measured). */
    val minCount: Int,
    val confidence: Float,
    /** False when no card backs have been trained yet and this came from the colour heuristic. */
    val fromTraining: Boolean,
)

data class TableScene(
    val frameWidth: Int,
    val frameHeight: Int,
    val groups: List<CardGroup>,
    val backs: List<BackGroup>,
    /** Card-like regions nothing could be read from. */
    val unknown: List<Quad>,
    /** Estimated card height in frame pixels (null if no card was found). */
    val cardHeightPx: Float?,
    val elapsedMs: Long,
) {
    val cards: List<SeenCard> get() = groups.flatMap { it.cards }
    val isEmpty get() = groups.isEmpty() && backs.isEmpty()

    companion object {
        fun empty(w: Int, h: Int) = TableScene(w, h, emptyList(), emptyList(), emptyList(), null, 0)
    }
}

data class TableReaderConfig(
    /** Longest side of the segmentation image. Larger finds smaller (farther) cards but is slower. */
    val workMaxSide: Int = Segmentation.WORK_MAX,
    /** Ignore blobs smaller than this fraction of the working image. */
    val minBlobFraction: Float = 0.0003f,
    /** Blob area / fitted quad area at or above which a blob is treated as one rectangle. */
    val rectangularity: Float = 0.88f,
    /** Corner-index acceptance: k-NN best similarity and vote share. */
    val cornerMinSimilarity: Float = 0.80f,
    val cornerMinProbability: Float = 0.5f,
    /** Corner acceptance: best label's similarity minus the best other label's. */
    val cornerMinMargin: Float = 0.03f,
    /**
     * Corner acceptance when the index is built from sample photos. True index windows on real cards score
     * about 0.96-0.99 against their own photo; stray windows rarely pass 0.9.
     */
    val cornerMinSimilarityPhotos: Float = 0.92f,
    /** Centroid similarity a scan position needs before it is examined closely. */
    val screenMinSimilarity: Float = 0.55f,
    /**
     * Playing with more than one deck (Canasta, Rummy variants...). When false, a label is reported at most
     * once per group; when true, repeats are kept unless they are the same card's two corners.
     */
    val multiDeck: Boolean = false,
    /** Card-height multipliers tried when the card size has to be guessed from the blob. */
    val scaleGuesses: List<Float> = listOf(0.75f, 1.0f, 1.25f),
    /** Loose cards closer than this many card heights are grouped together. */
    val groupGap: Float = 0.35f,
)

/**
 * Reads everything on a table from one full camera frame: face-up hands (fanned, spread, overlapping or
 * loose), single cards, and face-down cards (backs, piles, spreads). The camera can be anywhere. Cards may be
 * rotated, seen at an angle, and small in the frame.
 *
 * It reuses the recognizer's trained close-up samples. Fully visible cards are rectified with a homography
 * and classified with the full vector. Partly hidden cards are read from their corner index through
 * [CornerIndex]. Train card backs in the trainer under the "Back" label (`BK`) for reliable back detection.
 * Until then, a colour heuristic flags likely backs at low confidence.
 *
 * Pure Kotlin, thread-safe, roughly 0.2-1 s per frame on a phone depending on how many cards are visible.
 * Hand-value logic is deliberately not here: consume [TableScene.groups].
 */
class TableReader(
    private val recognizer: KnnCardRecognizer,
    /** Can be swapped at any time (e.g. toggling multi-deck) without losing the cached photo index. */
    @Volatile var config: TableReaderConfig = TableReaderConfig(),
    /**
     * The JPEG of a training sample, decoded (sample id, label) -> image, or null if unavailable. When given,
     * the reader re-frames every photo with [Rectifier] and matches against those. Rectified photos line up
     * with rectified table cards far better than the stored guide-box vectors. It also uses the narrow
     * index-strip corner window, which survives the overlap in fanned hands. Called from the reading thread;
     * results are cached per sample.
     */
    private val sampleImages: ((id: String, label: String) -> CardImage?)? = null,
) {
    /** Rectified-photo data per sample id: full vector + corner descriptors (null = no usable photo). */
    private class Rectified(val full: FloatArray, val corners: List<FloatArray>)
    private val rectCache = java.util.concurrent.ConcurrentHashMap<String, Rectified>()

    /** Rebuild the indexes on the next read, e.g. after more sample photos became available. */
    fun invalidate() { cornerVersion = -1 }
    @Volatile private var faceRec: KnnCardRecognizer = recognizer

    /** Test hook for tracing decisions. */
    internal var debug: ((String) -> Unit)? = null

    @Volatile private var corner: CornerIndex? = null
    @Volatile private var cornerVersion = -1L

    private fun cornerIndex(): CornerIndex {
        val v = recognizer.dataVersion
        corner?.let { if (cornerVersion == v) return it }
        synchronized(this) {
            if (corner == null || cornerVersion != v) {
                corner = buildIndexes()
                cornerVersion = v
            }
            return corner!!
        }
    }

    /**
     * Without photos: the trainer's recognizer for whole cards and the wide corner view of its vectors.
     * With photos: a recognizer and a narrow corner index built from rectified photos. Samples without a
     * usable photo keep their stored vector for whole-card matching.
     */
    private fun buildIndexes(): CornerIndex {
        val src = sampleImages ?: run { faceRec = recognizer; return CornerIndex.from(recognizer) }
        val samples = ArrayList<Triple<String, String, FloatArray>>()
        recognizer.forEachSample { id, label, v -> samples.add(Triple(id, label, v.copyOf())) }
        val rec = KnnCardRecognizer(recognizer.extractor, recognizer.thresholds)
        val corners = ArrayList<Pair<String, FloatArray>>()
        for ((id, label, stored) in samples) {
            // Only successes are cached, so photos that arrive later (downloads) are picked up by invalidate().
            val r = rectCache[id] ?: runCatching {
                src(id, label)?.let { img -> Rectifier.rectify(img) }?.let { img ->
                    Rectified(rec.extractor.extract(img), CornerIndex.cornerDescriptors(img, CornerLayout.NARROW))
                }
            }.getOrNull()?.also { rectCache[id] = it }
            val card = Card.parse(label) ?: continue
            rec.learn(id, card, r?.full ?: stored)
            r?.corners?.forEach { corners.add(label to it) }
        }
        rectCache.keys.retainAll(samples.map { it.first }.toSet())
        faceRec = rec
        // Too few photos (e.g. still downloading): fall back to the wide view, which needs none.
        return if (corners.size >= samples.size) CornerIndex.fromDescriptors(corners, CornerLayout.NARROW)
        else CornerIndex.from(recognizer)
    }

    private val backsTrained get() = (recognizer.labelCounts()[Card.BACK_LABEL] ?: 0) > 0

    fun read(frame: CardImage): TableScene {
        val t0 = System.currentTimeMillis()
        if (recognizer.sampleCount == 0) return TableScene.empty(frame.width, frame.height)
        val seg = Segmentation.of(frame, config.workMaxSide)
        val planes = FramePlanes.of(frame)
        debug?.invoke("time prep=${System.currentTimeMillis() - t0}ms")
        val ci = cornerIndex()
        val minArea = max(30, (seg.w * seg.h * config.minBlobFraction).toInt())
        val frameArea = seg.w * seg.h

        val singles = ArrayList<SeenCard>()
        val backs = ArrayList<BackGroup>()
        val unknown = ArrayList<Quad>()
        val toScan = ArrayList<Segmentation.Component>()
        val cardHeights = ArrayList<Float>()
        // Blobs whose single-card read was ambiguous (aspect a bit off): settled after scanning them.
        val pendingSingle = java.util.IdentityHashMap<Segmentation.Component, SeenCard>()

        // ---- pass 1: rectangles in the face mask (faces, white-bordered backs) ----
        for (c in seg.faceComponents(minArea)) {
            if (c.area > frameArea * 0.97f) continue
            val quad = quadOf(c, seg.scale)
            val rect = if (quad == null) 0f else c.area / (quad.area / sq(seg.scale))
            val cardLike = quad != null && rect >= config.rectangularity && plausibleAspect(quad)
            if (!cardLike) { toScan.add(c); continue }
            when (val r = readRectangle(frame, planes, ci, quad!!, seg, c)) {
                is RectResult.Face -> {
                    // A near-card-shaped blob might still be 2 tightly overlapped cards: double-check those.
                    if (strictAspect(quad)) { singles.add(r.card); cardHeights.add(longSide(quad)) }
                    else { toScan.add(c); pendingSingle[c] = r.card }
                }
                is RectResult.Back ->
                    if (strictAspect(quad)) { backs.add(r.group); cardHeights.add(longSide(quad)) }
                    // A rectangle of backs that is too long for one card: a neat spread.
                    else backs.add(spreadBack(c, seg, quad.sidePairs().let { (x, y) -> min(x, y) }, fromTraining = r.group.fromTraining))
                RectResult.None -> toScan.add(c)
            }
        }
        val hEstimate = cardHeights.sorted().let { if (it.isEmpty()) null else it[it.size / 2] }

        // ---- pass 2: overlapping cards, read by sliding a corner window along every straight edge ----
        val scanned = ArrayList<CardGroup>()
        for (c in toScan) {
            // Ink first: its hits calibrate the card size for this blob, which makes the edge scan cheap and exact.
            val tA = System.nanoTime()
            val inkDets = inkScan(c, seg, planes, ci, hEstimate)
            val tB = System.nanoTime()
            // Cards nearer the camera are bigger, so a blob's own evidence beats the frame-wide estimate.
            val hInk = inkDets.map { it.quad.let { q -> (q.p3 - q.p0).length / ci.layout.fracH } }
                .sorted().let { if (it.isEmpty()) null else it[it.size / 2] }
            val hLocal = hInk ?: hEstimate
            val edgeDets = scanComponent(c, seg, planes, ci, hLocal, exact = hInk != null)
            debug?.invoke("time ink=${(tB - tA) / 1e6}ms edge=${(System.nanoTime() - tB) / 1e6}ms hLocal=$hLocal")
            val dets = edgeDets + inkDets
            val faces = dedupe(dets.filter { !it.card.isBack }, hEstimate ?: guessHeight(c, seg))
            val backHits = dets.count { it.card.isBack }
            val single = pendingSingle.remove(c)
            when {
                faces.size >= 2 -> scanned.add(CardGroup(faces, c.box.scaled(seg.scale), overlapping = true))
                single != null -> singles.add(single)
                faces.isNotEmpty() -> scanned.add(CardGroup(faces, c.box.scaled(seg.scale), overlapping = true))
                backHits > 0 -> backs.add(spreadBack(c, seg, hEstimate, fromTraining = true))
                else -> quadOf(c, seg.scale)?.let { if (c.area > minArea * 4) unknown.add(it) }
            }
            // A face-up hand touching face-down cards (e.g. pushed against the deck).
            if (faces.isNotEmpty() && backHits >= 3) backs.add(spreadBack(c, seg, hEstimate, fromTraining = true))
        }

        // ---- pass 3: face-down cards with no white border (object mask) ----
        val cardArea = hEstimate?.let { it * it * CARD_ASPECT / sq(seg.scale) }
        for (c in seg.objectComponents(minArea)) {
            if (cardArea != null && c.area < cardArea * 0.3f) continue
            if (c.area > frameArea * 0.5f) continue
            val quad = quadOf(c, seg.scale) ?: continue
            val rect = c.area / (quad.area / sq(seg.scale))
            if (rect >= config.rectangularity && plausibleAspect(quad)) {
                val crop = planes.warp(frame, quad.cardOrdered()) ?: continue
                val rec = faceRec.recognize(crop, topN = 2, locate = false)
                if (rec.card?.isBack == true) backs.add(BackGroup(BackLayout.SINGLE_OR_PILE, Box.of(quad.points), quad, 1, rec.confidence, true))
                else if (!backsTrained && strictAspect(quad) && looksLikeBack(crop, planes, ci, quad.cardOrdered()))
                    backs.add(BackGroup(BackLayout.SINGLE_OR_PILE, Box.of(quad.points), quad, 1, 0.4f, false))
            } else if (backsTrained) {
                val dets = scanComponent(c, seg, planes, ci, hEstimate)
                if (dets.count { it.card.isBack } >= 1) backs.add(spreadBack(c, seg, hEstimate, fromTraining = true))
            }
        }

        val groups = groupLoose(singles, scanned, hEstimate)
        val h = hEstimate ?: groups.flatMap { it.cards }.firstOrNull()?.let { longSide(it.quad) / ci.layout.fracH }
        return TableScene(frame.width, frame.height, groups, backs, unknown, h, System.currentTimeMillis() - t0)
    }

    // ------------------------------------------------------------------ rectangles

    private sealed interface RectResult {
        data class Face(val card: SeenCard) : RectResult
        data class Back(val group: BackGroup) : RectResult
        data object None : RectResult
    }

    private fun readRectangle(
        frame: CardImage, planes: FramePlanes, ci: CornerIndex, rawQuad: Quad, seg: Segmentation, c: Segmentation.Component,
    ): RectResult {
        // Seen at an angle, a card's foreshortened height can look shorter than its width, so if the natural
        // orientation is unsure, also try the quad turned by 90 degrees.
        val first = rawQuad.cardOrdered()
        var quad = first
        var crop = planes.warp(frame, quad) ?: return RectResult.None
        var rec = faceRec.recognize(crop, topN = 3, locate = false)
        if (rec.card == null) {
            val alt = Quad(first.p1, first.p2, first.p3, first.p0)
            val altCrop = planes.warp(frame, alt)
            val altRec = altCrop?.let { faceRec.recognize(it, topN = 3, locate = false) }
            if (altRec != null && (altRec.card != null || altRec.similarity > rec.similarity + 0.05f)) {
                quad = alt; crop = altCrop; rec = altRec
            }
        }
        debug?.invoke("rect quad=$quad -> ${rec.card} sim=${rec.similarity} cands=${rec.candidates.map { it.card.label + "@" + it.similarity }}")
        val card = rec.card
        if (card != null) {
            return if (card.isBack) RectResult.Back(BackGroup(BackLayout.SINGLE_OR_PILE, Box.of(quad.points), quad, 1, rec.confidence, true))
            else RectResult.Face(SeenCard(card, rec.confidence, rec.similarity, ReadMethod.FULL_CARD, quad, rec.candidates.drop(1)))
        }
        // Whole card unsure (glare, a chip on it, a finger): try its two index corners.
        val corners = listOf(
            cornerFrame(ci, quad.p0, quad.p1, quad.p3),
            cornerFrame(ci, quad.p2, quad.p3, quad.p1),
        )
        var best: SeenCard? = null
        for ((o, ax, ay) in corners) {
            val q = ci.describe(planes, o, ax, ay) ?: continue
            val s = acceptCorner(ci, q) ?: continue
            if (best == null || s.similarity > best.similarity) {
                val cd = Card.parse(s.label) ?: continue
                best = SeenCard(cd, s.probability, s.similarity, ReadMethod.CORNER_INDEX, quad)
            }
        }
        best?.let {
            return if (it.card.isBack) RectResult.Back(BackGroup(BackLayout.SINGLE_OR_PILE, Box.of(quad.points), quad, 1, it.confidence, true))
            else RectResult.Face(it)
        }
        if (!backsTrained && strictAspect(quad) && looksLikeBack(crop, planes, ci, quad))
            return RectResult.Back(BackGroup(BackLayout.SINGLE_OR_PILE, Box.of(quad.points), quad, 1, 0.4f, false))
        return RectResult.None
    }

    /** Corner patch frame at [o] with neighbours [alongW] (width direction) and [alongH]. */
    private fun cornerFrame(ci: CornerIndex, o: Pt, alongW: Pt, alongH: Pt) =
        Triple(o, (alongW - o) * ci.layout.fracW, (alongH - o) * ci.layout.fracH)

    /**
     * Untrained back heuristic: the middle of the card is mostly colourful/patterned rather than white, and
     * neither index corner resembles any trained face card.
     */
    private fun looksLikeBack(crop: CardImage, planes: FramePlanes, ci: CornerIndex, quad: Quad): Boolean {
        var white = 0; var n = 0
        for (y in (crop.height * 0.2f).toInt() until (crop.height * 0.8f).toInt() step 2)
            for (x in (crop.width * 0.2f).toInt() until (crop.width * 0.8f).toInt() step 2) {
                val p = crop.argb[y * crop.width + x]
                val r = p shr 16 and 0xFF; val g = p shr 8 and 0xFF; val b = p and 0xFF
                val mx = max(r, max(g, b)); val mn = min(r, min(g, b))
                if (mx > 150 && (mx - mn) < 0.25f * mx) white++
                n++
            }
        if (n == 0 || white.toFloat() / n > 0.2f) return false
        for ((o, ax, ay) in listOf(cornerFrame(ci, quad.p0, quad.p1, quad.p3), cornerFrame(ci, quad.p2, quad.p3, quad.p1))) {
            val q = ci.describe(planes, o, ax, ay) ?: continue
            val s = ci.screen(q) ?: continue
            if (s.similarity > 0.7f) return false
        }
        return true
    }

    // ------------------------------------------------------------------ edge scan

    internal data class Det(val card: Card, val prob: Float, val sim: Float, val quad: Quad, val center: Pt, val angle: Float)

    /**
     * For each straight stretch of the blob outline, slide a corner-sized window along it in the two
     * orientations an index can have against that edge. Edge = card top/bottom (index runs along it) or
     * edge = card side (index runs down it). Peaks that match the corner index become detections.
     */
    private fun scanComponent(
        c: Segmentation.Component, seg: Segmentation, planes: FramePlanes, ci: CornerIndex, hKnown: Float?,
        exact: Boolean = false,
    ): List<Det> {
        val s = seg.scale
        val hGuess = hKnown ?: guessHeight(c, seg)
        val heights = if (exact) listOf(hGuess) else config.scaleGuesses.map { it * hGuess }
        val wWork = hGuess * CARD_ASPECT / s
        val contour = c.contour
        if (contour.size < 8) return emptyList()
        val eps = max(1.0f, wWork * 0.012f)
        val verts = Polygons.simplifyClosed(contour, eps).sorted()
        if (verts.size < 3) return emptyList()

        val out = ArrayList<Det>()
        debug?.invoke("scan comp area=${c.area} box=${c.box} hGuess=$hGuess heights=$heights verts=${verts.size} contour=${contour.size} start=${contour.first()}")
        debug?.invoke("  verts " + verts.map { "$it:" + contour[it].let { p -> "(${p.x.toInt()},${p.y.toInt()})" } })
        for (vi in verts.indices) {
            val ia = verts[vi]; val ib = verts[(vi + 1) % verts.size]
            val run = if (ib > ia) contour.subList(ia, ib + 1) else contour.subList(ia, contour.size) + contour.subList(0, ib + 1)
            val a = contour[ia]; val b = contour[ib]
            val segLen = (b - a).length
            if (segLen < wWork * 0.16f) continue
            val inner = run.subList((run.size * 0.15f).toInt(), max((run.size * 0.15f).toInt() + 2, (run.size * 0.85f).toInt()).coerceAtMost(run.size))
            val line = Polygons.fitLine(inner) ?: continue
            var t = line.d.let { if (it.dot(b - a) < 0) -it else it }
            var n = t.perp()
            val mid = line.p
            if (!c.contains(mid + n * 2.5f)) n = -n
            // Edge line in full-res pixels, moved half a working pixel outward onto the true boundary.
            val p0 = (line.p - n * 0.5f) * s
            val sa = (a - line.p).dot(t) * s; val sb = (b - line.p).dot(t) * s
            t = t.normalized(); n = n.normalized()
            debug?.invoke("  edge ${a}->${b} len=$segLen p0=$p0 t=$t n=$n")
            for (h in heights) out += scanEdge(planes, ci, p0, t, n, sa, sb, h)
        }
        return out
    }

    private fun scanEdge(planes: FramePlanes, ci: CornerIndex, p0: Pt, t: Pt, n: Pt, sa: Float, sb: Float, h: Float): List<Det> {
        val w = h * CARD_ASPECT
        val cw = w * ci.layout.fracW
        val ch = h * ci.layout.fracH
        val out = ArrayList<Det>()
        // Orientation A: index runs along the edge (edge = card's short side). x = ±t, y = n.
        val xA = if (t.cross(n) > 0) t else -t
        // Orientation B: edge = card's long side. x = n, y = ±t.
        val yB = if (n.cross(t) > 0) t else -t
        val frames = listOf(xA * cw to n * ch, n * cw to yB * ch)
        val step = max(1f, w * 0.045f)
        for ((ax, ay) in frames) {
            val lo = sa - w * 0.12f; val hi = sb + w * 0.12f
            val samples = ArrayList<Triple<Float, Float, String>>()
            var u = lo
            while (u <= hi) {
                // The window origin is the card corner; scanning past both run ends covers both corner positions.
                val q = ci.describe(planes, p0 + t * u, ax, ay)
                val sc = q?.let { ci.screen(it) }
                samples.add(Triple(u, sc?.similarity ?: -1f, sc?.label ?: ""))
                u += step
            }
            for (i in samples.indices) {
                val (pos, sim, _) = samples[i]
                if (sim < config.screenMinSimilarity) continue
                if (i > 0 && samples[i - 1].second > sim) continue
                if (i < samples.size - 1 && samples[i + 1].second >= sim) continue
                val d = refineWindow(planes, ci, p0 + t * pos, ax, ay)
                debug?.invoke("    peak u=$pos screen=${samples[i].third}@$sim h=$h -> ${d?.card}@${d?.sim} p=${d?.prob}")
                d?.let(out::add)
            }
        }
        return out
    }

    /**
     * Index search inside a blob, for corners that lie over another card's face (white on white, so they are
     * not on the outline). This is the usual case in a hand fanned from one corner. Each ink cluster about
     * the size of a rank+suit index proposes a corner window: its long axis gives the card's vertical, its
     * length gives the card's scale. The window is refined by a small local search and then classified.
     */
    private fun inkScan(c: Segmentation.Component, seg: Segmentation, planes: FramePlanes, ci: CornerIndex, hKnown: Float?): List<Det> {
        val s = seg.scale
        val hGuess = hKnown ?: guessHeight(c, seg)
        val hW = hGuess / s
        val clusters = seg.inkClusters(c, mergeGap = max(1f, hW * 0.02f), maxArea = (hW * hW * CARD_ASPECT * 0.12f).toInt())
        val out = ArrayList<Det>()
        for (ink in clusters) {
            if (ink.majorLen < hW * 0.08f || ink.majorLen > hW * 0.45f) continue
            val ratio = ink.minorLen / ink.majorLen
            if (ratio < 0.2f) continue
            val hInk = ink.majorLen * s / INDEX_FRAC_H
            val h = hInk
            val dirs = if (ratio > 0.75f) listOf(ink.majorDir, ink.majorDir.perp()) else listOf(ink.majorDir)
            var best: Pair<Float, Triple<Pt, Pt, Pt>>? = null
            for (d in dirs) for (sign in floatArrayOf(1f, -1f)) {
                val y = d * sign
                val x = Pt(y.y, -y.x) // right-handed on screen: cross(x, y) > 0
                val ax = x * (h * CARD_ASPECT * ci.layout.fracW)
                val ay = y * (h * ci.layout.fracH)
                // The cluster centre is only roughly the index centre (neighbouring ink can merge in): multi-start.
                for (du in START_OFFSETS) for (dv in START_OFFSETS) {
                    val origin = ink.center * s - ax * (0.5f + du) - ay * (0.5f + dv)
                    val q = ci.describe(planes, origin, ax, ay) ?: continue
                    val sim = ci.match(q)?.similarity ?: continue
                    if (sim > (best?.first ?: config.screenMinSimilarity)) best = sim to Triple(origin, ax, ay)
                }
            }
            debug?.invoke("  ink at ${ink.center} dir=${ink.majorDir} len=${ink.majorLen} ratio=$ratio h=$h screen=${best?.first}")
            val (_, f) = best ?: continue
            val d = refineWindow(planes, ci, f.first, f.second, f.third)
            debug?.invoke("    -> ${d?.card} ${d?.sim}")
            d?.let { out.add(it) }
        }
        return out
    }

    /**
     * Local search around a corner window: hill-climb over shift (along both axes), rotation and scale, halving
     * the steps whenever no move improves the centroid score; then k-NN classify the best window.
     */
    private fun refineWindow(planes: FramePlanes, ci: CornerIndex, origin: Pt, ax: Pt, ay: Pt): Det? {
        val center0 = origin + ax * 0.5f + ay * 0.5f
        // params: du, dv (fractions of the window), angle (rad), log-scale
        val p = floatArrayOf(0f, 0f, 0f, 0f)
        val step = floatArrayOf(0.15f, 0.12f, Math.toRadians(8.0).toFloat(), 0.12f)
        fun window(v: FloatArray): Triple<Pt, Pt, Pt> {
            val ca = kotlin.math.cos(v[2]); val sa = kotlin.math.sin(v[2]); val sc = kotlin.math.exp(v[3])
            fun rot(q: Pt) = Pt(q.x * ca - q.y * sa, q.x * sa + q.y * ca) * sc
            val rx = rot(ax); val ry = rot(ay)
            val c = center0 + ax * v[0] + ay * v[1]
            return Triple(c - rx * 0.5f - ry * 0.5f, rx, ry)
        }
        fun score(v: FloatArray): Pair<Float, FloatArray?> {
            val (o, rx, ry) = window(v)
            val q = ci.describe(planes, o, rx, ry) ?: return -1f to null
            return (ci.match(q)?.similarity ?: -1f) to q
        }
        var (bestSim, bestQ) = score(p)
        repeat(7) {
            var moved = false
            for (k in 0 until 4) for (dir in floatArrayOf(1f, -1f)) {
                val v = p.copyOf(); v[k] += dir * step[k]
                val (sim, q) = score(v)
                if (sim > bestSim) { bestSim = sim; bestQ = q; System.arraycopy(v, 0, p, 0, 4); moved = true }
            }
            if (!moved) for (k in 0 until 4) step[k] *= 0.5f
        }
        val q = bestQ ?: return null
        val top = acceptCorner(ci, q) ?: return null
        debug?.invoke("    refine best screen=$bestSim knn=${top.label}@${top.similarity}/${top.probability}")
        val card = Card.parse(top.label) ?: return null
        val (bo, bax, bay) = window(p)
        val quad = Quad(bo, bo + bax, bo + bax + bay, bo + bay)
        return Det(card, top.probability, top.similarity, quad, quad.center, bay.angle)
    }

    /**
     * k-NN verdict on a corner window, or null if it is not convincing. Besides the similarity and vote
     * share, the best label must beat every other label's closest sample by [TableReaderConfig.cornerMinMargin].
     * With only one or two samples per deck design, ambiguous windows (card margins, court-card frames)
     * otherwise slip through as confident misreads.
     */
    private fun acceptCorner(ci: CornerIndex, q: FloatArray): com.extrahand.cards.core.KnnIndex.LabelScore? {
        val m = ci.match(q) ?: return null
        debug?.invoke("    accept? ${m.label} sim=${m.similarity} margin=${m.margin}")
        val minSim = if (ci.layout == CornerLayout.NARROW) config.cornerMinSimilarityPhotos else config.cornerMinSimilarity
        if (m.similarity < minSim || m.margin < config.cornerMinMargin) return null
        // Reported "confidence" grows with the margin over the next label (0.5 at the minimum margin).
        return com.extrahand.cards.core.KnnIndex.LabelScore(m.label, (0.5f + (m.margin - config.cornerMinMargin) * 5).coerceIn(0.5f, 1f), m.similarity)
    }

    /** Spatial NMS (one detection per spot), then per-label de-duplication. */
    private fun dedupe(dets: List<Det>, h: Float): List<SeenCard> {
        val sorted = dets.sortedByDescending { it.sim }
        val kept = ArrayList<Det>()
        for (d in sorted) {
            val r = d.quad.let { (it.p2 - it.p0).length } * 0.6f
            if (kept.none { dist(it.center, d.center) < r }) kept.add(d)
        }
        val diag = h * sqrt(1 + CARD_ASPECT * CARD_ASPECT)
        val final = ArrayList<Det>()
        for (d in kept) {
            val same = final.filter { it.card == d.card }
            val dup = if (!config.multiDeck) same.isNotEmpty()
            else same.any { dist(it.center, d.center) < diag * 1.3f && angleDiff(it.angle, d.angle) > PI * 0.7 }
            if (!dup) final.add(d)
        }
        return final.map { SeenCard(it.card, it.prob, it.sim, ReadMethod.CORNER_INDEX, it.quad) }
    }

    // ------------------------------------------------------------------ grouping / helpers

    private fun groupLoose(singles: List<SeenCard>, scanned: List<CardGroup>, h: Float?): List<CardGroup> {
        data class Unit(val cards: List<SeenCard>, val box: Box, val overlapping: Boolean)
        val units = scanned.map { Unit(it.cards, it.box, true) } + singles.map { Unit(listOf(it), Box.of(it.quad.points), false) }
        if (units.isEmpty()) return emptyList()
        val gap = (h ?: units.maxOf { max(it.box.width, it.box.height) }) * config.groupGap
        val parent = IntArray(units.size) { it }
        fun find(i: Int): Int { var x = i; while (parent[x] != x) x = parent[x]; return x }
        for (i in units.indices) for (j in i + 1 until units.size) {
            val a = units[i].box; val b = units[j].box
            val dx = max(0f, max(a.left, b.left) - min(a.right, b.right))
            val dy = max(0f, max(a.top, b.top) - min(a.bottom, b.bottom))
            if (dx <= gap && dy <= gap) parent[find(i)] = find(j)
        }
        return units.indices.groupBy { find(it) }.values.map { idx ->
            val us = idx.map { units[it] }
            val cards = if (config.multiDeck) us.flatMap { it.cards } else us.flatMap { it.cards }.groupBy { it.card }.values.map { v -> v.maxBy { it.similarity } }
            CardGroup(cards, us.map { it.box }.reduce { a, b -> a.union(b) }, us.any { it.overlapping })
        }.sortedWith(compareBy({ it.box.top }, { it.box.left }))
    }

    private fun spreadBack(c: Segmentation.Component, seg: Segmentation, h: Float?, fromTraining: Boolean): BackGroup {
        val hh = h ?: guessHeight(c, seg)
        val cardArea = hh * hh * CARD_ASPECT / sq(seg.scale)
        val count = max(1, (c.area / cardArea).roundToInt())
        return BackGroup(if (count <= 1) BackLayout.SINGLE_OR_PILE else BackLayout.SPREAD, c.box.scaled(seg.scale), null, count, 0.6f, fromTraining)
    }

    /** Card height guess for a blob of overlapping cards: its longest straight edge, else its shorter box side. */
    private fun guessHeight(c: Segmentation.Component, seg: Segmentation): Float {
        val eps = max(1f, max(c.box.width, c.box.height) * 0.02f)
        val v = Polygons.simplifyClosed(c.contour, eps).sorted()
        var longest = 0f
        for (i in v.indices) longest = max(longest, (c.contour[v[(i + 1) % v.size]] - c.contour[v[i]]).length)
        val boxShort = min(c.box.width, c.box.height)
        val h = if (longest > boxShort * 0.5f) min(longest, max(c.box.width, c.box.height)) else boxShort
        return h * seg.scale
    }

    /** Best-fit quad of a blob, in full-resolution pixels. */
    private fun quadOf(c: Segmentation.Component, scale: Float): Quad? {
        val q = Polygons.quadFromHull(Polygons.convexHull(c.contour), c.contour) ?: return null
        // Contour points are pixel centres; push the corners out to the pixel edges.
        val ctr = q.center
        fun grow(p: Pt) = p + (p - ctr).normalized() * 0.7f
        return Quad(grow(q.p0), grow(q.p1), grow(q.p2), grow(q.p3)).scaled(scale)
    }

    private fun plausibleAspect(q: Quad): Boolean {
        val (a, b) = q.sidePairs()
        val r = min(a, b) / max(a, b)
        return r in 0.35f..0.98f
    }

    private fun strictAspect(q: Quad): Boolean {
        val (a, b) = q.sidePairs()
        val r = min(a, b) / max(a, b)
        return abs(r - CARD_ASPECT) < 0.12f
    }

    private fun longSide(q: Quad): Float { val (a, b) = q.sidePairs(); return max(a, b) }

    private fun angleDiff(a: Float, b: Float): Double {
        var d = abs(a - b).toDouble() % (2 * PI)
        if (d > PI) d = 2 * PI - d
        return d
    }

    companion object {
        const val CARD_ASPECT = 63f / 88f
        /** Typical rank+suit index height as a fraction of card height (standard poker decks: 0.22-0.26). */
        const val INDEX_FRAC_H = 0.24f
        private val START_OFFSETS = floatArrayOf(-0.18f, 0f, 0.18f)
    }
}
