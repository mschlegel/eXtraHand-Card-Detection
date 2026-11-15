package com.extrahand.cards.core

import kotlin.math.max
import kotlin.math.sqrt

/**
 * Turns a tightly cropped card image into a fixed-length, L2-normalized feature vector.
 * Vectors from the same extractor [id] are comparable with a dot product (cosine similarity).
 *
 * The dataset records which extractor produced its vectors. If you change the algorithm, give it a NEW id
 * and rebuild features from the stored JPEGs (see docs/FEATURE_SPEC.md and tools/python/cardnet.py).
 */
interface FeatureExtractor {
    val id: String
    val dim: Int
    fun extract(card: CardImage): FloatArray
}

/**
 * "luma-grid-v1": a deterministic, model-free descriptor. Exact spec in docs/FEATURE_SPEC.md; a matching
 * numpy implementation lives in tools/python/cardnet.py. Keep the three in sync.
 *
 * Blocks (concatenated, then the whole vector is L2-normalized):
 *  1. Corner index luma, 16x24   (384) - zero-mean, unit-norm, weight 1.0
 *  2. Whole-card luma,   24x32   (768) - zero-mean, unit-norm, weight 0.6
 *  3. Corner redness,     8x12   ( 96) - raw, scaled by 3/sqrt(96)
 * Corner blocks average the top-left corner with the 180-degree-rotated bottom-right corner, and the
 * whole-card block averages the card with its 180-degree rotation, so upside-down cards match.
 */
object LumaGridV1 : FeatureExtractor {
    override val id = "luma-grid-v1"

    const val CANON_W = 126
    const val CANON_H = 176
    const val CORNER_W = 30   // canonical px
    const val CORNER_H = 52   // canonical px
    const val CORNER_GX = 16
    const val CORNER_GY = 24
    const val CARD_GX = 24
    const val CARD_GY = 32
    const val RED_GX = 8
    const val RED_GY = 12
    const val W_CORNER = 1.0f
    const val W_CARD = 0.6f
    const val RED_OFFSET = 0.06f
    val W_RED = 3f / sqrt((RED_GX * RED_GY).toFloat())

    override val dim = CORNER_GX * CORNER_GY + CARD_GX * CARD_GY + RED_GX * RED_GY // 1248

    override fun extract(card: CardImage): FloatArray {
        val img = if (card.width > card.height) card.rotate90() else card

        // 1. Per-pixel luma and redness at source resolution, then area-resample to canonical size.
        val n = img.width * img.height
        val luma = FloatArray(n)
        val red = FloatArray(n)
        for (i in 0 until n) {
            val p = img.argb[i]
            val r = (p shr 16 and 0xFF) / 255f
            val g = (p shr 8 and 0xFF) / 255f
            val b = (p and 0xFF) / 255f
            luma[i] = 0.299f * r + 0.587f * g + 0.114f * b
            red[i] = max(0f, r - max(g, b) - RED_OFFSET)
        }
        val L = Grid(img.width, img.height, luma).resample(0, 0, img.width, img.height, CANON_W, CANON_H)
        val R = Grid(img.width, img.height, red).resample(0, 0, img.width, img.height, CANON_W, CANON_H)

        // 2. Blocks.
        val corner = cornerBlock(L, CORNER_GX, CORNER_GY)
        val whole = L.resample(0, 0, CANON_W, CANON_H, CARD_GX, CARD_GY).let { averageWithRot180(it) }
        val redCorner = cornerBlock(R, RED_GX, RED_GY)

        val out = FloatArray(dim)
        var o = 0
        normalizeInto(corner.data, W_CORNER, true, out, o); o += corner.data.size
        normalizeInto(whole.data, W_CARD, true, out, o); o += whole.data.size
        for (v in redCorner.data) out[o++] = v * W_RED
        l2Normalize(out)
        return out
    }

    private fun cornerBlock(src: Grid, gx: Int, gy: Int): Grid {
        val tl = src.resample(0, 0, CORNER_W, CORNER_H, gx, gy)
        val br = src.resample(CANON_W - CORNER_W, CANON_H - CORNER_H, CANON_W, CANON_H, gx, gy).rot180()
        return Grid(gx, gy, FloatArray(gx * gy) { (tl.data[it] + br.data[it]) * 0.5f })
    }

    private fun averageWithRot180(g: Grid): Grid {
        val r = g.rot180()
        return Grid(g.w, g.h, FloatArray(g.data.size) { (g.data[it] + r.data[it]) * 0.5f })
    }

    private fun normalizeInto(v: FloatArray, weight: Float, zeroMean: Boolean, out: FloatArray, offset: Int) {
        val mean = if (zeroMean) v.average().toFloat() else 0f
        var ss = 0.0
        for (x in v) { val d = x - mean; ss += d * d }
        val norm = sqrt(ss).toFloat()
        if (norm < 1e-6f) return // leave zeros
        for (i in v.indices) out[offset + i] = (v[i] - mean) / norm * weight
    }

    /** Float grid with box-filter ("area average") resampling. Spec-critical: see FEATURE_SPEC.md. */
    internal class Grid(val w: Int, val h: Int, val data: FloatArray) {
        /**
         * Resample the source sub-rectangle [x0,x1) x [y0,y1) to dw x dh. Destination cell (i, j) averages
         * source pixels with xs in [x0 + floor(i*sw/dw), x0 + max(floor(i*sw/dw)+1, floor((i+1)*sw/dw)))
         * and the analogous ys range, where sw = x1-x0, sh = y1-y0.
         */
        fun resample(x0: Int, y0: Int, x1: Int, y1: Int, dw: Int, dh: Int): Grid {
            val sw = x1 - x0
            val sh = y1 - y0
            val out = FloatArray(dw * dh)
            for (j in 0 until dh) {
                val ya = (j.toLong() * sh / dh).toInt()
                val yb = max(ya + 1, ((j + 1).toLong() * sh / dh).toInt())
                for (i in 0 until dw) {
                    val xa = (i.toLong() * sw / dw).toInt()
                    val xb = max(xa + 1, ((i + 1).toLong() * sw / dw).toInt())
                    var s = 0f
                    for (y in ya until yb) {
                        val row = (y0 + y) * w + x0
                        for (x in xa until xb) s += data[row + x]
                    }
                    out[j * dw + i] = s / ((yb - ya) * (xb - xa))
                }
            }
            return Grid(dw, dh, out)
        }

        fun rot180(): Grid = Grid(w, h, FloatArray(data.size) { data[data.size - 1 - it] })
    }
}

internal fun l2Normalize(v: FloatArray) {
    var ss = 0.0
    for (x in v) ss += x * x
    val n = sqrt(ss).toFloat()
    if (n < 1e-12f) return
    for (i in v.indices) v[i] /= n
}
