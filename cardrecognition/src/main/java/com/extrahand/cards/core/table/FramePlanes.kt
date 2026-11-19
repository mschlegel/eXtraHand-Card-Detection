package com.extrahand.cards.core.table

import com.extrahand.cards.core.CardImage
import com.extrahand.cards.core.LumaGridV1
import kotlin.math.floor
import kotlin.math.max
import kotlin.math.min

/**
 * Per-pixel planes of a full-resolution frame, in the same units [LumaGridV1] uses:
 * luma in [0,1] (BT.601) and "redness" = max(0, r - max(g,b) - 0.06).
 * Patches are sampled from these so far-away cards keep every pixel the camera captured.
 */
class FramePlanes(val width: Int, val height: Int, val luma: FloatArray, val red: FloatArray) {

    /** 2x box-downsampled copy (lazily built), so big patches can be sampled with few taps. */
    private val half: FramePlanes? by lazy {
        if (width < 16 || height < 16) null else {
            val w2 = width / 2; val h2 = height / 2
            fun down(p: FloatArray) = FloatArray(w2 * h2) { i ->
                val x = (i % w2) * 2; val y = (i / w2) * 2; val o = y * width + x
                (p[o] + p[o + 1] + p[o + width] + p[o + width + 1]) * 0.25f
            }
            FramePlanes(w2, h2, down(luma), down(red))
        }
    }

    fun bilinear(plane: FloatArray, x: Float, y: Float): Float {
        val xf = x.coerceIn(0f, width - 1.001f); val yf = y.coerceIn(0f, height - 1.001f)
        val x0 = floor(xf).toInt(); val y0 = floor(yf).toInt()
        val fx = xf - x0; val fy = yf - y0
        val i = y0 * width + x0
        val a = plane[i]; val b = plane[i + 1]; val c = plane[i + width]; val d = plane[i + width + 1]
        return (a * (1 - fx) + b * fx) * (1 - fy) + (c * (1 - fx) + d * fx) * fy
    }

    /** True if (x, y) is inside the frame. */
    fun contains(x: Float, y: Float) = x >= 0 && y >= 0 && x <= width - 1 && y <= height - 1

    /**
     * Sample an oriented rectangle into a [gw] x [gh] grid of area averages. The rectangle starts at [origin]
     * and spans [ax] (one full grid width) and [ay] (one full grid height). Returns (luma, red) grids,
     * or null if more than 25% of the rectangle falls outside the frame.
     */
    fun sampleGrid(origin: Pt, ax: Pt, ay: Pt, gw: Int, gh: Int, red: Boolean): FloatArray? {
        val cellPx = max(ax.length / gw, ay.length / gh)
        // Big cells: sample a half-resolution level instead (2x2 taps per cell is then enough).
        if (cellPx > 3f) half?.let { return it.sampleGrid(origin * 0.5f - Pt(0.25f, 0.25f), ax * 0.5f, ay * 0.5f, gw, gh, red) }
        val plane = if (red) this.red else luma
        val k = if (cellPx > 1.5f) 2 else 1
        val out = FloatArray(gw * gh)
        var outside = 0
        for (j in 0 until gh) for (i in 0 until gw) {
            var s = 0f
            for (sy in 0 until k) for (sx in 0 until k) {
                val u = (i + (sx + 0.5f) / k) / gw
                val v = (j + (sy + 0.5f) / k) / gh
                val x = origin.x + ax.x * u + ay.x * v
                val y = origin.y + ax.y * u + ay.y * v
                if (!contains(x, y)) outside++
                s += bilinear(plane, x, y)
            }
            out[j * gw + i] = s / (k * k)
        }
        return if (outside > gw * gh * k * k / 4) null else out
    }

    /** Rectify the quad (p0 = card corner, p0->p1 = width) into an upright [w] x [h] image. */
    fun warp(src: CardImage, quad: Quad, w: Int = 2 * LumaGridV1.CANON_W, h: Int = 2 * LumaGridV1.CANON_H): CardImage? {
        val hgr = Homography.fromPoints(
            listOf(Pt(0f, 0f), Pt(w.toFloat(), 0f), Pt(w.toFloat(), h.toFloat()), Pt(0f, h.toFloat())),
            quad.points,
        ) ?: return null
        val px = IntArray(w * h)
        for (y in 0 until h) for (x in 0 until w) {
            val p = hgr.map(x + 0.5f, y + 0.5f)
            px[y * w + x] = bilinearArgb(src, p.x - 0.5f, p.y - 0.5f)
        }
        return CardImage(w, h, px)
    }

    companion object {
        fun of(img: CardImage): FramePlanes {
            val n = img.width * img.height
            val luma = FloatArray(n); val red = FloatArray(n)
            for (i in 0 until n) {
                val p = img.argb[i]
                val r = (p shr 16 and 0xFF) / 255f; val g = (p shr 8 and 0xFF) / 255f; val b = (p and 0xFF) / 255f
                luma[i] = 0.299f * r + 0.587f * g + 0.114f * b
                red[i] = max(0f, r - max(g, b) - LumaGridV1.RED_OFFSET)
            }
            return FramePlanes(img.width, img.height, luma, red)
        }

        internal fun bilinearArgb(img: CardImage, x: Float, y: Float): Int {
            val xf = x.coerceIn(0f, img.width - 1.001f); val yf = y.coerceIn(0f, img.height - 1.001f)
            val x0 = floor(xf).toInt(); val y0 = floor(yf).toInt()
            val x1 = min(x0 + 1, img.width - 1); val y1 = min(y0 + 1, img.height - 1)
            val fx = xf - x0; val fy = yf - y0
            val a = img.argb[y0 * img.width + x0]; val b = img.argb[y0 * img.width + x1]
            val c = img.argb[y1 * img.width + x0]; val d = img.argb[y1 * img.width + x1]
            fun ch(s: Int): Int {
                val top = (a shr s and 0xFF) * (1 - fx) + (b shr s and 0xFF) * fx
                val bot = (c shr s and 0xFF) * (1 - fx) + (d shr s and 0xFF) * fx
                return (top * (1 - fy) + bot * fy + 0.5f).toInt().coerceIn(0, 255)
            }
            return (0xFF shl 24) or (ch(16) shl 16) or (ch(8) shl 8) or ch(0)
        }
    }
}
