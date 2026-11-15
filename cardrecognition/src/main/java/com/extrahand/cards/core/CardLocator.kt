package com.extrahand.cards.core

/**
 * Tightens a rough crop (e.g. the on-screen guide box) to the bright card face inside it.
 *
 * Cheap heuristic, no ML: Otsu-threshold the luma, then find the rows/columns where most pixels are
 * "bright". If the result does not look like a card (too small or wrong aspect) the input is returned
 * unchanged, so this is always safe to call.
 */
object CardLocator {
    private const val WORK_H = 96
    private const val LINE_FILL = 0.55f
    private const val MIN_AREA_FRACTION = 0.30f

    data class Located(val image: CardImage, val rect: NormRect, val refined: Boolean)

    fun locate(rough: CardImage): Located {
        val full = NormRect(0f, 0f, 1f, 1f)
        val scale = rough.height.toFloat() / WORK_H
        val ww = (rough.width / scale).toInt().coerceAtLeast(8)
        val wh = WORK_H
        val src = FloatArray(rough.width * rough.height) {
            val p = rough.argb[it]
            0.299f * (p shr 16 and 0xFF) + 0.587f * (p shr 8 and 0xFF) + 0.114f * (p and 0xFF)
        }
        val small = LumaGridV1.Grid(rough.width, rough.height, src)
            .resample(0, 0, rough.width, rough.height, ww, wh).data
        val t = otsu(small)
        val bright = BooleanArray(small.size) { small[it] > t }

        fun colFill(x: Int) = (0 until wh).count { bright[it * ww + x] }.toFloat() / wh
        fun rowFill(y: Int) = (0 until ww).count { bright[y * ww + it] }.toFloat() / ww
        val cols = (0 until ww).filter { colFill(it) >= LINE_FILL }
        val rows = (0 until wh).filter { rowFill(it) >= LINE_FILL }
        if (cols.isEmpty() || rows.isEmpty()) return Located(rough, full, false)

        val rect = NormRect(
            cols.first().toFloat() / ww, rows.first().toFloat() / wh,
            (cols.last() + 1).toFloat() / ww, (rows.last() + 1).toFloat() / wh
        )
        val area = rect.width * rect.height
        val aspect = (rect.width * rough.width) / (rect.height * rough.height)
        val portraitAspect = if (aspect > 1f) 1f / aspect else aspect
        val looksLikeCard = area >= MIN_AREA_FRACTION &&
            portraitAspect in (NormRect.CARD_ASPECT * 0.8f)..(NormRect.CARD_ASPECT * 1.2f)
        if (!looksLikeCard) return Located(rough, full, false)
        return Located(rough.cropFraction(rect), rect, true)
    }

    /** Otsu threshold over values in [0,255]. */
    internal fun otsu(v: FloatArray): Float {
        val hist = IntArray(256)
        for (x in v) hist[x.toInt().coerceIn(0, 255)]++
        val total = v.size
        var sum = 0.0
        for (i in 0..255) sum += i * hist[i].toDouble()
        var sumB = 0.0
        var wB = 0
        var best = 0.0
        var threshold = 127
        for (i in 0..255) {
            wB += hist[i]
            if (wB == 0) continue
            val wF = total - wB
            if (wF == 0) break
            sumB += i * hist[i].toDouble()
            val mB = sumB / wB
            val mF = (sum - sumB) / wF
            val between = wB.toDouble() * wF * (mB - mF) * (mB - mF)
            if (between > best) { best = between; threshold = i }
        }
        return threshold.toFloat()
    }
}
