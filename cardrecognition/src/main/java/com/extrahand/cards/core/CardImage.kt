package com.extrahand.cards.core

/**
 * Platform-neutral image: packed ARGB_8888 pixels (0xAARRGGBB), row-major.
 * On Android convert with `Bitmap.toCardImage()` (see com.extrahand.cards.android).
 */
class CardImage(val width: Int, val height: Int, val argb: IntArray) {
    init {
        require(width > 0 && height > 0) { "empty image" }
        require(argb.size == width * height) { "pixel array size mismatch" }
    }

    fun pixel(x: Int, y: Int): Int = argb[y * width + x]

    /** Crop to the given integer rectangle (clamped to bounds). */
    fun crop(left: Int, top: Int, right: Int, bottom: Int): CardImage {
        val l = left.coerceIn(0, width - 1)
        val t = top.coerceIn(0, height - 1)
        val r = right.coerceIn(l + 1, width)
        val b = bottom.coerceIn(t + 1, height)
        val w = r - l
        val h = b - t
        val out = IntArray(w * h)
        for (y in 0 until h) System.arraycopy(argb, (t + y) * width + l, out, y * w, w)
        return CardImage(w, h, out)
    }

    /** Crop using fractional coordinates in [0,1]. */
    fun cropFraction(rect: NormRect): CardImage = crop(
        (rect.left * width).toInt(), (rect.top * height).toInt(),
        (rect.right * width).toInt(), (rect.bottom * height).toInt()
    )

    /** Rotate 90 degrees clockwise. */
    fun rotate90(): CardImage {
        val out = IntArray(width * height)
        val nw = height
        for (y in 0 until height) for (x in 0 until width) {
            // (x, y) -> (nw - 1 - y, x)
            out[x * nw + (nw - 1 - y)] = argb[y * width + x]
        }
        return CardImage(nw, width, out)
    }
}

/** Rectangle in normalized [0,1] coordinates. */
data class NormRect(val left: Float, val top: Float, val right: Float, val bottom: Float) {
    val width get() = right - left
    val height get() = bottom - top

    companion object {
        /** Standard poker card aspect ratio (width / height) = 63mm / 88mm. */
        const val CARD_ASPECT = 63f / 88f

        /**
         * A centered, card-shaped guide box for a frame with aspect [frameAspect] (= width / height of the
         * upright frame). The box is [heightFraction] of the frame height (shrunk if it would not fit).
         */
        fun centeredCard(frameAspect: Float, heightFraction: Float = 0.8f): NormRect {
            var h = heightFraction
            var w = h * CARD_ASPECT / frameAspect
            if (w > 0.95f) { w = 0.95f; h = w * frameAspect / CARD_ASPECT }
            return NormRect((1 - w) / 2, (1 - h) / 2, (1 + w) / 2, (1 + h) / 2)
        }
    }
}
