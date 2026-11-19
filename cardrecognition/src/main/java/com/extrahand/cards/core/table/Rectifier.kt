package com.extrahand.cards.core.table

import com.extrahand.cards.core.CardImage

/**
 * Re-frames a training crop exactly the way [TableReader] frames cards it finds: segment the card, fit its
 * outline, and warp it to an upright, edge-to-edge image.
 *
 * Trainer crops come from a guide box tightened by `CardLocator`, so they often keep a strip of table and a
 * little perspective. Features from such crops do not line up with features from rectified table cards.
 * Matching works much better when both sides are framed the same way. Returns null if no card outline is
 * found; then use the crop as-is.
 */
object Rectifier {
    fun rectify(crop: CardImage): CardImage? {
        // Pad with a dark border so a card that fills the crop is still a separate blob with clean edges.
        val pad = (maxOf(crop.width, crop.height) * 0.08f).toInt().coerceAtLeast(4)
        val w = crop.width + 2 * pad; val h = crop.height + 2 * pad
        val px = IntArray(w * h) { 0xFF101010.toInt() }
        for (y in 0 until crop.height) System.arraycopy(crop.argb, y * crop.width, px, (y + pad) * w + pad, crop.width)
        val img = CardImage(w, h, px)
        val seg = Segmentation.of(img, maxSide = 480)
        val c = seg.faceComponents(16).maxByOrNull { it.area } ?: return null
        val q = Polygons.quadFromHull(Polygons.convexHull(c.contour), c.contour) ?: return null
        val quad = q.scaled(seg.scale).cardOrdered()
        // Must cover most of the crop, else segmentation latched onto part of the card (e.g. a white panel).
        if (quad.area < crop.width * crop.height * 0.6f) return null
        return FramePlanes.of(img).warp(img, quad)
    }
}
