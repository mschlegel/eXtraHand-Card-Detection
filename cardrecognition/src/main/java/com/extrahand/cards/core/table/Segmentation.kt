package com.extrahand.cards.core.table

import com.extrahand.cards.core.CardImage
import com.extrahand.cards.core.CardLocator
import kotlin.math.max
import kotlin.math.min
import kotlin.math.sqrt

/**
 * Low-resolution segmentation of a full table frame into candidate card regions. Model-free:
 *
 *  - **face mask**: bright, low-saturation pixels (card faces and the white border most backs have),
 *    with enclosed holes (pips, indices, back patterns) filled.
 *  - **object mask**: pixels whose colour is far from the dominant table colour and that are not in the
 *    face mask. Catches borderless or dark card backs.
 *
 * Everything here works on a downscaled copy; callers map coordinates back with [scale].
 */
internal class Segmentation private constructor(
    val w: Int, val h: Int,
    /** Full-res pixels per working pixel. */
    val scale: Float,
    val face: BooleanArray,
    val obj: BooleanArray,
    /** Raw bright-and-neutral pixels before morphology/hole filling. Inside a face blob, !bright = ink. */
    val bright: BooleanArray,
    /** Mean saturation of working pixels, 0..1 (exposed for the back heuristic). */
    val saturation: FloatArray,
) {
    inner class Component(val area: Int, val box: Box, val contour: List<Pt>, private val label: IntArray, private val id: Int) {
        /** Is the working-image point [p] part of this component? */
        fun contains(p: Pt): Boolean {
            val x = p.x.toInt(); val y = p.y.toInt()
            return x in 0 until w && y in 0 until h && label[y * w + x] == id
        }
    }

    /** An ink cluster (index glyphs, pips) inside a component, with its principal axes. */
    class Ink(val center: Pt, val majorDir: Pt, val majorLen: Float, val minorLen: Float, val area: Int)

    /**
     * Ink clusters inside [c]: connected non-bright pixels, merged when closer than [mergeGap] working px.
     * Clusters bigger than [maxArea] (court-card art) are dropped.
     */
    fun inkClusters(c: Component, mergeGap: Float, maxArea: Int): List<Ink> {
        val bx0 = c.box.left.toInt(); val by0 = c.box.top.toInt()
        val bw = c.box.width.toInt(); val bh = c.box.height.toInt()
        val lab = IntArray(bw * bh) { -1 }
        fun isInk(x: Int, y: Int): Boolean {
            val i = (by0 + y) * w + bx0 + x
            return !bright[i] && c.contains(Pt(bx0 + x + 0.5f, by0 + y + 0.5f))
        }
        class Blob(var minX: Int, var minY: Int, var maxX: Int, var maxY: Int, val pts: ArrayList<Int>)
        val blobs = ArrayList<Blob>()
        val stack = IntArray(bw * bh)
        for (y0 in 0 until bh) for (x0 in 0 until bw) {
            val s0 = y0 * bw + x0
            if (lab[s0] >= 0 || !isInk(x0, y0)) continue
            val b = Blob(x0, y0, x0, y0, ArrayList())
            var sp = 0; stack[sp++] = s0; lab[s0] = blobs.size
            while (sp > 0) {
                val p = stack[--sp]; b.pts.add(p)
                val x = p % bw; val y = p / bw
                b.minX = min(b.minX, x); b.maxX = max(b.maxX, x); b.minY = min(b.minY, y); b.maxY = max(b.maxY, y)
                for (dy in -1..1) for (dx in -1..1) {
                    val nx = x + dx; val ny = y + dy
                    if (nx < 0 || ny < 0 || nx >= bw || ny >= bh) continue
                    val q = ny * bw + nx
                    if (lab[q] < 0 && isInk(nx, ny)) { lab[q] = blobs.size; stack[sp++] = q }
                }
            }
            if (b.pts.size <= maxArea) blobs.add(b) else b.pts.forEach { lab[it] = Int.MAX_VALUE }
        }
        // Union blobs whose boxes are within mergeGap.
        val parent = IntArray(blobs.size) { it }
        fun find(i: Int): Int { var x = i; while (parent[x] != x) { parent[x] = parent[parent[x]]; x = parent[x] }; return x }
        for (i in blobs.indices) for (j in i + 1 until blobs.size) {
            val a = blobs[i]; val b = blobs[j]
            val dx = max(0, max(a.minX, b.minX) - min(a.maxX, b.maxX) - 1)
            val dy = max(0, max(a.minY, b.minY) - min(a.maxY, b.maxY) - 1)
            if (dx <= mergeGap && dy <= mergeGap) parent[find(i)] = find(j)
        }
        return blobs.indices.groupBy { find(it) }.values.mapNotNull { idx ->
            val pts = idx.flatMap { blobs[it].pts }
            if (pts.size > maxArea || pts.size < 4) return@mapNotNull null
            var mx = 0.0; var my = 0.0
            for (p in pts) { mx += p % bw; my += p / bw }
            mx /= pts.size; my /= pts.size
            var sxx = 0.0; var syy = 0.0; var sxy = 0.0
            for (p in pts) { val dx = p % bw - mx; val dy = p / bw - my; sxx += dx * dx; syy += dy * dy; sxy += dx * dy }
            sxx /= pts.size; syy /= pts.size; sxy /= pts.size
            val tr = sxx + syy; val det = sxx * syy - sxy * sxy
            val disc = kotlin.math.sqrt(max(0.0, tr * tr / 4 - det))
            val l1 = tr / 2 + disc; val l2 = max(0.0, tr / 2 - disc)
            val ang = 0.5 * kotlin.math.atan2(2 * sxy, sxx - syy)
            Ink(
                Pt((bx0 + mx + 0.5).toFloat(), (by0 + my + 0.5).toFloat()),
                Pt(kotlin.math.cos(ang).toFloat(), kotlin.math.sin(ang).toFloat()),
                (kotlin.math.sqrt(12 * l1) + 1).toFloat(), (kotlin.math.sqrt(12 * l2) + 1).toFloat(), pts.size,
            )
        }
    }

    fun faceComponents(minArea: Int) = components(face, minArea)
    fun objectComponents(minArea: Int) = components(obj, minArea)

    private fun components(mask: BooleanArray, minArea: Int): List<Component> {
        val label = IntArray(w * h) { -1 }
        val out = ArrayList<Component>()
        val stack = IntArray(w * h)
        var next = 0
        for (start in mask.indices) {
            if (!mask[start] || label[start] >= 0) continue
            var sp = 0; stack[sp++] = start; label[start] = next
            var count = 0
            var minX = w; var minY = h; var maxX = 0; var maxY = 0
            while (sp > 0) {
                val p = stack[--sp]; count++
                val x = p % w; val y = p / w
                minX = min(minX, x); maxX = max(maxX, x); minY = min(minY, y); maxY = max(maxY, y)
                if (x > 0 && mask[p - 1] && label[p - 1] < 0) { label[p - 1] = next; stack[sp++] = p - 1 }
                if (x < w - 1 && mask[p + 1] && label[p + 1] < 0) { label[p + 1] = next; stack[sp++] = p + 1 }
                if (y > 0 && mask[p - w] && label[p - w] < 0) { label[p - w] = next; stack[sp++] = p - w }
                if (y < h - 1 && mask[p + w] && label[p + w] < 0) { label[p + w] = next; stack[sp++] = p + w }
            }
            if (count >= minArea) {
                val contour = traceContour(label, next, start)
                out.add(Component(count, Box(minX.toFloat(), minY.toFloat(), maxX + 1f, maxY + 1f), contour, label, next))
            }
            next++
        }
        return out
    }

    /** Moore-neighbour boundary trace starting from [start] (the first pixel of the component in raster order). */
    private fun traceContour(label: IntArray, id: Int, start: Int): List<Pt> {
        val dx = intArrayOf(1, 1, 0, -1, -1, -1, 0, 1)
        val dy = intArrayOf(0, 1, 1, 1, 0, -1, -1, -1)
        fun inside(x: Int, y: Int) = x in 0 until w && y in 0 until h && label[y * w + x] == id
        val sx = start % w; val sy = start / w
        val out = ArrayList<Pt>()
        var x = sx; var y = sy
        var dir = 7 // came from the left/up; begin searching clockwise
        val limit = 4 * w * h
        var steps = 0
        do {
            out.add(Pt(x + 0.5f, y + 0.5f))
            var found = false
            var d = (dir + 6) % 8
            for (i in 0 until 8) {
                val nx = x + dx[d]; val ny = y + dy[d]
                if (inside(nx, ny)) { x = nx; y = ny; dir = d; found = true; break }
                d = (d + 1) % 8
            }
            if (!found) break
        } while ((x != sx || y != sy) && ++steps < limit)
        return out
    }

    companion object {
        /** Longest side of the working image. */
        const val WORK_MAX = 960

        fun of(img: CardImage, maxSide: Int = WORK_MAX): Segmentation {
            val s = max(1f, max(img.width, img.height).toFloat() / maxSide)
            val w = max(8, (img.width / s).toInt()); val h = max(8, (img.height / s).toInt())
            val r = FloatArray(w * h); val g = FloatArray(w * h); val b = FloatArray(w * h)
            // Area-average downscale.
            for (j in 0 until h) {
                val y0 = (j * s).toInt(); val y1 = max(y0 + 1, min(img.height, ((j + 1) * s).toInt()))
                for (i in 0 until w) {
                    val x0 = (i * s).toInt(); val x1 = max(x0 + 1, min(img.width, ((i + 1) * s).toInt()))
                    var sr = 0f; var sg = 0f; var sb = 0f
                    for (y in y0 until y1) for (x in x0 until x1) {
                        val p = img.argb[y * img.width + x]
                        sr += p shr 16 and 0xFF; sg += p shr 8 and 0xFF; sb += p and 0xFF
                    }
                    val n = ((y1 - y0) * (x1 - x0)).toFloat()
                    r[j * w + i] = sr / n; g[j * w + i] = sg / n; b[j * w + i] = sb / n
                }
            }
            val n = w * h
            val luma = FloatArray(n) { 0.299f * r[it] + 0.587f * g[it] + 0.114f * b[it] }
            val sat = FloatArray(n) {
                val mx = max(r[it], max(g[it], b[it])); val mn = min(r[it], min(g[it], b[it]))
                if (mx < 1f) 0f else (mx - mn) / mx
            }

            // Face mask: above the Otsu split AND fairly neutral, with an absolute floor so a dim frame of only
            // table does not get split in half.
            val t = max(CardLocator.otsu(luma), 90f)
            val bright = BooleanArray(n) { luma[it] > t && sat[it] < 0.30f }
            val face = bright.copyOf()
            // Close first so index ink touching the card edge (common when cards are small) cannot leak the
            // outside into the card, then fill enclosed ink/pattern, then drop specks.
            close3(face, w, h)
            fillHoles(face, w, h)
            open3(face, w, h)

            // Object mask: far from the dominant (table) colour.
            val bg = dominantColor(r, g, b, face)
            val obj = BooleanArray(n) {
                if (face[it]) false else {
                    val d = sqrt(sq(r[it] - bg[0]) + sq(g[it] - bg[1]) + sq(b[it] - bg[2]))
                    d > 60f
                }
            }
            close3(obj, w, h)
            fillHoles(obj, w, h)
            open3(obj, w, h)
            return Segmentation(w, h, s, face, obj, bright, sat)
        }

        /** Mean RGB of the most common 3-bit-per-channel colour bin outside [exclude]. */
        private fun dominantColor(r: FloatArray, g: FloatArray, b: FloatArray, exclude: BooleanArray): FloatArray {
            val count = IntArray(512)
            fun bin(i: Int) = ((r[i].toInt() shr 5) shl 6) or ((g[i].toInt() shr 5) shl 3) or (b[i].toInt() shr 5)
            for (i in r.indices) if (!exclude[i]) count[bin(i)]++
            val mode = count.indices.maxBy { count[it] }
            var sr = 0.0; var sg = 0.0; var sb = 0.0; var c = 0
            for (i in r.indices) if (!exclude[i] && bin(i) == mode) { sr += r[i]; sg += g[i]; sb += b[i]; c++ }
            if (c == 0) return floatArrayOf(0f, 0f, 0f)
            return floatArrayOf((sr / c).toFloat(), (sg / c).toFloat(), (sb / c).toFloat())
        }

        /** 3x3 (cross) morphological closing: dilate then erode. Seals 1-px gaps. */
        private fun close3(m: BooleanArray, w: Int, h: Int) {
            val d = BooleanArray(m.size)
            for (y in 0 until h) for (x in 0 until w) {
                val i = y * w + x
                d[i] = m[i] || (x > 0 && m[i - 1]) || (x < w - 1 && m[i + 1]) || (y > 0 && m[i - w]) || (y < h - 1 && m[i + w])
            }
            for (y in 0 until h) for (x in 0 until w) {
                val i = y * w + x
                m[i] = d[i] && (x == 0 || d[i - 1]) && (x == w - 1 || d[i + 1]) && (y == 0 || d[i - w]) && (y == h - 1 || d[i + w])
            }
        }

        /** 3x3 morphological opening (erode then dilate): removes specks and 1-px bridges. */
        private fun open3(m: BooleanArray, w: Int, h: Int) {
            val e = BooleanArray(m.size)
            for (y in 1 until h - 1) for (x in 1 until w - 1) {
                val i = y * w + x
                e[i] = m[i] && m[i - 1] && m[i + 1] && m[i - w] && m[i + w]
            }
            java.util.Arrays.fill(m, false)
            for (y in 1 until h - 1) for (x in 1 until w - 1) {
                val i = y * w + x
                if (e[i] || e[i - 1] || e[i + 1] || e[i - w] || e[i + w]) m[i] = true
            }
        }

        /** Set every background pixel not 4-connected to the frame border (i.e. enclosed holes). */
        private fun fillHoles(m: BooleanArray, w: Int, h: Int) {
            val outside = BooleanArray(m.size)
            val stack = IntArray(m.size)
            var sp = 0
            fun seed(i: Int) { if (!m[i] && !outside[i]) { outside[i] = true; stack[sp++] = i } }
            for (x in 0 until w) { seed(x); seed((h - 1) * w + x) }
            for (y in 0 until h) { seed(y * w); seed(y * w + w - 1) }
            while (sp > 0) {
                val p = stack[--sp]; val x = p % w; val y = p / w
                if (x > 0) seed(p - 1); if (x < w - 1) seed(p + 1)
                if (y > 0) seed(p - w); if (y < h - 1) seed(p + w)
            }
            for (i in m.indices) if (!m[i] && !outside[i]) m[i] = true
        }
    }
}
