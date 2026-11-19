package com.extrahand.cards.core.table

import kotlin.math.abs
import kotlin.math.atan2
import kotlin.math.hypot
import kotlin.math.sqrt

/** 2-D point / vector in pixel coordinates (y grows downward, like images). */
data class Pt(val x: Float, val y: Float) {
    operator fun plus(o: Pt) = Pt(x + o.x, y + o.y)
    operator fun minus(o: Pt) = Pt(x - o.x, y - o.y)
    operator fun times(s: Float) = Pt(x * s, y * s)
    operator fun unaryMinus() = Pt(-x, -y)
    fun dot(o: Pt) = x * o.x + y * o.y
    /** z of the 3-D cross product. > 0 means [o] is clockwise from this on screen (y down). */
    fun cross(o: Pt) = x * o.y - y * o.x
    val length get() = hypot(x, y)
    fun normalized(): Pt { val l = length; return if (l < 1e-9f) Pt(0f, 0f) else Pt(x / l, y / l) }
    /** Rotated 90 degrees clockwise on screen. */
    fun perp() = Pt(-y, x)
    val angle get() = atan2(y, x)
}

/** A 4-corner polygon, corners in order around the outline. */
data class Quad(val p0: Pt, val p1: Pt, val p2: Pt, val p3: Pt) {
    val points get() = listOf(p0, p1, p2, p3)
    val center get() = Pt((p0.x + p1.x + p2.x + p3.x) / 4, (p0.y + p1.y + p2.y + p3.y) / 4)
    val area get() = abs(Polygons.signedArea(points))
    fun scaled(s: Float) = Quad(p0 * s, p1 * s, p2 * s, p3 * s)

    /** Mean lengths of the two pairs of opposite sides: (p0p1 + p2p3)/2 and (p1p2 + p3p0)/2. */
    fun sidePairs(): Pair<Float, Float> =
        ((p1 - p0).length + (p3 - p2).length) / 2 to ((p2 - p1).length + (p0 - p3).length) / 2

    /**
     * The same quad re-ordered so that p0->p1 is a SHORT side and the order is clockwise on screen.
     * Then p0 is a card corner, p0->p1 runs along the card width and p0->p3 along its height.
     */
    fun cardOrdered(): Quad {
        var q = if (Polygons.signedArea(points) < 0) Quad(p0, p3, p2, p1) else this
        val (a, b) = q.sidePairs()
        if (a > b) q = Quad(q.p1, q.p2, q.p3, q.p0)
        return q
    }
}

/** Axis-aligned box in pixel coordinates. */
data class Box(val left: Float, val top: Float, val right: Float, val bottom: Float) {
    val width get() = right - left
    val height get() = bottom - top
    val center get() = Pt((left + right) / 2, (top + bottom) / 2)
    fun union(o: Box) = Box(minOf(left, o.left), minOf(top, o.top), maxOf(right, o.right), maxOf(bottom, o.bottom))
    fun scaled(s: Float) = Box(left * s, top * s, right * s, bottom * s)

    companion object {
        fun of(points: Collection<Pt>): Box = Box(
            points.minOf { it.x }, points.minOf { it.y }, points.maxOf { it.x }, points.maxOf { it.y }
        )
    }
}

internal object Polygons {
    /** Shoelace area; positive = clockwise on screen (y down). */
    fun signedArea(p: List<Pt>): Float {
        var s = 0.0
        for (i in p.indices) {
            val a = p[i]; val b = p[(i + 1) % p.size]
            s += a.x.toDouble() * b.y - b.x.toDouble() * a.y
        }
        return (s / 2).toFloat()
    }

    /** Andrew's monotone chain. Returns the hull clockwise on screen. */
    fun convexHull(points: List<Pt>): List<Pt> {
        if (points.size < 3) return points
        val s = points.sortedWith(compareBy({ it.x }, { it.y }))
        fun cross(o: Pt, a: Pt, b: Pt) = (a - o).cross(b - o)
        val lower = ArrayList<Pt>()
        for (p in s) { while (lower.size >= 2 && cross(lower[lower.size - 2], lower.last(), p) <= 0) lower.removeAt(lower.size - 1); lower.add(p) }
        val upper = ArrayList<Pt>()
        for (p in s.asReversed()) { while (upper.size >= 2 && cross(upper[upper.size - 2], upper.last(), p) <= 0) upper.removeAt(upper.size - 1); upper.add(p) }
        lower.removeAt(lower.size - 1); upper.removeAt(upper.size - 1)
        val hull = lower + upper
        return if (signedArea(hull) < 0) hull.asReversed() else hull
    }

    /**
     * Reduce a convex polygon to 4 vertices by repeatedly dropping the vertex whose removal loses the least
     * area, then snap each side to the hull points it covers (helps with rounded card corners).
     */
    fun quadFromHull(hull: List<Pt>, outline: List<Pt> = hull): Quad? {
        if (hull.size < 4) return null
        val v = hull.toMutableList()
        while (v.size > 4) {
            var best = -1; var bestLoss = Float.MAX_VALUE
            for (i in v.indices) {
                val a = v[(i - 1 + v.size) % v.size]; val b = v[i]; val c = v[(i + 1) % v.size]
                val loss = abs((b - a).cross(c - a)) / 2
                if (loss < bestLoss) { bestLoss = loss; best = i }
            }
            v.removeAt(best)
        }
        // Two passes: a loose one (the reduced vertices can sit well inside rounded corners), then a tight one.
        val q0 = Quad(v[0], v[1], v[2], v[3])
        return refineQuad(refineQuad(q0, outline, 0.10f, 2f), outline, 0.03f, 1.5f)
    }

    /**
     * Fit a line to the outline points along the middle of each side and intersect neighbouring lines. This
     * recovers the sharp corners of a rounded-corner card (the reduced hull vertices sit on the arcs).
     */
    private fun refineQuad(q: Quad, outline: List<Pt>, tolFrac: Float, tolPx: Float): Quad {
        val corners = q.points
        val lines = (0 until 4).map { i ->
            val a = corners[i]; val b = corners[(i + 1) % 4]
            val d = (b - a).normalized(); val len = (b - a).length
            val near = outline.filter { p ->
                val t = (p - a).dot(d); val off = abs((p - a).cross(d))
                t > len * 0.15f && t < len * 0.85f && off < len * tolFrac + tolPx
            }
            if (near.size >= 2) fitLine(near) ?: Line(a, d) else Line(a, d)
        }
        val out = (0 until 4).map { i -> lines[(i + 3) % 4].intersect(lines[i]) ?: corners[i] }
        // Reject a refinement that moved a corner implausibly far.
        val diag = (q.p2 - q.p0).length
        for (i in 0 until 4) if ((out[i] - corners[i]).length > diag * 0.15f) return q
        return Quad(out[0], out[1], out[2], out[3])
    }

    /** Douglas-Peucker simplification of a closed contour. */
    fun simplifyClosed(c: List<Pt>, eps: Float): List<Int> {
        if (c.size < 4) return c.indices.toList()
        // Split at the two mutually farthest-ish points so the closed curve becomes two open ones.
        val a = 0
        val b = c.indices.maxBy { (c[it] - c[a]).length }
        val first = simplifyOpen(c, a, b, eps)
        val second = simplifyOpen(c + c[0], b, c.size, eps).map { if (it == c.size) 0 else it }
        return (first.dropLast(1) + second.dropLast(1)).distinct()
    }

    private fun simplifyOpen(c: List<Pt>, from: Int, to: Int, eps: Float): List<Int> {
        val keep = BooleanArray(to - from + 1)
        keep[0] = true; keep[keep.size - 1] = true
        val stack = ArrayDeque<Pair<Int, Int>>()
        stack.addLast(from to to)
        while (stack.isNotEmpty()) {
            val (i, j) = stack.removeLast()
            if (j - i < 2) continue
            val a = c[i]; val d = (c[j] - a).normalized()
            var best = -1; var bestD = eps
            for (k in i + 1 until j) {
                val dist = if (d.length == 0f) (c[k] - a).length else abs((c[k] - a).cross(d))
                if (dist > bestD) { bestD = dist; best = k }
            }
            if (best >= 0) { keep[best - from] = true; stack.addLast(i to best); stack.addLast(best to j) }
        }
        return keep.indices.filter { keep[it] }.map { it + from }
    }

    data class Line(val p: Pt, val d: Pt) {
        fun intersect(o: Line): Pt? {
            val den = d.cross(o.d)
            if (abs(den) < 1e-4f) return null
            val t = (o.p - p).cross(o.d) / den
            return p + d * t
        }
    }

    /** Total-least-squares line through points. */
    fun fitLine(pts: List<Pt>): Line? {
        if (pts.size < 2) return null
        val mx = pts.sumOf { it.x.toDouble() } / pts.size
        val my = pts.sumOf { it.y.toDouble() } / pts.size
        var sxx = 0.0; var syy = 0.0; var sxy = 0.0
        for (p in pts) { val dx = p.x - mx; val dy = p.y - my; sxx += dx * dx; syy += dy * dy; sxy += dx * dy }
        val theta = 0.5 * atan2(2 * sxy, sxx - syy)
        return Line(Pt(mx.toFloat(), my.toFloat()), Pt(kotlin.math.cos(theta).toFloat(), kotlin.math.sin(theta).toFloat()))
    }
}

/** 3x3 projective transform, row-major [h0..h8]. */
class Homography(private val h: DoubleArray) {
    fun map(x: Float, y: Float): Pt {
        val w = h[6] * x + h[7] * y + h[8]
        return Pt(((h[0] * x + h[1] * y + h[2]) / w).toFloat(), ((h[3] * x + h[4] * y + h[5]) / w).toFloat())
    }

    companion object {
        /** Homography mapping src[i] -> dst[i] (4 point pairs), or null if degenerate. */
        fun fromPoints(src: List<Pt>, dst: List<Pt>): Homography? {
            require(src.size == 4 && dst.size == 4)
            val a = Array(8) { DoubleArray(9) }
            for (i in 0 until 4) {
                val (x, y) = src[i].x.toDouble() to src[i].y.toDouble()
                val (u, v) = dst[i].x.toDouble() to dst[i].y.toDouble()
                a[2 * i] = doubleArrayOf(x, y, 1.0, 0.0, 0.0, 0.0, -u * x, -u * y, u)
                a[2 * i + 1] = doubleArrayOf(0.0, 0.0, 0.0, x, y, 1.0, -v * x, -v * y, v)
            }
            // Gaussian elimination with partial pivoting on the 8x8 system (last column = rhs).
            for (col in 0 until 8) {
                val piv = (col until 8).maxBy { abs(a[it][col]) }
                if (abs(a[piv][col]) < 1e-10) return null
                val t = a[col]; a[col] = a[piv]; a[piv] = t
                for (r in 0 until 8) if (r != col) {
                    val f = a[r][col] / a[col][col]
                    if (f != 0.0) for (c in col until 9) a[r][c] -= f * a[col][c]
                }
            }
            val sol = DoubleArray(9) { if (it < 8) a[it][8] / a[it][it] else 1.0 }
            return Homography(sol)
        }
    }
}

internal fun sq(x: Float) = x * x
internal fun dist(a: Pt, b: Pt) = sqrt(sq(a.x - b.x) + sq(a.y - b.y))
