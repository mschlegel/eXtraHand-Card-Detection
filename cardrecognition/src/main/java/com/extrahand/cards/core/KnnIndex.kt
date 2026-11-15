package com.extrahand.cards.core

import kotlin.math.exp

/**
 * In-memory k-nearest-neighbour index over L2-normalized vectors (cosine similarity via dot product).
 * Appends are cheap and thread-safe; queries run lock-free against an immutable snapshot.
 */
class KnnIndex(val dim: Int) {
    private class Snapshot(val data: FloatArray, val ids: Array<String?>, val labels: Array<String?>, val count: Int)

    @Volatile private var snap = Snapshot(FloatArray(0), arrayOfNulls(0), arrayOfNulls(0), 0)
    private val lock = Any()

    val size: Int get() = snap.count

    fun labelCounts(): Map<String, Int> {
        val s = snap
        val m = HashMap<String, Int>()
        for (i in 0 until s.count) m.merge(s.labels[i]!!, 1, Int::plus)
        return m
    }

    fun add(id: String, label: String, vector: FloatArray) {
        require(vector.size == dim) { "vector dim ${vector.size} != $dim" }
        synchronized(lock) {
            var s = snap
            if (s.count == s.ids.size) {
                val cap = maxOf(64, s.ids.size * 2)
                s = Snapshot(s.data.copyOf(cap * dim), s.ids.copyOf(cap), s.labels.copyOf(cap), s.count)
            }
            System.arraycopy(vector, 0, s.data, s.count * dim, dim)
            s.ids[s.count] = id
            s.labels[s.count] = label
            snap = Snapshot(s.data, s.ids, s.labels, s.count + 1)
        }
    }

    fun remove(id: String): Boolean = synchronized(lock) {
        val s = snap
        val idx = (0 until s.count).firstOrNull { s.ids[it] == id } ?: return false
        // Copy (rather than mutate in place) so concurrent readers of the old snapshot stay consistent.
        val data = s.data.copyOf()
        val ids = s.ids.copyOf()
        val labels = s.labels.copyOf()
        val last = s.count - 1
        if (idx != last) {
            System.arraycopy(data, last * dim, data, idx * dim, dim)
            ids[idx] = ids[last]; labels[idx] = labels[last]
        }
        ids[last] = null; labels[last] = null
        snap = Snapshot(data, ids, labels, last)
        true
    }

    fun clear() = synchronized(lock) { snap = Snapshot(FloatArray(0), arrayOfNulls(0), arrayOfNulls(0), 0) }

    data class Neighbor(val id: String, val label: String, val similarity: Float)

    fun nearest(query: FloatArray, k: Int): List<Neighbor> {
        val s = snap
        if (s.count == 0) return emptyList()
        val kk = minOf(k, s.count)
        val bestIdx = IntArray(kk) { -1 }
        val bestSim = FloatArray(kk) { Float.NEGATIVE_INFINITY }
        for (i in 0 until s.count) {
            var dot = 0f
            val base = i * dim
            for (d in 0 until dim) dot += s.data[base + d] * query[d]
            if (dot > bestSim[kk - 1]) {
                var p = kk - 1
                while (p > 0 && bestSim[p - 1] < dot) { bestSim[p] = bestSim[p - 1]; bestIdx[p] = bestIdx[p - 1]; p-- }
                bestSim[p] = dot; bestIdx[p] = i
            }
        }
        return (0 until kk).filter { bestIdx[it] >= 0 }
            .map { Neighbor(s.ids[bestIdx[it]]!!, s.labels[bestIdx[it]]!!, bestSim[it]) }
    }

    /**
     * Soft-voting classification: weight each of the k neighbours by exp((sim - bestSim) / temperature)
     * and sum per label. Returns labels sorted by probability (sums to 1 over returned labels).
     */
    fun classify(query: FloatArray, k: Int = 7, temperature: Float = 0.03f): List<LabelScore> {
        val nn = nearest(query, k)
        if (nn.isEmpty()) return emptyList()
        val top = nn.first().similarity
        val weight = HashMap<String, Float>()
        val bestSim = HashMap<String, Float>()
        for (n in nn) {
            weight.merge(n.label, exp((n.similarity - top) / temperature), Float::plus)
            bestSim.merge(n.label, n.similarity) { a, b -> maxOf(a, b) }
        }
        val total = weight.values.sum()
        return weight.entries.map { LabelScore(it.key, it.value / total, bestSim[it.key]!!) }
            .sortedByDescending { it.probability }
    }

    data class LabelScore(val label: String, val probability: Float, val similarity: Float)
}
