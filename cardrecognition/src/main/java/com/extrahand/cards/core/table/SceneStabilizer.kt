package com.extrahand.cards.core.table

import com.extrahand.cards.core.Card

/**
 * Smooths live [TableScene]s. A card is reported once it shows up in at least [minHits] of the last [window]
 * frames, and it stays until it misses that many. This stops the overlay from flickering when a corner is
 * read only on some frames. Not thread-safe: feed it from a single analyzer thread.
 */
class SceneStabilizer(private val window: Int = 4, private val minHits: Int = 2) {
    private val history = ArrayDeque<TableScene>()

    fun reset() = history.clear()

    fun update(scene: TableScene): TableScene {
        history.addLast(scene)
        while (history.size > window) history.removeFirst()
        if (history.size == 1 && window > 1) return scene.copy(groups = emptyList(), backs = emptyList())

        val hits = HashMap<Card, Int>()
        val latest = HashMap<Card, SeenCard>()
        for (s in history) for (c in s.cards.distinctBy { it.card }) {
            hits.merge(c.card, 1, Int::plus)
            latest[c.card] = c
        }
        val stable = hits.filterValues { it >= minHits }.keys

        // Keep the newest frame's grouping; re-attach stable cards it missed to the group that contains them.
        val groups = scene.groups.map { g -> g.copy(cards = g.cards.filter { it.card in stable }) }.toMutableList()
        val present = groups.flatMap { it.cards }.map { it.card }.toSet()
        for (card in stable - present) {
            val seen = latest[card] ?: continue
            val ctr = seen.quad.center
            val gi = groups.indexOfFirst { ctr.x >= it.box.left && ctr.x <= it.box.right && ctr.y >= it.box.top && ctr.y <= it.box.bottom }
            if (gi >= 0) groups[gi] = groups[gi].copy(cards = groups[gi].cards + seen)
            else groups.add(CardGroup(listOf(seen), Box.of(seen.quad.points), overlapping = false))
        }

        val framesWithBacks = history.count { it.backs.isNotEmpty() }
        val backs = when {
            framesWithBacks < minHits -> emptyList()
            scene.backs.isNotEmpty() -> scene.backs
            else -> history.last { it.backs.isNotEmpty() }.backs
        }
        return scene.copy(groups = groups.filter { it.cards.isNotEmpty() }, backs = backs)
    }
}
