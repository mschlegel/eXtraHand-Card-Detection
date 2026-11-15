package com.extrahand.cards.core

/** Card rank. [code] is the canonical label fragment used in dataset files. */
enum class Rank(val code: String, val display: String) {
    ACE("A", "A"), TWO("2", "2"), THREE("3", "3"), FOUR("4", "4"), FIVE("5", "5"),
    SIX("6", "6"), SEVEN("7", "7"), EIGHT("8", "8"), NINE("9", "9"), TEN("10", "10"),
    JACK("J", "J"), QUEEN("Q", "Q"), KING("K", "K");

    companion object {
        fun fromCode(code: String): Rank? = entries.firstOrNull { it.code.equals(code, ignoreCase = true) }
            ?: if (code.equals("T", ignoreCase = true)) TEN else null
    }
}

/** Card suit. [code] is the canonical label fragment used in dataset files. */
enum class Suit(val code: String, val symbol: String, val isRed: Boolean) {
    SPADES("S", "♠", false), HEARTS("H", "♥", true),
    DIAMONDS("D", "♦", true), CLUBS("C", "♣", false);

    companion object {
        fun fromCode(code: String): Suit? = entries.firstOrNull { it.code.equals(code, ignoreCase = true) }
    }
}

/**
 * A playing card. Labels are strings like "AS", "10H", "QD", "7C", and "JK" (joker).
 * The label string is the stable identifier used everywhere in the dataset.
 */
data class Card(val rank: Rank?, val suit: Suit?) {
    init {
        require((rank == null) == (suit == null)) { "rank and suit must both be set (or both null for a joker)" }
    }

    val isJoker: Boolean get() = rank == null

    /** Canonical label, e.g. "10H". */
    val label: String get() = if (isJoker) JOKER_LABEL else rank!!.code + suit!!.code

    /** Human friendly, e.g. "10♥". */
    val display: String get() = if (isJoker) "Joker" else rank!!.display + suit!!.symbol

    override fun toString(): String = label

    companion object {
        const val JOKER_LABEL = "JK"
        val JOKER = Card(null, null)

        /** All 52 standard cards (no joker), ordered by suit then rank. */
        val STANDARD_DECK: List<Card> = Suit.entries.flatMap { s -> Rank.entries.map { r -> Card(r, s) } }

        /** Parse a label such as "AS", "10h", "TD", "JK". Returns null if invalid. */
        fun parse(label: String): Card? {
            val l = label.trim().uppercase()
            if (l == JOKER_LABEL) return JOKER
            if (l.length < 2) return null
            val suit = Suit.fromCode(l.takeLast(1)) ?: return null
            val rank = Rank.fromCode(l.dropLast(1)) ?: return null
            return Card(rank, suit)
        }

        fun of(label: String): Card = parse(label) ?: throw IllegalArgumentException("Invalid card label: $label")
    }
}
