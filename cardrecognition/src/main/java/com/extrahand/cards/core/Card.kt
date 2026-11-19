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

/** Cards without a rank and suit. [label] is the dataset label. */
enum class Special(val label: String, val display: String) {
    /** A joker whose high/low identity was not recorded (older samples, or decks with identical jokers). */
    JOKER("JK", "Joker"),
    /** The high joker, also called the big joker (usually the coloured one). */
    JOKER_HIGH("JKH", "High joker"),
    /** The low joker, also called the small or little joker (usually the black-and-white one). */
    JOKER_LOW("JKL", "Low joker"),
    /** The back of a card, i.e. a face-down card. */
    BACK("BK", "Card back");

    val isJoker get() = this != BACK
}

/**
 * A playing card. Labels are strings like "AS", "10H", "QD", "7C", the jokers "JKH" (high/big), "JKL"
 * (low/small) and "JK" (joker, unspecified), and "BK" (the back of a card, i.e. a face-down card). The label
 * string is the stable identifier used everywhere in the dataset.
 *
 * Exactly one of (rank + suit) or [special] is set. `Card(null, null)` is still the unspecified joker.
 */
data class Card(val rank: Rank?, val suit: Suit?, val special: Special? = if (rank == null) Special.JOKER else null) {
    init {
        require((rank == null) == (suit == null)) { "rank and suit must both be set (or both null for a special card)" }
        require((rank == null) == (special != null)) { "a card is either rank+suit or special" }
    }

    /** Any joker: high, low or unspecified. */
    val isJoker: Boolean get() = special?.isJoker == true
    val isBack: Boolean get() = special == Special.BACK

    /** True for the 52 ranked cards and jokers, i.e. anything that is not a back. */
    val isFace: Boolean get() = !isBack

    /** Canonical label, e.g. "10H". */
    val label: String get() = special?.label ?: (rank!!.code + suit!!.code)

    /** Human friendly, e.g. "10♥". */
    val display: String get() = special?.display ?: (rank!!.display + suit!!.symbol)

    override fun toString(): String = label

    companion object {
        const val JOKER_LABEL = "JK"
        const val JOKER_HIGH_LABEL = "JKH"
        const val JOKER_LOW_LABEL = "JKL"
        const val BACK_LABEL = "BK"
        val JOKER = Card(null, null, Special.JOKER)
        val JOKER_HIGH = Card(null, null, Special.JOKER_HIGH)
        val JOKER_LOW = Card(null, null, Special.JOKER_LOW)

        /** The back of a card. Train it like any card so face-down cards and draw piles can be detected. */
        val BACK = Card(null, null, Special.BACK)

        /** All 52 standard cards (no joker), ordered by suit then rank. */
        val STANDARD_DECK: List<Card> = Suit.entries.flatMap { s -> Rank.entries.map { r -> Card(r, s) } }

        /** Parse a label such as "AS", "10h", "TD", "JK", "JKH", "BK". Returns null if invalid. */
        fun parse(label: String): Card? {
            val l = label.trim().uppercase()
            Special.entries.firstOrNull { it.label == l }?.let { return Card(null, null, it) }
            if (l.length < 2) return null
            val suit = Suit.fromCode(l.takeLast(1)) ?: return null
            val rank = Rank.fromCode(l.dropLast(1)) ?: return null
            return Card(rank, suit)
        }

        fun of(label: String): Card = parse(label) ?: throw IllegalArgumentException("Invalid card label: $label")
    }
}
