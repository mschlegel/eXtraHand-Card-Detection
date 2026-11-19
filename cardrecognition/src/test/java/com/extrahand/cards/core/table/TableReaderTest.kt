package com.extrahand.cards.core.table

import com.extrahand.cards.core.Card
import com.extrahand.cards.core.CardImage
import com.extrahand.cards.core.CardLocator
import com.extrahand.cards.core.KnnCardRecognizer
import com.extrahand.cards.core.LumaGridV1
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.BeforeClass
import org.junit.Test
import kotlin.math.cos
import kotlin.math.sin
import kotlin.random.Random

/**
 * End-to-end tests on synthetic table photos: cards are rendered with rotation, scale, perspective and
 * overlap onto a noisy felt background, then read back with [TableReader]. The recognizer is trained only on
 * upright close-up crops, like the real trainer app produces.
 */
class TableReaderTest {
    companion object {
        const val CW = 252
        const val CH = 352
        private val LABELS = listOf("AS", "7H", "10C", "QD", "KS", "3D", "JH", "9C", "5S", "2H", "8D", "JK")
        private val faces = HashMap<String, CardImage>()
        lateinit var rec: KnnCardRecognizer
        lateinit var recWithBacks: KnnCardRecognizer

        /** Synthetic card face: white, a label-specific glyph block in both index corners, a centre motif. */
        fun face(label: String): CardImage = faces.getOrPut(label) {
            val card = Card.of(label)
            val ink = if (card.suit?.isRed == true) 0xFFC01818.toInt() else 0xFF101010.toInt()
            val px = IntArray(CW * CH) { 0xFFF4F4F0.toInt() }
            val pat = Random(label.hashCode() * 31 + 7)
            val cells = BooleanArray(5 * 9) { pat.nextFloat() < 0.5f }
            fun stamp(x0: Float, y0: Float, flip: Boolean) {
                val cw = CW * 0.18f / 5; val ch = CH * 0.25f / 9
                for (cy in 0 until 9) for (cx in 0 until 5) {
                    if (!cells[(if (flip) 8 - cy else cy) * 5 + (if (flip) 4 - cx else cx)]) continue
                    for (y in (y0 + cy * ch).toInt() until (y0 + (cy + 1) * ch).toInt())
                        for (x in (x0 + cx * cw).toInt() until (x0 + (cx + 1) * cw).toInt()) px[y * CW + x] = ink
                }
            }
            stamp(CW * 0.03f, CH * 0.025f, false)
            stamp(CW * (1 - 0.03f - 0.18f), CH * (1 - 0.025f - 0.25f), true)
            val mid = Random(label.hashCode())
            repeat(6) {
                val x0 = (CW * (0.3f + 0.3f * mid.nextFloat())).toInt(); val y0 = (CH * (0.3f + 0.35f * mid.nextFloat())).toInt()
                for (y in y0 until y0 + 22) for (x in x0 until x0 + 18) px[y * CW + x] = ink
            }
            CardImage(CW, CH, px)
        }

        /** A blue patterned back with a white border. */
        val back: CardImage by lazy {
            val px = IntArray(CW * CH) { i ->
                val x = i % CW; val y = i / CW
                val border = x < 14 || y < 14 || x >= CW - 14 || y >= CH - 14
                when {
                    border -> 0xFFF2F2F2.toInt()
                    ((x / 9 + y / 9) % 2 == 0) -> 0xFF1E3C9A.toInt()
                    else -> 0xFF3A62D0.toInt()
                }
            }
            CardImage(CW, CH, px)
        }

        /** Like a trainer capture: the card on a dark background, tightened by CardLocator. */
        fun trainingCrop(img: CardImage, rnd: Random): CardImage {
            val m = 30
            val w = CW + 2 * m; val h = CH + 2 * m
            val px = IntArray(w * h) { 0xFF202420.toInt() }
            for (y in 0 until CH) for (x in 0 until CW) {
                val p = img.argb[y * CW + x]; val d = rnd.nextInt(-12, 13)
                fun c(s: Int) = ((p shr s and 0xFF) + d).coerceIn(0, 255)
                px[(y + m) * w + x + m] = (0xFF shl 24) or (c(16) shl 16) or (c(8) shl 8) or c(0)
            }
            return CardLocator.locate(CardImage(w, h, px)).image
        }

        @BeforeClass @JvmStatic fun train() {
            rec = KnnCardRecognizer(); recWithBacks = KnnCardRecognizer()
            val rnd = Random(1)
            for (l in LABELS) for (s in 0 until 3) {
                val f = LumaGridV1.extract(trainingCrop(face(l), rnd))
                rec.learn("$l-$s", Card.of(l), f); recWithBacks.learn("$l-$s", Card.of(l), f)
            }
            for (s in 0 until 3) recWithBacks.learn("BK-$s", Card.BACK, LumaGridV1.extract(trainingCrop(back, rnd)))
        }
    }

    /** A card placed in the scene: its image and the scene positions of its corners TL, TR, BR, BL. */
    internal class Placed(val img: CardImage, val corners: List<Pt>)

    /** Card with center [c], height [h] px, rotated [deg] degrees clockwise. */
    internal fun placed(label: String?, c: Pt, h: Float, deg: Float, img: CardImage? = null): Placed {
        val w = h * CW / CH
        val a = Math.toRadians(deg.toDouble())
        val ux = Pt(cos(a).toFloat(), sin(a).toFloat()); val uy = Pt(-sin(a).toFloat(), cos(a).toFloat())
        val tl = c - ux * (w / 2) - uy * (h / 2)
        return Placed(img ?: face(label!!), listOf(tl, tl + ux * w, tl + ux * w + uy * h, tl + uy * h))
    }

    /** Render cards (last = on top) onto a noisy green felt. */
    internal fun render(w: Int, h: Int, cards: List<Placed>, seed: Int = 3): CardImage {
        val rnd = Random(seed)
        val px = IntArray(w * h) {
            val d = rnd.nextInt(-10, 11)
            (0xFF shl 24) or ((34 + d).coerceIn(0, 255) shl 16) or ((110 + d).coerceIn(0, 255) shl 8) or (58 + d).coerceIn(0, 255)
        }
        val inv = cards.map { p ->
            Homography.fromPoints(p.corners, listOf(Pt(0f, 0f), Pt(CW.toFloat(), 0f), Pt(CW.toFloat(), CH.toFloat()), Pt(0f, CH.toFloat())))!!
        }
        for (p in cards.indices) {
            val b = Box.of(cards[p].corners)
            for (y in b.top.toInt().coerceAtLeast(0) until b.bottom.toInt().coerceAtMost(h - 1) + 1)
                for (x in b.left.toInt().coerceAtLeast(0) until b.right.toInt().coerceAtMost(w - 1) + 1) {
                    val q = inv[p].map(x + 0.5f, y + 0.5f)
                    if (q.x < 0 || q.y < 0 || q.x >= CW || q.y >= CH) continue
                    // Rounded corners like real cards.
                    val r = CW * 0.06f
                    val cx = q.x.coerceIn(r, CW - r); val cy = q.y.coerceIn(r, CH - r)
                    if ((q.x - cx) * (q.x - cx) + (q.y - cy) * (q.y - cy) > r * r) continue
                    px[y * w + x] = FramePlanes.bilinearArgb(cards[p].img, q.x - 0.5f, q.y - 0.5f)
                }
        }
        return CardImage(w, h, px)
    }

    private fun labels(s: TableScene) = s.cards.map { it.card.label }.toSet()

    @Test fun singleCardsAnyRotation() {
        val scene = render(1280, 960, listOf(
            placed("AS", Pt(250f, 300f), 260f, 0f),
            placed("7H", Pt(700f, 320f), 240f, 37f),
            placed("QD", Pt(1050f, 600f), 250f, 95f),
            placed("10C", Pt(400f, 720f), 230f, -160f),
        ))
        val r = TableReader(rec).read(scene)
        assertEquals(setOf("AS", "7H", "QD", "10C"), labels(r))
        assertTrue(r.cards.all { it.method == ReadMethod.FULL_CARD })
    }

    @Test fun farAwayCards() {
        val scene = render(1920, 1440, listOf(
            placed("KS", Pt(500f, 500f), 80f, 12f),
            placed("3D", Pt(1400f, 900f), 70f, -30f),
        ))
        assertEquals(setOf("KS", "3D"), labels(TableReader(rec).read(scene)))
    }

    @Test fun perspectiveCard() {
        // A card on a table seen from a low angle: the far edge is much shorter.
        val p = Placed(face("JH"), listOf(Pt(560f, 300f), Pt(760f, 300f), Pt(860f, 520f), Pt(460f, 520f)))
        assertEquals(setOf("JH"), labels(TableReader(rec).read(render(1280, 960, listOf(p)))))
    }

    @Test fun fannedHand() {
        val hand = listOf("AS", "7H", "10C", "QD", "KS")
        val pivot = Pt(640f, 860f)
        val h = 320f
        val cards = hand.mapIndexed { i, l ->
            val deg = (i - 2) * 16f
            val a = Math.toRadians(deg.toDouble())
            // Card centre sits 0.42 h "above" the pivot along its own axis.
            val c = pivot + Pt(sin(a).toFloat(), -cos(a).toFloat()) * (h * 0.42f)
            placed(l, c, h, deg)
        }
        val r = TableReader(rec).read(render(1280, 960, cards))
        val got = labels(r)
        assertTrue("got $got", got.containsAll(hand))
        assertEquals("got $got", hand.size, got.size)
        assertEquals(1, r.groups.size)
    }

    @Test fun linearSpread() {
        val hand = listOf("9C", "5S", "2H", "8D")
        val h = 300f; val w = h * CW / CH
        val a = Math.toRadians(8.0)
        val dir = Pt(cos(a).toFloat(), sin(a).toFloat())
        val cards = hand.mapIndexed { i, l -> placed(l, Pt(400f, 450f) + dir * (i * w * 0.38f), h, 8f) }
        val got = labels(TableReader(rec).read(render(1280, 960, cards)))
        assertEquals(hand.toSet(), got)
    }

    @Test fun trainedBackIsDetected() {
        val scene = render(1280, 960, listOf(
            placed(null, Pt(300f, 400f), 260f, 15f, back),
            placed("AS", Pt(800f, 400f), 260f, -10f),
        ))
        val r = TableReader(recWithBacks).read(scene)
        assertEquals(setOf("AS"), labels(r))
        assertEquals(1, r.backs.size)
        assertTrue(r.backs[0].fromTraining)
        assertEquals(BackLayout.SINGLE_OR_PILE, r.backs[0].layout)
    }

    @Test fun untrainedBackUsesHeuristic() {
        val scene = render(1280, 960, listOf(placed(null, Pt(500f, 450f), 280f, 30f, back)))
        val r = TableReader(rec).read(scene)
        assertTrue(r.cards.isEmpty())
        assertEquals(1, r.backs.size)
        assertTrue(!r.backs[0].fromTraining)
    }

    @Test fun spreadOfBacks() {
        val cards = (0 until 5).map { i -> placed(null, Pt(300f + i * 70f, 450f), 280f, 0f, back) }
        val r = TableReader(recWithBacks).read(render(1280, 960, cards))
        assertTrue(r.cards.isEmpty())
        assertEquals(1, r.backs.size)
        assertEquals(BackLayout.SPREAD, r.backs[0].layout)
        assertTrue(r.backs[0].minCount >= 2)
    }

    @Test fun emptyTableAndClutterGiveNothing() {
        val reader = TableReader(recWithBacks)
        assertTrue(reader.read(render(1280, 960, emptyList())).isEmpty)
        // A dark phone-like slab and a bright, unpatterned coaster.
        val slab = CardImage(CW, CH, IntArray(CW * CH) { 0xFF151515.toInt() })
        val coaster = CardImage(CW, CH, IntArray(CW * CH) { i -> if ((i % CW) in 100..150) 0xFFDDDDDD.toInt() else 0xFFFAFAFA.toInt() })
        val r = reader.read(render(1280, 960, listOf(
            placed(null, Pt(300f, 400f), 300f, 20f, slab), placed(null, Pt(850f, 450f), 250f, -5f, coaster),
        )))
        assertTrue("cards=${r.cards}", r.cards.isEmpty())
        assertTrue("backs=${r.backs}", r.backs.none { it.fromTraining })
    }

    @Test fun mixedTable() {
        // A fanned hand, a draw pile, and two loose cards far from the hand.
        val hand = listOf("JH", "9C", "5S")
        val pivot = Pt(380f, 900f)
        val cards = hand.mapIndexed { i, l ->
            val deg = (i - 1) * 18f
            val a = Math.toRadians(deg.toDouble())
            placed(l, pivot + Pt(sin(a).toFloat(), -cos(a).toFloat()) * (300f * 0.42f), 300f, deg)
        } + listOf(
            placed(null, Pt(1500f, 300f), 260f, 80f, back),
            placed("2H", Pt(1250f, 820f), 240f, 4f),
            placed("8D", Pt(1500f, 840f), 240f, -6f),
        )
        val r = TableReader(recWithBacks).read(render(1920, 1080, cards))
        assertEquals(setOf("JH", "9C", "5S", "2H", "8D"), labels(r))
        assertEquals(2, r.groups.size)
        assertEquals(setOf("JH", "9C", "5S"), r.groups.first { it.overlapping }.cards.map { it.card.label }.toSet())
        assertEquals(1, r.backs.size)
    }

    @Test fun stabilizerNeedsRepeats() {
        val st = SceneStabilizer(window = 3, minHits = 2)
        val seen = TableReader(rec).read(render(1280, 960, listOf(placed("AS", Pt(400f, 400f), 260f, 0f))))
        val none = TableScene.empty(1280, 960)
        assertTrue(st.update(seen).cards.isEmpty())
        assertEquals(setOf("AS"), labels(st.update(seen)))
        assertEquals(setOf("AS"), labels(st.update(none)))   // one miss is tolerated
        assertTrue(st.update(none).cards.isEmpty())
    }

    @Test fun cornerViewMatchesItself() {
        // The corner descriptor of an upright card's TL patch should match that card's stored corner view.
        val img = face("QD")
        val planes = FramePlanes.of(img)
        val q = CornerIndex.describe(planes, Pt(0f, 0f), Pt(CW * CornerIndex.CORNER_FRAC_W, 0f), Pt(0f, CH * CornerIndex.CORNER_FRAC_H))!!
        val v = CornerIndex.cornerView(LumaGridV1.extract(img))
        val cos = q.indices.sumOf { (q[it] * v[it]).toDouble() }
        assertTrue("cos=$cos", cos > 0.95)
    }
}
