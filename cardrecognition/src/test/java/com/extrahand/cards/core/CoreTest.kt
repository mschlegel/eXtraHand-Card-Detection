package com.extrahand.cards.core

import com.extrahand.cards.sync.LocalDatasetStore
import org.junit.Assert.assertEquals
import org.junit.Assert.assertNotNull
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Test
import java.nio.file.Files
import kotlin.random.Random

class CoreTest {
    /** Synthetic "card": white face, glyph pattern in both corners derived from the label, optional red ink. */
    private fun fakeCard(label: String, w: Int = 252, h: Int = 352, noise: Int = 0, seed: Int = 0, bg: Int = 0xFF203020.toInt()): CardImage {
        val rnd = Random(seed)
        val card = Card.of(label)
        val ink = if (card.suit?.isRed == true) 0xFFC01818.toInt() else 0xFF101010.toInt()
        val px = IntArray(w * h) { 0xFFF4F4F0.toInt() }
        val pattern = Random(label.hashCode())
        val cells = BooleanArray(6 * 10) { pattern.nextBoolean() }
        fun stamp(x0: Int, y0: Int, flip: Boolean) {
            val cw = w * 0.20f / 6; val ch = h * 0.27f / 10
            for (cy in 0 until 10) for (cx in 0 until 6) {
                val on = cells[(if (flip) 9 - cy else cy) * 6 + (if (flip) 5 - cx else cx)]
                if (!on) continue
                for (y in (y0 + cy * ch).toInt() until (y0 + (cy + 1) * ch).toInt())
                    for (x in (x0 + cx * cw).toInt() until (x0 + (cx + 1) * cw).toInt()) px[y * w + x] = ink
            }
        }
        stamp((w * 0.02f).toInt(), (h * 0.02f).toInt(), false)
        stamp((w * 0.78f).toInt(), (h * 0.71f).toInt(), true)
        if (noise > 0) for (i in px.indices) {
            val d = rnd.nextInt(-noise, noise + 1)
            val p = px[i]
            fun c(s: Int) = ((p shr s and 0xFF) + d).coerceIn(0, 255)
            px[i] = (0xFF shl 24) or (c(16) shl 16) or (c(8) shl 8) or c(0)
        }
        // Put it on a dark background with a margin, like a camera guide-box crop.
        val m = 30
        val bw = w + 2 * m; val bh = h + 2 * m
        val out = IntArray(bw * bh) { bg }
        for (y in 0 until h) System.arraycopy(px, y * w, out, (y + m) * bw + m, w)
        return CardImage(bw, bh, out)
    }

    @Test fun cardLabels() {
        assertEquals("10H", Card.of("10h").label)
        assertEquals("10H", Card.of("TH").label)
        assertEquals(Card.JOKER, Card.of("JK"))
        assertEquals(Card.BACK, Card.of("bk"))
        assertTrue(Card.BACK.isBack && !Card.BACK.isJoker && !Card.JOKER.isBack)
        assertEquals("BK", Card.BACK.label)
        assertEquals(Card.JOKER_HIGH, Card.of("jkh"))
        assertEquals(Card.JOKER_LOW, Card.of("JKL"))
        assertEquals(Card.JOKER, Card(null, null))
        assertTrue(Card.JOKER_HIGH.isJoker && Card.JOKER_LOW.isJoker && !Card.JOKER_HIGH.isBack)
        assertTrue(Card.JOKER_HIGH != Card.JOKER_LOW && Card.JOKER_HIGH != Card.JOKER)
        assertEquals("JH", Card.of("JH").label) // jack of hearts is unaffected
        assertNull(Card.parse("1X"))
        assertEquals(52, Card.STANDARD_DECK.map { it.label }.toSet().size)
    }

    @Test fun featureShapeAndNorm() {
        val f = LumaGridV1.extract(fakeCard("AS"))
        assertEquals(1248, f.size)
        assertEquals(1.0, f.sumOf { (it * it).toDouble() }, 1e-4)
    }

    @Test fun vectorCodecRoundTrip() {
        val f = LumaGridV1.extract(fakeCard("7D"))
        val back = DatasetCodec.decodeVector(DatasetCodec.encodeVector(f), f.size)
        val cos = f.indices.sumOf { (f[it] * back[it]).toDouble() }
        assertTrue("cos=$cos", cos > 0.999)
    }

    @Test fun locatorTightensCrop() {
        val loc = CardLocator.locate(fakeCard("KC"))
        assertTrue(loc.refined)
        assertTrue(loc.rect.left > 0.05f && loc.rect.right < 0.95f)
    }

    @Test fun learnsAndRecognizes() {
        val rec = KnnCardRecognizer()
        assertNull(rec.recognize(fakeCard("AS")).card)
        val labels = listOf("AS", "AH", "10C", "QD", "7S", "7H")
        for (l in labels) for (s in 0 until 3) {
            val crop = CardLocator.locate(fakeCard(l, noise = 20, seed = s)).image
            rec.learn("$l-$s", Card.of(l), LumaGridV1.extract(crop))
        }
        for (l in labels) {
            val r = rec.recognize(fakeCard(l, noise = 25, seed = 99))
            assertEquals(l, r.card?.label)
            assertTrue(r.confidence > 0.5f)
        }
        // Upside-down card is recognized too.
        val flipped = fakeCard("QD", noise = 10, seed = 5).rotate90().rotate90()
        assertEquals("QD", rec.recognize(flipped).card?.label)
        assertTrue(rec.forget("AS-0"))
        assertEquals(labels.size * 3 - 1, rec.sampleCount)
    }

    @Test fun datasetRoundTrip() {
        val samples = listOf("AS", "2C").mapIndexed { i, l ->
            SampleRecord("id$i", l, DatasetCodec.encodeVector(LumaGridV1.extract(fakeCard(l))), 1000L + i)
        }
        val shard = DatasetCodec.decodeShard(DatasetCodec.encodeShard(FeatureShard(samples = samples)))
        val manifest = DatasetCodec.decodeManifest(DatasetCodec.encodeManifest(
            DatasetManifest(datasetVersion = 3, shards = listOf(ShardRef("features/a.json", 2, "x")), deleted = listOf("id1"))
        ))
        val rec = KnnCardRecognizer.fromDataset(manifest, listOf(shard))
        assertEquals(1, rec.sampleCount)
        assertEquals("AS", rec.recognize(fakeCard("AS")).candidates.first().card.label)
    }

    @Test fun localStorePersists() {
        val dir = Files.createTempDirectory("store").toFile()
        val s1 = LocalDatasetStore(dir)
        val f = DatasetCodec.encodeVector(LumaGridV1.extract(fakeCard("AS")))
        s1.add(SampleRecord("a", "AS", f, 1, image = "samples/AS/a.jpg"), byteArrayOf(1, 2, 3))
        s1.add(SampleRecord("b", "2C", f, 2), null)
        s1.delete("b")
        val s2 = LocalDatasetStore(dir)
        assertEquals(listOf("a"), s2.samples().map { it.id })
        assertEquals(listOf("a"), s2.syncState().pendingAdds)
        assertTrue(s2.imageFile("samples/AS/a.jpg").exists())
        assertNotNull(s2.get("a"))
        dir.deleteRecursively()
    }

    @Test fun relabelMovesSampleAndImage() {
        val dir = Files.createTempDirectory("store").toFile()
        val st = LocalDatasetStore(dir)
        val f = DatasetCodec.encodeVector(LumaGridV1.extract(fakeCard("AS")))
        // Pretend both were published already: one has its JPEG on the device, one was pulled from GitHub.
        st.add(SampleRecord("local", "JK", f, 1, image = "samples/JK/local.jpg"), byteArrayOf(7, 7))
        st.add(SampleRecord("remote", "JK", f, 2, image = "samples/JK/remote.jpg"), null)
        st.markPublished(listOf("local", "remote"), emptyList(), emptyList(), 1, emptyList())

        val a = st.relabel("local", "JKH", "n1")!!
        val b = st.relabel("remote", "JKL", "n2")!!
        assertEquals("samples/JKH/n1.jpg", a.image)
        assertEquals(f, a.f)
        assertTrue(st.imageFile("samples/JKH/n1.jpg").exists())
        assertTrue(!st.imageFile("samples/JK/local.jpg").exists())
        val state = LocalDatasetStore(dir).syncState()
        assertEquals(setOf("n1", "n2"), state.pendingAdds.toSet())
        assertEquals(setOf("local", "remote"), state.pendingDeletes.map { it.id }.toSet())
        assertEquals(mapOf("samples/JKL/n2.jpg" to "samples/JK/remote.jpg"), state.pendingCopies)
        assertEquals(setOf("JKH", "JKL"), st.samples().map { it.label }.toSet())
        assertEquals(b.createdAt, 2L)
        dir.deleteRecursively()
    }
}
