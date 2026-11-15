package com.extrahand.cards.core

import org.junit.Test
import java.io.File
import kotlin.random.Random

/** Writes build/parity/{card.ppm, features.txt} so tools/python/cardnet.py can be checked against Kotlin. */
class ParityFixtureTest {
    @Test fun writeFixture() {
        val w = 233; val h = 331
        val rnd = Random(42)
        val px = IntArray(w * h) { i ->
            val x = i % w; val y = i / w
            val base = if ((x / 17 + y / 23) % 3 == 0) 0x202020 else 0xF0EEE8
            val red = if (x < 50 && y < 80 && rnd.nextInt(4) == 0) 0xC01010 else base
            (0xFF shl 24) or red
        }
        val img = CardImage(w, h, px)
        val f = LumaGridV1.extract(img)
        val dir = File("build/parity").apply { mkdirs() }
        // Binary PPM (P6): lossless and readable by Pillow without extra deps on the Android test classpath.
        val rgb = ByteArray(w * h * 3)
        px.forEachIndexed { i, p -> rgb[i * 3] = (p shr 16).toByte(); rgb[i * 3 + 1] = (p shr 8).toByte(); rgb[i * 3 + 2] = p.toByte() }
        File(dir, "card.ppm").writeBytes("P6\n$w $h\n255\n".toByteArray() + rgb)
        File(dir, "features.txt").writeText(f.joinToString("\n"))
    }
}
