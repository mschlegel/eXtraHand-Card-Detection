package com.extrahand.cards.core.table

import com.extrahand.cards.core.Card
import com.extrahand.cards.core.CardImage
import com.extrahand.cards.core.DatasetCodec
import com.extrahand.cards.core.KnnCardRecognizer
import com.extrahand.cards.core.SampleRecord
import org.junit.Assume.assumeTrue
import org.junit.Test
import java.io.File
import kotlin.math.cos
import kotlin.math.sin
import kotlin.random.Random

/**
 * Benchmark of the table reader on the REAL trained dataset in ../dataset. Rendered scenes (a fan of 5 and
 * 3 loose cards) use the owner's own sample photos as card faces. Opt-in because it takes about a minute:
 *
 *   EXTRAHAND_REAL=1 ./gradlew :cardrecognition:testDebugUnitTest --tests '*RealDatasetTest*' -i
 *
 * Results at the time of writing (272 samples, 5 deck designs, about one photo per card per deck):
 *   stored vectors only:          found 15/48, wrong 15
 *   with sample photos (default): found 31/48, wrong 5   (whole cards ~24/25; fanned cards are the weak part)
 */
class RealDatasetTest {
    private val root = File("../dataset")

    private fun load(): List<Pair<SampleRecord, CardImage?>> {
        val m = DatasetCodec.decodeManifest(File(root, "manifest.json").readText())
        val shards = m.shards.map { DatasetCodec.decodeShard(File(root, it.file).readText()) }
        return DatasetCodec.liveSamples(m, shards).map { s -> s to s.image?.let { File(root, it) }?.takeIf { it.exists() }?.let(::readJpeg) }
    }

    @Test fun realCardsInScenes() {
        assumeTrue(System.getenv("EXTRAHAND_REAL") == "1" && File(root, "manifest.json").exists())
        val all = load()
        println("REAL ${all.size} samples, ${all.count { it.second != null }} with images")
        val rect = all.map { p -> p.second?.let { Rectifier.rectify(it) ?: it } }
        val recStored = KnnCardRecognizer().apply { addAll(all.map { it.first }) }
        val byId = all.associate { it.first.id to it.second }
        val faces = all.indices.filter { Card.parse(all[it].first.label)?.isFace == true && rect[it] != null }.map { all[it].first.label to rect[it]!! }

        for ((name, mk) in listOf<Pair<String, () -> TableReader>>(
            "stored vectors only" to { TableReader(recStored) },
            "with sample photos" to { TableReader(recStored, sampleImages = { id, _ -> byId[id] }) },
        )) {
            val reader = mk()
            val t = TableReaderTest()
            val rnd = Random(42)
            var found = 0; var total = 0; var wrong = 0
            repeat(6) { trial ->
                val pick = faces.shuffled(rnd).distinctBy { it.first }.take(8)
                val h = listOf(260f, 320f, 180f)[trial % 3]
                val pivot = Pt(560f, 1000f)
                val fan = pick.take(5).mapIndexed { i, p ->
                    val deg = (i - 2) * 17f; val a = Math.toRadians(deg.toDouble())
                    t.placed(null, pivot + Pt(sin(a).toFloat(), -cos(a).toFloat()) * (h * 0.42f), h, deg, resize(p.second))
                }
                val loose = pick.drop(5).mapIndexed { i, p ->
                    t.placed(null, Pt(1150f + i * 260f, 400f + i * 120f), h * 0.9f, -20f + 25f * i, resize(p.second))
                }
                val scene = reader.read(t.render(1920, 1200, fan + loose, seed = trial))
                val want = pick.map { it.first }.toSet()
                val got = scene.cards.map { it.card.label }.toSet()
                found += (got intersect want).size; total += want.size; wrong += (got - want).size
                println("REAL [$name] h=$h fan=${pick.take(5).map { it.first }} loose=${pick.drop(5).map { it.first }} got=$got ${scene.elapsedMs}ms")
            }
            println("REAL [$name] found $found / $total, wrong $wrong")
        }
    }

    private fun resize(img: CardImage, w: Int = TableReaderTest.CW, h: Int = TableReaderTest.CH) = CardImage(w, h, IntArray(w * h) { i ->
        FramePlanes.bilinearArgb(img, (i % w + 0.5f) * img.width / w - 0.5f, (i / w + 0.5f) * img.height / h - 0.5f)
    })

    /** javax.imageio exists on the desktop JVM that runs unit tests, but not on the Android compile classpath. */
    private fun readJpeg(f: File): CardImage {
        val bi = Class.forName("javax.imageio.ImageIO").getMethod("read", File::class.java).invoke(null, f)
        val w = bi.javaClass.getMethod("getWidth").invoke(bi) as Int
        val h = bi.javaClass.getMethod("getHeight").invoke(bi) as Int
        val px = IntArray(w * h)
        val ip = Int::class.javaPrimitiveType
        bi.javaClass.getMethod("getRGB", ip, ip, ip, ip, IntArray::class.java, ip, ip).invoke(bi, 0, 0, w, h, px, 0, w)
        return CardImage(w, h, px)
    }
}
