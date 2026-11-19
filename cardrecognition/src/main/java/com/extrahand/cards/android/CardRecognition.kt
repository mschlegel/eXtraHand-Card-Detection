package com.extrahand.cards.android

import android.content.Context
import android.graphics.Bitmap
import com.extrahand.cards.core.CardRecognizer
import com.extrahand.cards.core.KnnCardRecognizer
import com.extrahand.cards.core.Recognition
import com.extrahand.cards.core.RecognitionThresholds
import com.extrahand.cards.core.table.TableReader
import com.extrahand.cards.core.table.TableReaderConfig
import com.extrahand.cards.sync.DatasetSource
import com.extrahand.cards.sync.RemoteDatasetLoader
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import java.io.File

/**
 * Entry point for host apps.
 *
 * ```
 * val recognizer = CardRecognition.load(context)            // latest trained data from GitHub (cached)
 * val result = recognizer.recognize(bitmap)                 // bitmap = card roughly filling the frame
 * result.card?.label                                         // "QH", or null if unsure
 * ```
 */
object CardRecognition {
    data class LoadResult(
        val recognizer: CardRecognizer,
        val datasetVersion: Long,
        val sampleCount: Int,
        /** True if the network was unavailable and the cached copy was used. */
        val fromCache: Boolean,
    )

    /** Load the newest dataset from [source] (default: the public eXtraHand dataset). Falls back to cache offline. */
    suspend fun load(
        context: Context,
        source: DatasetSource = DatasetSource.DEFAULT,
        thresholds: RecognitionThresholds = RecognitionThresholds(),
    ): CardRecognizer = loadDetailed(context, source, thresholds).recognizer

    suspend fun loadDetailed(
        context: Context,
        source: DatasetSource = DatasetSource.DEFAULT,
        thresholds: RecognitionThresholds = RecognitionThresholds(),
    ): LoadResult = withContext(Dispatchers.IO) {
        val loader = RemoteDatasetLoader(source, cacheDir(context))
        val loaded = loader.load()
        val rec = KnnCardRecognizer.fromDataset(loaded.manifest, loaded.shards, thresholds = thresholds)
        LoadResult(rec, loaded.manifest.datasetVersion, rec.sampleCount, loaded.fromCache)
    }

    /** Offline-only: the last dataset downloaded on this device, or null. */
    suspend fun loadCached(context: Context, source: DatasetSource = DatasetSource.DEFAULT): CardRecognizer? =
        withContext(Dispatchers.IO) {
            RemoteDatasetLoader(source, cacheDir(context)).loadCached()
                ?.let { KnnCardRecognizer.fromDataset(it.manifest, it.shards) }
        }

    fun cacheDir(context: Context) = File(context.cacheDir, "extrahand-cards")
}

/** Convenience: recognize straight from a Bitmap. */
fun CardRecognizer.recognize(bitmap: Bitmap, topN: Int = 3, locate: Boolean = true): Recognition =
    recognize(bitmap.toCardImage(), topN, locate)

/**
 * Load a [TableReader] (whole-table reading: hands, piles, card backs) over the newest dataset from [source].
 *
 * With [usePhotos] (default) it also downloads the sample JPEGs (cached on disk, each fetched once) and
 * matches against them. That is much more accurate (see RealDatasetTest), at the cost of a one-time
 * download of a few MB. Photos are downloaded here, before the reader is returned.
 */
suspend fun CardRecognition.loadTableReader(
    context: Context,
    source: DatasetSource = DatasetSource.DEFAULT,
    config: TableReaderConfig = TableReaderConfig(),
    usePhotos: Boolean = true,
): TableReader = withContext(Dispatchers.IO) {
    val loader = RemoteDatasetLoader(source, cacheDir(context))
    val loaded = loader.load()
    val samples = com.extrahand.cards.core.DatasetCodec.liveSamples(loaded.manifest, loaded.shards)
    val photoDir = File(cacheDir(context), source.cacheKey + "-photos")
    val paths = HashMap<String, File>()
    if (usePhotos) for (s in samples) {
        val rel = s.image ?: continue
        val f = File(photoDir, rel)
        if (!f.exists()) runCatching { loader.fetch(rel) }.getOrNull()?.let { f.parentFile?.mkdirs(); f.writeBytes(it) }
        if (f.exists()) paths[s.id] = f
    }
    TableReader(
        KnnCardRecognizer.fromDataset(loaded.manifest, loaded.shards), config,
        sampleImages = if (paths.isEmpty()) null else { id, _ -> paths[id]?.let { android.graphics.BitmapFactory.decodeFile(it.path)?.toCardImage() } },
    )
}
