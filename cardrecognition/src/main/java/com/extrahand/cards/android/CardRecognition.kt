package com.extrahand.cards.android

import android.content.Context
import android.graphics.Bitmap
import com.extrahand.cards.core.CardRecognizer
import com.extrahand.cards.core.KnnCardRecognizer
import com.extrahand.cards.core.Recognition
import com.extrahand.cards.core.RecognitionThresholds
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
