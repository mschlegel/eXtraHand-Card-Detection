package com.extrahand.cards.sync

import com.extrahand.cards.core.DatasetCodec
import com.extrahand.cards.core.DatasetManifest
import com.extrahand.cards.core.FeatureShard
import java.io.File
import java.io.IOException

/**
 * Downloads a dataset (manifest + feature shards) with an on-disk cache.
 *
 * Shards are immutable, so each is downloaded at most once ever; a refresh costs one small manifest
 * request. If the network fails, the last cached dataset is returned (offline-friendly).
 *
 * Pure JVM (java.net + java.io): usable from Android, desktop, or server Kotlin/Java.
 */
class RemoteDatasetLoader(private val source: DatasetSource, cacheRoot: File) {
    private val dir = File(cacheRoot, source.cacheKey).apply { mkdirs() }
    private val shardDir = File(dir, "features").apply { mkdirs() }
    private val manifestFile = File(dir, "manifest.json")

    data class Loaded(val manifest: DatasetManifest, val shards: List<FeatureShard>, val fromCache: Boolean)

    /** Fetch the latest dataset (blocking). Falls back to cache on network errors; throws if neither works. */
    fun load(): Loaded {
        val manifestText: String
        var fromCache = false
        try {
            val r = Http.get(source.url("manifest.json"), source.headers())
            when {
                r.ok -> { manifestText = r.text(); manifestFile.writeText(manifestText) }
                r.code == 404 -> return Loaded(DatasetManifest(), emptyList(), false) // nothing published yet
                else -> throw Http.HttpException(r.code, r.text().take(200))
            }
        } catch (e: IOException) {
            if (!manifestFile.exists()) throw e
            return loadCached() ?: throw e
        }
        val manifest = DatasetCodec.decodeManifest(manifestText)
        val shards = manifest.shards.map { ref ->
            val f = File(shardDir, File(ref.file).name)
            if (!f.exists() || DatasetCodec.sha256(f.readBytes()) != ref.sha256) {
                val r = Http.get(source.url(ref.file), source.headers())
                if (!r.ok) throw Http.HttpException(r.code, "shard ${ref.file}")
                f.writeBytes(r.body)
            } else fromCache = true
            DatasetCodec.decodeShard(f.readText())
        }
        return Loaded(manifest, shards, fromCache && manifest.shards.isNotEmpty())
    }

    /** Download one dataset file (e.g. a sample JPEG at "samples/JK/<id>.jpg"). Null if missing. Blocking. */
    fun fetch(relativePath: String): ByteArray? {
        val r = Http.get(source.url(relativePath), source.headers())
        return if (r.ok) r.body else null
    }

    /** Whatever was last downloaded, without touching the network. Null if nothing cached. */
    fun loadCached(): Loaded? {
        if (!manifestFile.exists()) return null
        val manifest = DatasetCodec.decodeManifest(manifestFile.readText())
        val shards = manifest.shards.map { ref ->
            val f = File(shardDir, File(ref.file).name)
            if (!f.exists()) return null
            DatasetCodec.decodeShard(f.readText())
        }
        return Loaded(manifest, shards, true)
    }
}
