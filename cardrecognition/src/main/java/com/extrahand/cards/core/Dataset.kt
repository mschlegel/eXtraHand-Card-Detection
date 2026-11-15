package com.extrahand.cards.core

import kotlinx.serialization.Serializable
import kotlinx.serialization.json.Json
import java.security.MessageDigest
import java.util.Base64
import kotlin.math.abs
import kotlin.math.roundToInt

/*
 * Dataset wire format (formatVersion 1). Full description: docs/DATASET_FORMAT.md
 *
 *   <root>/manifest.json            DatasetManifest - small, the only mutable file; read it first
 *   <root>/features/<shard>.json    FeatureShard    - IMMUTABLE once written; safe to cache forever by name
 *   <root>/samples/<LABEL>/<id>.jpg the cropped card image each sample was computed from (source of truth)
 */

@Serializable
data class DatasetManifest(
    val formatVersion: Int = FORMAT_VERSION,
    /** Monotonic counter, +1 on every published change. */
    val datasetVersion: Long = 0,
    /** ISO-8601 UTC. */
    val updatedAt: String = "",
    val extractor: String = LumaGridV1.id,
    val featureDim: Int = LumaGridV1.dim,
    val encoding: String = ENCODING_INT8_B64,
    val sampleCount: Int = 0,
    val labelCounts: Map<String, Int> = emptyMap(),
    /** In order of creation. Load all of them, then drop ids listed in [deleted]. */
    val shards: List<ShardRef> = emptyList(),
    /** Sample ids removed after their shard was written (tombstones). */
    val deleted: List<String> = emptyList(),
) {
    companion object {
        const val FORMAT_VERSION = 1
        const val ENCODING_INT8_B64 = "int8-base64"
    }
}

@Serializable
data class ShardRef(val file: String, val count: Int, val sha256: String)

@Serializable
data class FeatureShard(
    val formatVersion: Int = DatasetManifest.FORMAT_VERSION,
    val extractor: String = LumaGridV1.id,
    val featureDim: Int = LumaGridV1.dim,
    val encoding: String = DatasetManifest.ENCODING_INT8_B64,
    val samples: List<SampleRecord> = emptyList(),
)

@Serializable
data class SampleRecord(
    /** Globally unique, e.g. "20261115T203012Z-1a2b3c". */
    val id: String,
    /** Card label, e.g. "10H" (see Card.label). */
    val label: String,
    /** Feature vector, encoded per manifest.encoding (int8-base64). */
    val f: String,
    /** Epoch millis when captured. */
    val createdAt: Long,
    /** What the recognizer guessed before the user labelled it (null = no guess). */
    val predicted: String? = null,
    /** Was the guess right? (null = no guess). */
    val correct: Boolean? = null,
    /** Image path relative to the dataset root, e.g. "samples/10H/<id>.jpg". */
    val image: String? = null,
    /** Free-form capture source, e.g. "card-trainer/0.1.0". */
    val source: String? = null,
)

object DatasetCodec {
    val json = Json { ignoreUnknownKeys = true; prettyPrint = false; explicitNulls = false; encodeDefaults = true }
    val prettyJson = Json { ignoreUnknownKeys = true; prettyPrint = true; explicitNulls = false; encodeDefaults = true }

    /** Quantize an L2-normalized vector to int8 (scaled so max |v| -> 127) and base64 it. */
    fun encodeVector(v: FloatArray): String {
        var maxAbs = 0f
        for (x in v) maxAbs = maxOf(maxAbs, abs(x))
        val scale = if (maxAbs > 0f) 127f / maxAbs else 0f
        val bytes = ByteArray(v.size) { (v[it] * scale).roundToInt().coerceIn(-127, 127).toByte() }
        return Base64.getEncoder().encodeToString(bytes)
    }

    /** Decode int8-base64 back to an L2-normalized float vector. */
    fun decodeVector(s: String, expectedDim: Int): FloatArray {
        val bytes = Base64.getDecoder().decode(s)
        require(bytes.size == expectedDim) { "vector length ${bytes.size} != $expectedDim" }
        val v = FloatArray(bytes.size) { bytes[it].toFloat() }
        l2Normalize(v)
        return v
    }

    fun encodeManifest(m: DatasetManifest): String = prettyJson.encodeToString(DatasetManifest.serializer(), m)
    fun decodeManifest(s: String): DatasetManifest = json.decodeFromString(DatasetManifest.serializer(), s)
    fun encodeShard(s: FeatureShard): String = json.encodeToString(FeatureShard.serializer(), s)
    fun decodeShard(s: String): FeatureShard = json.decodeFromString(FeatureShard.serializer(), s)
    fun encodeSample(s: SampleRecord): String = json.encodeToString(SampleRecord.serializer(), s)
    fun decodeSample(s: String): SampleRecord = json.decodeFromString(SampleRecord.serializer(), s)

    fun sha256(bytes: ByteArray): String =
        MessageDigest.getInstance("SHA-256").digest(bytes).joinToString("") { "%02x".format(it) }

    /** Resolve the live sample set described by a manifest + its loaded shards. */
    fun liveSamples(manifest: DatasetManifest, shards: List<FeatureShard>): List<SampleRecord> {
        val deleted = manifest.deleted.toHashSet()
        val seen = HashSet<String>()
        return shards.flatMap { it.samples }.filter { it.id !in deleted && seen.add(it.id) }
    }
}
