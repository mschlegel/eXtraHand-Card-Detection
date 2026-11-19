package com.extrahand.cards.sync

import com.extrahand.cards.core.DatasetCodec
import com.extrahand.cards.core.DatasetManifest
import com.extrahand.cards.core.FeatureShard
import com.extrahand.cards.core.LumaGridV1
import com.extrahand.cards.core.SampleRecord
import com.extrahand.cards.core.ShardRef
import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale
import java.util.TimeZone

data class GitHubConfig(
    val owner: String,
    val repo: String,
    val branch: String = "main",
    /** Dataset root inside the repo. */
    val path: String = "dataset",
    val token: String,
)

/**
 * Two-way sync between a [LocalDatasetStore] and a dataset folder in a GitHub repo.
 *
 * 1. Pull: merge any remote shards this device has not seen (e.g. after a reinstall) and remote tombstones.
 * 2. Push: write ONE new immutable shard with all pending samples, upload their JPEGs, tombstone pending
 *    deletes, rewrite manifest.json - all in a single commit. Retries if the branch moved meanwhile.
 *
 * Blocking; call from a background thread / Dispatchers.IO.
 */
class GitHubDatasetSync(private val store: LocalDatasetStore, private val cfg: GitHubConfig) {
    private val gh = GitHubClient(cfg.owner, cfg.repo, cfg.token)
    private fun p(rel: String) = listOf(cfg.path.trim('/'), rel).filter { it.isNotEmpty() }.joinToString("/")

    data class Result(
        val pulled: Int,
        val pushed: Int,
        val deleted: Int,
        val datasetVersion: Long,
        val commitSha: String?,
        /** Samples learned from the remote; callers should add these to their recognizer. */
        val importedSamples: List<SampleRecord>,
        val removedIds: List<String>,
    )

    fun interface Progress { fun update(message: String) }

    fun sync(progress: Progress = Progress { }): Result {
        val imported = ArrayList<SampleRecord>()
        val removed = ArrayList<String>()
        repeat(MAX_ATTEMPTS) { attempt ->
            progress.update(if (attempt == 0) "Checking GitHub..." else "Branch moved, retrying...")
            val head = gh.headCommit(cfg.branch)
            val baseTree = gh.commitTree(head)
            val remote = gh.getFile(p("manifest.json"), head)?.let { DatasetCodec.decodeManifest(String(it)) }
                ?: DatasetManifest()
            require(remote.extractor == LumaGridV1.id || remote.shards.isEmpty()) {
                "Remote dataset uses extractor '${remote.extractor}', this app uses '${LumaGridV1.id}'"
            }

            // ---- pull ----
            val known = store.syncState().knownShards.map { it.file }.toHashSet()
            for (ref in remote.shards.filter { it.file !in known }) {
                progress.update("Downloading ${ref.file}...")
                val bytes = gh.getFile(p(ref.file), head) ?: continue
                imported += store.importRemote(ref, DatasetCodec.decodeShard(String(bytes)).samples)
            }
            removed += store.applyRemoteDeletes(remote.deleted)

            // ---- push ----
            val adds = store.pendingAddRecords()
            val dels = store.syncState().pendingDeletes
            if (adds.isEmpty() && dels.isEmpty()) {
                store.markPulled(remote.datasetVersion)
                return Result(imported.size, 0, 0, remote.datasetVersion, null, imported, removed)
            }

            val version = remote.datasetVersion + 1
            val entries = ArrayList<GitHubClient.TreeEntry>()
            adds.forEachIndexed { i, s ->
                val img = s.image ?: return@forEachIndexed
                val f = store.imageFile(img)
                if (f.exists()) {
                    progress.update("Uploading image ${i + 1}/${adds.size}...")
                    entries += GitHubClient.TreeEntry(p(img), gh.createBlob(f.readBytes()))
                } else store.syncState().pendingCopies[img]?.let { src ->
                    // Relabelled sample whose JPEG only exists on GitHub: copy it to its new path.
                    gh.getFile(p(src), head)?.let { entries += GitHubClient.TreeEntry(p(img), gh.createBlob(it)) }
                }
            }
            val shards = remote.shards.toMutableList()
            if (adds.isNotEmpty()) {
                val shardFile = "features/%06d-%s.json".format(version, stamp())
                val shardBytes = DatasetCodec.encodeShard(FeatureShard(samples = adds)).toByteArray()
                entries += GitHubClient.TreeEntry(p(shardFile), gh.createBlob(shardBytes))
                shards += ShardRef(shardFile, adds.size, DatasetCodec.sha256(shardBytes))
            }
            val newDeleted = (remote.deleted + dels.map { it.id }).distinct()
            dels.forEach { d -> d.image?.let { entries += GitHubClient.TreeEntry(p(it), null) } }

            val liveSamples = store.samples()
            val manifest = DatasetManifest(
                datasetVersion = version,
                updatedAt = iso(Date()),
                sampleCount = liveSamples.size,
                labelCounts = liveSamples.groupingBy { it.label }.eachCount().toSortedMap(),
                shards = shards,
                deleted = newDeleted,
            )
            entries += GitHubClient.TreeEntry(p("manifest.json"), gh.createBlob(DatasetCodec.encodeManifest(manifest).toByteArray()))

            progress.update("Committing...")
            val tree = gh.createTree(baseTree, entries)
            val msg = buildString {
                append("dataset v$version: ")
                append(listOfNotNull(
                    if (adds.isNotEmpty()) "+${adds.size} samples" else null,
                    if (dels.isNotEmpty()) "-${dels.size} deleted" else null,
                ).joinToString(", "))
                append(" (${liveSamples.size} total)\n\n[card-trainer sync]")
            }
            val commit = gh.createCommit(msg, tree, listOf(head))
            try {
                gh.updateRef(cfg.branch, commit)
            } catch (e: GitHubClient.NotFastForward) {
                return@repeat // someone pushed (e.g. a code commit); redo against the new head
            }
            store.markPublished(adds.map { it.id }, dels.map { it.id }, shards, version, manifest.deleted)
            return Result(imported.size, adds.size, dels.size, version, commit, imported, removed)
        }
        error("Could not publish after $MAX_ATTEMPTS attempts (branch kept moving)")
    }

    companion object {
        private const val MAX_ATTEMPTS = 4
        private fun fmt(p: String) = SimpleDateFormat(p, Locale.US).apply { timeZone = TimeZone.getTimeZone("UTC") }
        fun stamp(d: Date = Date()): String = fmt("yyyyMMdd'T'HHmmss'Z'").format(d)
        fun iso(d: Date): String = fmt("yyyy-MM-dd'T'HH:mm:ss'Z'").format(d)
    }
}
