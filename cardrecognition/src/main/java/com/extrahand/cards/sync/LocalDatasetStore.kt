package com.extrahand.cards.sync

import com.extrahand.cards.core.DatasetCodec
import com.extrahand.cards.core.ShardRef
import com.extrahand.cards.core.SampleRecord
import kotlinx.serialization.Serializable
import java.io.File

/**
 * On-device training store. The directory layout mirrors the published dataset so images can be uploaded
 * as-is:
 *
 *   <root>/samples.jsonl            append-only log of LogEntry lines (adds and deletes)
 *   <root>/samples/<LABEL>/<id>.jpg card crops
 *   <root>/sync_state.json          what has / has not been published yet
 *
 * Thread-safe (all public methods synchronize on the instance).
 */
class LocalDatasetStore(val root: File) {
    @Serializable
    private data class LogEntry(val op: String, val sample: SampleRecord? = null, val id: String? = null)

    @Serializable
    data class PendingDelete(val id: String, val image: String?)

    @Serializable
    data class SyncState(
        val pendingAdds: List<String> = emptyList(),
        val pendingDeletes: List<PendingDelete> = emptyList(),
        /** Shards already merged from / published to the remote. */
        val knownShards: List<ShardRef> = emptyList(),
        val remoteDatasetVersion: Long = 0,
        val remoteDeleted: List<String> = emptyList(),
        val lastSyncAt: Long = 0,
    )

    private val logFile = File(root, "samples.jsonl")
    private val stateFile = File(root, "sync_state.json")
    private val live = LinkedHashMap<String, SampleRecord>()
    private var state = SyncState()

    init {
        root.mkdirs()
        if (logFile.exists()) logFile.forEachLine { line ->
            if (line.isBlank()) return@forEachLine
            val e = runCatching { DatasetCodec.json.decodeFromString(LogEntry.serializer(), line) }.getOrNull()
                ?: return@forEachLine
            when (e.op) {
                "add" -> e.sample?.let { live[it.id] = it }
                "delete" -> e.id?.let { live.remove(it) }
            }
        }
        if (stateFile.exists()) state = runCatching {
            DatasetCodec.json.decodeFromString(SyncState.serializer(), stateFile.readText())
        }.getOrDefault(SyncState())
    }

    @Synchronized fun samples(): List<SampleRecord> = live.values.toList()
    @Synchronized fun get(id: String): SampleRecord? = live[id]
    @Synchronized fun syncState(): SyncState = state
    @Synchronized fun pendingAddRecords(): List<SampleRecord> = state.pendingAdds.mapNotNull { live[it] }
    @Synchronized fun pendingCount(): Int = state.pendingAdds.size + state.pendingDeletes.size

    fun imageFile(relativePath: String): File = File(root, relativePath)

    /** Save a new labelled sample (and its JPEG, if given). Marks it pending for the next sync. */
    @Synchronized fun add(record: SampleRecord, jpeg: ByteArray?) {
        if (jpeg != null && record.image != null) imageFile(record.image).apply { parentFile?.mkdirs() }.writeBytes(jpeg)
        append(LogEntry("add", sample = record))
        live[record.id] = record
        saveState(state.copy(pendingAdds = state.pendingAdds + record.id))
    }

    /** Delete a sample. Unpublished samples vanish locally; published ones become a remote tombstone on sync. */
    @Synchronized fun delete(id: String): Boolean {
        val rec = live.remove(id) ?: return false
        append(LogEntry("delete", id = id))
        if (id in state.pendingAdds) {
            rec.image?.let { imageFile(it).delete() }
            saveState(state.copy(pendingAdds = state.pendingAdds - id))
        } else {
            rec.image?.let { imageFile(it).delete() }
            saveState(state.copy(pendingDeletes = state.pendingDeletes + PendingDelete(id, rec.image)))
        }
        return true
    }

    // ---- used by GitHubDatasetSync ----

    /** Merge samples published elsewhere (e.g. after reinstall). Returns the records actually added. */
    @Synchronized internal fun importRemote(shard: ShardRef, records: List<SampleRecord>): List<SampleRecord> {
        val deleted = state.remoteDeleted.toHashSet() + state.pendingDeletes.map { it.id }
        val added = records.filter { it.id !in live && it.id !in deleted }
        added.forEach { append(LogEntry("add", sample = it)); live[it.id] = it }
        saveState(state.copy(knownShards = (state.knownShards + shard).distinctBy { it.file }))
        return added
    }

    /** Apply remote tombstones. Returns ids removed locally. */
    @Synchronized internal fun applyRemoteDeletes(ids: List<String>): List<String> {
        val removed = ids.filter { it in live && it !in state.pendingAdds }
        removed.forEach { id -> live.remove(id)?.image?.let { imageFile(it).delete() }; append(LogEntry("delete", id = id)) }
        saveState(state.copy(remoteDeleted = (state.remoteDeleted + ids).distinct()))
        return removed
    }

    @Synchronized internal fun markPublished(
        addedIds: Collection<String>, deletedIds: Collection<String>, shards: List<ShardRef>,
        datasetVersion: Long, remoteDeleted: List<String>,
    ) {
        saveState(state.copy(
            pendingAdds = state.pendingAdds - addedIds.toSet(),
            pendingDeletes = state.pendingDeletes.filter { it.id !in deletedIds },
            knownShards = shards,
            remoteDatasetVersion = datasetVersion,
            remoteDeleted = remoteDeleted,
            lastSyncAt = System.currentTimeMillis(),
        ))
    }

    @Synchronized internal fun markPulled(datasetVersion: Long) {
        saveState(state.copy(remoteDatasetVersion = datasetVersion, lastSyncAt = System.currentTimeMillis()))
    }

    private fun append(e: LogEntry) =
        logFile.appendText(DatasetCodec.json.encodeToString(LogEntry.serializer(), e) + "\n")

    private fun saveState(s: SyncState) {
        state = s
        val tmp = File(root, "sync_state.json.tmp")
        tmp.writeText(DatasetCodec.json.encodeToString(SyncState.serializer(), s))
        if (!tmp.renameTo(stateFile)) { stateFile.delete(); tmp.renameTo(stateFile) }
    }
}
