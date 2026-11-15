package com.extrahand.cards.core

import com.extrahand.cards.sync.DatasetSource
import com.extrahand.cards.sync.GitHubConfig
import com.extrahand.cards.sync.GitHubDatasetSync
import com.extrahand.cards.sync.LocalDatasetStore
import com.extrahand.cards.sync.RemoteDatasetLoader
import org.junit.Assert.assertEquals
import org.junit.Assume.assumeTrue
import org.junit.Test
import java.nio.file.Files

/**
 * End-to-end sync against a real GitHub branch. Skipped unless these env vars are set:
 *   CARD_SYNC_TOKEN, CARD_SYNC_REPO (owner/name), CARD_SYNC_BRANCH (a THROWAWAY branch - it gets commits)
 */
class GitHubSyncLiveTest {
    @Test fun pushPullDelete() {
        val token = System.getenv("CARD_SYNC_TOKEN")
        val repo = System.getenv("CARD_SYNC_REPO")
        val branch = System.getenv("CARD_SYNC_BRANCH")
        assumeTrue(token != null && repo != null && branch != null && branch != "main")
        val (owner, name) = repo!!.split("/")
        val cfg = GitHubConfig(owner, name, branch!!, "dataset", token!!)
        val f = DatasetCodec.encodeVector(FloatArray(LumaGridV1.dim) { (it % 7) - 3f })

        // Device A records two samples and pushes.
        val a = LocalDatasetStore(Files.createTempDirectory("a").toFile())
        val base = a.syncState().remoteDatasetVersion
        a.add(SampleRecord("live-1", "AS", f, 1, image = "samples/AS/live-1.jpg", source = "live-test"), byteArrayOf(-1, -40, -1))
        a.add(SampleRecord("live-2", "KD", f, 2, source = "live-test"), null)
        val r1 = GitHubDatasetSync(a, cfg).sync()
        assertEquals(2, r1.pushed)

        // Fresh install pulls them.
        val b = LocalDatasetStore(Files.createTempDirectory("b").toFile())
        val r2 = GitHubDatasetSync(b, cfg).sync()
        assertEquals(setOf("live-1", "live-2"), b.samples().map { it.id }.filter { it.startsWith("live-") }.toSet())
        assertEquals(0, r2.pushed)

        // Delete one on B, push tombstone; A sees it disappear.
        b.delete("live-1")
        assertEquals(1, GitHubDatasetSync(b, cfg).sync().deleted)
        val r4 = GitHubDatasetSync(a, cfg).sync()
        assertEquals(listOf("live-1"), r4.removedIds)

        // Read path used by consumer apps.
        val loaded = RemoteDatasetLoader(DatasetSource.GitHub(owner, name, branch, token = token), Files.createTempDirectory("c").toFile()).load()
        val live = DatasetCodec.liveSamples(loaded.manifest, loaded.shards).map { it.id }
        assertEquals(true, "live-2" in live && "live-1" !in live)
        println("live test ok: base v$base -> v${loaded.manifest.datasetVersion}, ${live.size} live samples")
    }
}
