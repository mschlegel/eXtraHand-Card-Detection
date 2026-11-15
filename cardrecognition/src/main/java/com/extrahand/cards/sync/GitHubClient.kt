package com.extrahand.cards.sync

import kotlinx.serialization.json.Json
import kotlinx.serialization.json.JsonArray
import kotlinx.serialization.json.JsonNull
import kotlinx.serialization.json.JsonObject
import kotlinx.serialization.json.JsonPrimitive
import kotlinx.serialization.json.boolean
import kotlinx.serialization.json.buildJsonArray
import kotlinx.serialization.json.buildJsonObject
import kotlinx.serialization.json.jsonObject
import kotlinx.serialization.json.jsonPrimitive
import kotlinx.serialization.json.put
import java.net.URLEncoder
import java.util.Base64

/**
 * Minimal GitHub REST client for publishing many files in ONE commit via the Git Data API
 * (blobs -> tree -> commit -> fast-forward ref). Blocking; call off the main thread.
 *
 * The token needs "Contents: read & write" on the repo (fine-grained PAT) or the classic "repo"/"public_repo" scope.
 */
class GitHubClient(val owner: String, val repo: String, private val token: String) {
    private val api = "https://api.github.com/repos/$owner/$repo"
    private val json = Json { ignoreUnknownKeys = true }

    data class TreeEntry(val path: String, val blobSha: String?) // blobSha == null deletes the path

    class NotFastForward : Exception("Branch moved while publishing")

    private fun headers(accept: String = "application/vnd.github+json") = mapOf(
        "Authorization" to "Bearer $token",
        "Accept" to accept,
        "X-GitHub-Api-Version" to "2022-11-28",
    )

    private fun call(method: String, path: String, body: JsonObject? = null): JsonObject {
        val r = Http.request(api + path, method, headers(), body?.toString()?.toByteArray())
        if (!r.ok) throw Http.HttpException(r.code, "$method $path: ${r.text().take(300)}")
        return json.parseToJsonElement(r.text()).jsonObject
    }

    /** Returns true if the token can push to the repo. */
    fun canPush(): Boolean {
        val o = call("GET", "")
        return (o["permissions"] as? JsonObject)?.get("push")?.jsonPrimitive?.boolean == true
    }

    fun headCommit(branch: String): String =
        call("GET", "/git/ref/heads/$branch")["object"]!!.jsonObject["sha"]!!.jsonPrimitive.content

    fun commitTree(commitSha: String): String =
        call("GET", "/git/commits/$commitSha")["tree"]!!.jsonObject["sha"]!!.jsonPrimitive.content

    /** Raw file bytes at [ref], or null if it does not exist. Works for files up to 100 MB. */
    fun getFile(path: String, ref: String): ByteArray? {
        val p = path.split('/').joinToString("/") { URLEncoder.encode(it, "UTF-8").replace("+", "%20") }
        val r = Http.request("$api/contents/$p?ref=$ref", headers = headers("application/vnd.github.raw+json"))
        if (r.code == 404) return null
        if (!r.ok) throw Http.HttpException(r.code, "GET $path: ${r.text().take(300)}")
        return r.body
    }

    fun createBlob(bytes: ByteArray): String = call("POST", "/git/blobs", buildJsonObject {
        put("content", Base64.getEncoder().encodeToString(bytes))
        put("encoding", "base64")
    })["sha"]!!.jsonPrimitive.content

    fun createTree(baseTree: String, entries: List<TreeEntry>): String = call("POST", "/git/trees", buildJsonObject {
        put("base_tree", baseTree)
        put("tree", buildJsonArray {
            entries.forEach { e ->
                add(JsonObject(mapOf(
                    "path" to JsonPrimitive(e.path),
                    "mode" to JsonPrimitive("100644"),
                    "type" to JsonPrimitive("blob"),
                    "sha" to (e.blobSha?.let { JsonPrimitive(it) } ?: JsonNull),
                )))
            }
        })
    })["sha"]!!.jsonPrimitive.content

    fun createCommit(message: String, tree: String, parents: List<String>): String =
        call("POST", "/git/commits", buildJsonObject {
            put("message", message)
            put("tree", tree)
            put("parents", JsonArray(parents.map { JsonPrimitive(it) }))
        })["sha"]!!.jsonPrimitive.content

    /** Fast-forward [branch] to [commitSha]. Throws [NotFastForward] if someone else pushed first. */
    fun updateRef(branch: String, commitSha: String) {
        val body = buildJsonObject { put("sha", commitSha); put("force", false) }
        val r = Http.request("$api/git/refs/heads/$branch", "PATCH", headers(), body.toString().toByteArray())
        if (r.code == 422 || r.code == 409) throw NotFastForward()
        if (!r.ok) throw Http.HttpException(r.code, "PATCH ref: ${r.text().take(300)}")
    }
}
