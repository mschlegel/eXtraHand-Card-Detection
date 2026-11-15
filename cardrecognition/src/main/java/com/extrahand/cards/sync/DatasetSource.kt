package com.extrahand.cards.sync

/** Where a consuming app loads the trained dataset from. */
sealed interface DatasetSource {
    /** Stable key used to name the local cache directory. */
    val cacheKey: String

    /** Resolve a dataset-relative path (e.g. "manifest.json") to a URL. */
    fun url(relativePath: String): String

    /** Extra HTTP headers (auth). */
    fun headers(): Map<String, String> = emptyMap()

    /**
     * A GitHub repo. With [token] == null it uses raw.githubusercontent.com (public repos, no rate limit,
     * CDN-cached ~5 min). With a token it uses the REST contents API (works for private repos, always fresh).
     */
    data class GitHub(
        val owner: String,
        val repo: String,
        val branch: String = "main",
        val path: String = "dataset",
        val token: String? = null,
    ) : DatasetSource {
        override val cacheKey get() = "gh_${owner}_${repo}_${branch}_${path.replace('/', '_')}"
        override fun url(relativePath: String): String {
            val p = listOf(path.trim('/'), relativePath.trim('/')).filter { it.isNotEmpty() }.joinToString("/")
            return if (token == null) "https://raw.githubusercontent.com/$owner/$repo/$branch/$p"
            else "https://api.github.com/repos/$owner/$repo/contents/$p?ref=$branch"
        }
        override fun headers(): Map<String, String> =
            if (token == null) emptyMap()
            else mapOf("Authorization" to "Bearer $token", "Accept" to "application/vnd.github.raw+json")
    }

    /** Any static HTTP(S) base URL that serves the dataset layout, e.g. "https://example.com/cards/dataset". */
    data class Http(val baseUrl: String, val extraHeaders: Map<String, String> = emptyMap()) : DatasetSource {
        override val cacheKey get() = "http_" + baseUrl.hashCode().toUInt().toString(16)
        override fun url(relativePath: String) = baseUrl.trimEnd('/') + "/" + relativePath.trimStart('/')
        override fun headers() = extraHeaders
    }

    companion object {
        /** The canonical dataset trained by the eXtraHand Card Trainer app. */
        val DEFAULT = GitHub(owner = "mschlegel", repo = "eXtraHand-Card-Detection", branch = "main", path = "dataset")
    }
}
