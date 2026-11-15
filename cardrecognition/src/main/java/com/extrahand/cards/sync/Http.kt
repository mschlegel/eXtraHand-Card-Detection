package com.extrahand.cards.sync

import java.io.IOException
import java.net.HttpURLConnection
import java.net.URL

/** Tiny HttpURLConnection wrapper (no extra dependencies for host apps). Blocking: call off the main thread. */
internal object Http {
    class Response(val code: Int, val body: ByteArray, val headers: Map<String, String>) {
        val ok get() = code in 200..299
        fun text() = String(body, Charsets.UTF_8)
    }

    class HttpException(val code: Int, message: String) : IOException("HTTP $code: $message")

    fun request(
        url: String,
        method: String = "GET",
        headers: Map<String, String> = emptyMap(),
        body: ByteArray? = null,
        timeoutMs: Int = 30_000,
    ): Response {
        val c = URL(url).openConnection() as HttpURLConnection
        try {
            setMethod(c, method)
            c.connectTimeout = timeoutMs
            c.readTimeout = timeoutMs * 2
            c.setRequestProperty("User-Agent", "extrahand-cardrecognition")
            headers.forEach { (k, v) -> c.setRequestProperty(k, v) }
            if (body != null) {
                c.doOutput = true
                if (headers.keys.none { it.equals("Content-Type", true) }) c.setRequestProperty("Content-Type", "application/json")
                c.outputStream.use { it.write(body) }
            }
            val code = c.responseCode
            val stream = if (code >= 400) c.errorStream else c.inputStream
            val bytes = stream?.use { it.readBytes() } ?: ByteArray(0)
            val h = c.headerFields.filterKeys { it != null }.mapValues { it.value.joinToString(",") }
            return Response(code, bytes, h)
        } finally {
            c.disconnect()
        }
    }

    fun get(url: String, headers: Map<String, String> = emptyMap()): Response = request(url, headers = headers)

    /** Android's HttpURLConnection accepts PATCH; the desktop JDK one does not, so fall back to reflection. */
    private fun setMethod(c: HttpURLConnection, method: String) {
        try {
            c.requestMethod = method
        } catch (e: java.net.ProtocolException) {
            val f = HttpURLConnection::class.java.getDeclaredField("method")
            f.isAccessible = true
            f.set(c, method)
            // HTTPS connections on the JDK wrap a delegate connection that also needs the method.
            runCatching {
                val d = c.javaClass.getDeclaredField("delegate").apply { isAccessible = true }.get(c)
                if (d is HttpURLConnection) f.set(d, method)
            }
        }
    }
}
