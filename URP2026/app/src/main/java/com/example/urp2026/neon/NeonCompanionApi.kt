package com.example.urp2026.neon

import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import okhttp3.MediaType.Companion.toMediaType
import okhttp3.OkHttpClient
import okhttp3.Request
import okhttp3.RequestBody.Companion.toRequestBody
import org.json.JSONObject
import java.util.concurrent.TimeUnit

/**
 * Kotlin mirror of [motorola/android/neon_companion_api.py].
 * All public calls are [suspend] and perform network I/O on [Dispatchers.IO]
 * so callers cannot accidentally block the UI thread.
 */
class NeonCompanionApi(
    private val baseUrl: String = DEFAULT_BASE_URL,
    private val client: OkHttpClient = createStandardClient(),
) {
    companion object {
        const val DEFAULT_BASE_URL = "http://localhost:8080/api"

        /** Aligns roughly with Python `timeout=3` on status / event. */
        private fun createShortTimeoutClient(): OkHttpClient = OkHttpClient.Builder()
            .connectTimeout(3, TimeUnit.SECONDS)
            .readTimeout(3, TimeUnit.SECONDS)
            .writeTimeout(3, TimeUnit.SECONDS)
            .callTimeout(6, TimeUnit.SECONDS)
            .build()

        /** Aligns roughly with Python `timeout=5` on recording start/stop. */
        private fun createStandardClient(): OkHttpClient = OkHttpClient.Builder()
            .connectTimeout(5, TimeUnit.SECONDS)
            .readTimeout(5, TimeUnit.SECONDS)
            .writeTimeout(5, TimeUnit.SECONDS)
            .callTimeout(10, TimeUnit.SECONDS)
            .build()
    }

    private val shortTimeoutClient: OkHttpClient = createShortTimeoutClient()

    suspend fun neonStatus(): NeonHttpResult<JSONObject> = withContext(Dispatchers.IO) {
        val req = Request.Builder().url("$baseUrl/status").get().build()
        executeJsonIfSuccessful(shortTimeoutClient, req)
    }

    suspend fun neonStartRecording(): NeonHttpResult<String> = withContext(Dispatchers.IO) {
        val empty = ByteArray(0).toRequestBody(contentType = null)
        val req = Request.Builder()
            .url("$baseUrl/recording:start")
            .post(empty)
            .build()
        try {
            client.newCall(req).execute().use { resp ->
                val body = resp.body?.string()
                if (!resp.isSuccessful) {
                    return@withContext NeonHttpResult.Err(
                        httpCode = resp.code,
                        message = "recording:start not OK",
                        bodySnippet = body,
                    )
                }
                if (body.isNullOrBlank()) {
                    return@withContext NeonHttpResult.Err(
                        httpCode = resp.code,
                        message = "Empty JSON body",
                        bodySnippet = null,
                    )
                }
                val id = JSONObject(body).optString("id", "unknown").ifEmpty { "unknown" }
                NeonHttpResult.Ok(id)
            }
        } catch (e: Exception) {
            NeonHttpResult.Err(null, e.message ?: e.javaClass.simpleName, null)
        }
    }

    suspend fun neonStopRecording(): NeonHttpResult<Unit> = withContext(Dispatchers.IO) {
        val empty = ByteArray(0).toRequestBody(contentType = null)
        val req = Request.Builder()
            .url("$baseUrl/recording:stop_and_save")
            .post(empty)
            .build()
        postExpectSuccess(client, req)
    }

    suspend fun neonSendEvent(name: String, timestampNs: Long? = null): NeonHttpResult<Unit> =
        withContext(Dispatchers.IO) {
            val json = JSONObject().put("name", name)
            if (timestampNs != null) {
                json.put("timestamp", timestampNs)
            }
            val mediaType = "application/json; charset=utf-8".toMediaType()
            val reqBody = json.toString().toRequestBody(mediaType)
            val req = Request.Builder()
                .url("$baseUrl/event")
                .post(reqBody)
                .build()
            postExpectSuccess(shortTimeoutClient, req)
        }

    private fun executeJsonIfSuccessful(
        http: OkHttpClient,
        req: Request,
    ): NeonHttpResult<JSONObject> = try {
        http.newCall(req).execute().use { resp ->
            val body = resp.body?.string()
            if (!resp.isSuccessful) {
                NeonHttpResult.Err(
                    httpCode = resp.code,
                    message = "GET /status not OK",
                    bodySnippet = body,
                )
            } else if (body.isNullOrBlank()) {
                NeonHttpResult.Err(resp.code, "Empty JSON body", null)
            } else {
                NeonHttpResult.Ok(JSONObject(body))
            }
        }
    } catch (e: Exception) {
        NeonHttpResult.Err(null, e.message ?: e.javaClass.simpleName, null)
    }

    private fun postExpectSuccess(http: OkHttpClient, req: Request): NeonHttpResult<Unit> =
        try {
            http.newCall(req).execute().use { resp ->
                if (resp.isSuccessful) {
                    NeonHttpResult.Ok(Unit)
                } else {
                    val body = resp.body?.string()
                    NeonHttpResult.Err(
                        httpCode = resp.code,
                        message = "POST not OK",
                        bodySnippet = body,
                    )
                }
            }
        } catch (e: Exception) {
            NeonHttpResult.Err(null, e.message ?: e.javaClass.simpleName, null)
        }
}
