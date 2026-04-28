package com.example.urp2026.neon

/**
 * Typed outcome for Neon REST calls (replaces silent null / boolean-only failures).
 */
sealed interface NeonHttpResult<out T> {
    data class Ok<T>(val value: T) : NeonHttpResult<T>

    data class Err(
        val httpCode: Int?,
        val message: String,
        val bodySnippet: String? = null,
    ) : NeonHttpResult<Nothing>

    fun errorOrNull(): Err? = this as? Err
}

fun NeonHttpResult.Err.describe(prefix: String): String = buildString {
    append(prefix)
    append(" — ")
    append(message)
    httpCode?.let { append(" (HTTP ").append(it).append(")") }
    bodySnippet?.takeIf { it.isNotBlank() }?.let { snippet ->
        append(" [")
        val oneLine = snippet.replace('\n', ' ').trim()
        append(if (oneLine.length > 160) oneLine.take(160) + "…" else oneLine)
        append("]")
    }
}
