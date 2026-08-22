package com.example.urp2026.integration

import android.content.Context

/**
 * Process-wide holder for [ImuRecorderIntegration].
 * Owned by [com.example.urp2026.service.RecordingForegroundService] so USB / CSV / PC bridge
 * survive Activity backgrounding (Compose dispose no longer closes the recorder).
 */
object RecorderRuntime {
    @Volatile
    private var instance: ImuRecorderIntegration? = null
    private val lock = Any()

    fun get(context: Context): ImuRecorderIntegration {
        instance?.let { return it }
        return synchronized(lock) {
            instance ?: ImuRecorderIntegration(context.applicationContext).also { instance = it }
        }
    }

    fun peek(): ImuRecorderIntegration? = instance

    fun release() {
        synchronized(lock) {
            try {
                instance?.close()
            } catch (_: Exception) {
            }
            instance = null
        }
    }
}
