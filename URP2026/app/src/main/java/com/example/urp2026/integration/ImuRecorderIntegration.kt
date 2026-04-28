package com.example.urp2026.integration

import android.content.Context
import com.example.urp2026.neon.NeonCompanionApi
import com.example.urp2026.neon.NeonHttpResult
import com.example.urp2026.neon.describe
import com.example.urp2026.qtpy.QtPyBridgeState
import com.example.urp2026.qtpy.QtPySerialBridge
import kotlinx.coroutines.flow.StateFlow
import org.json.JSONArray
import org.json.JSONObject
import java.util.concurrent.atomic.AtomicReference
import java.util.concurrent.atomic.AtomicBoolean

/**
 * High-level wiring (analogous to Termux `dual_imu_recorder_core.run_recorder` + entry script).
 * QT Py is stubbed; Neon REST is polished here before USB serial lands in [QtPySerialBridge].
 */
class ImuRecorderIntegration(
    context: Context,
    private val neon: NeonCompanionApi = NeonCompanionApi(),
) {
    private val qtPy = QtPySerialBridge(context)

    /** Last Neon recording id from a successful `recording:start` (for UI / debugging). */
    private val activeRecordingId = AtomicReference<String?>(null)
    private val combinedSessionActive = AtomicBoolean(false)

    fun qtPyReady(): Boolean = qtPy.isReady()
    fun qtPyStateFlow(): StateFlow<QtPyBridgeState> = qtPy.state

    suspend fun qtPyConnectResult(): String = qtPy.connect()
    suspend fun qtPyDisconnectResult(): String = qtPy.disconnect()
    fun qtPyClearMonitor() = qtPy.clearRecentLines()
    fun close() = qtPy.close()

    fun neonActiveRecordingId(): String? = activeRecordingId.get()
    fun isCombinedSessionActive(): Boolean = combinedSessionActive.get()

    suspend fun neonStatusSummary(): String = when (val r = neon.neonStatus()) {
        is NeonHttpResult.Ok -> summarizeStatus(r.value)
        is NeonHttpResult.Err -> r.describe("GET /status")
    }

    suspend fun neonStartRecordingResult(): String = when (val r = neon.neonStartRecording()) {
        is NeonHttpResult.Ok -> {
            activeRecordingId.set(r.value)
            "recording:start OK, id=${r.value}"
        }
        is NeonHttpResult.Err -> {
            activeRecordingId.set(null)
            r.describe("POST recording:start")
        }
    }

    suspend fun neonStopRecordingResult(): String = when (val r = neon.neonStopRecording()) {
        is NeonHttpResult.Ok -> {
            activeRecordingId.set(null)
            "recording:stop_and_save OK."
        }
        is NeonHttpResult.Err -> r.describe("POST recording:stop_and_save")
    }

    suspend fun neonSendProbeEvent(): String = when (
        val r = neon.neonSendEvent("android_probe", System.nanoTime())
    ) {
        is NeonHttpResult.Ok -> "POST /event android_probe OK."
        is NeonHttpResult.Err -> r.describe("POST /event")
    }

    /**
     * One tap: [neonStatus] then a probe event — quick sanity check without starting a recording.
     */
    suspend fun neonSmokeTest(): String = when (val st = neon.neonStatus()) {
        is NeonHttpResult.Err -> st.describe("Smoke (status)")
        is NeonHttpResult.Ok -> when (
            val ev = neon.neonSendEvent("android_probe_smoke", System.nanoTime())
        ) {
            is NeonHttpResult.Err ->
                "Status OK (${summarizeStatus(st.value)}) but " + ev.describe("Smoke (event)")
            is NeonHttpResult.Ok ->
                "Smoke OK — ${summarizeStatus(st.value)}; event android_probe_smoke OK."
        }
    }

    /**
     * Same order as Python when firmware enters STREAMING: `recording:start` then `imu_stream_start`.
     */
    suspend fun neonStartFootSessionLikeRecorder(): String {
        when (val start = neon.neonStartRecording()) {
            is NeonHttpResult.Err -> {
                activeRecordingId.set(null)
                return start.describe("Session start (recording:start)")
            }
            is NeonHttpResult.Ok -> {
                activeRecordingId.set(start.value)
                val t = System.nanoTime()
                when (val ev = neon.neonSendEvent("imu_stream_start", t)) {
                    is NeonHttpResult.Ok ->
                        return "Session started — id=${start.value}, imu_stream_start OK."
                    is NeonHttpResult.Err ->
                        return "WARNING: recording id=${start.value} but imu_stream_start failed — " +
                            ev.describe("imu_stream_start") +
                            " (Neon may be recording without marker; use Stop session.)"
                }
            }
        }
    }

    /**
     * Same order as Python when stream ends: `imu_stream_end` then `recording:stop_and_save`.
     */
    suspend fun neonStopFootSessionLikeRecorder(): String {
        val t = System.nanoTime()
        val endEv = neon.neonSendEvent("imu_stream_end", t)
        val stop = neon.neonStopRecording()
        activeRecordingId.set(null)
        val evLine = when (endEv) {
            is NeonHttpResult.Ok -> "imu_stream_end OK."
            is NeonHttpResult.Err -> endEv.describe("imu_stream_end")
        }
        val stopLine = when (stop) {
            is NeonHttpResult.Ok -> "recording:stop_and_save OK."
            is NeonHttpResult.Err -> stop.describe("recording:stop_and_save")
        }
        return "$evLine $stopLine"
    }

    /**
     * Recorder-aligned flow:
     * 1) Require QT Py already connected over USB serial.
     * 2) Start Neon recording + imu_stream_start marker.
     */
    suspend fun startBothIfQtPyConnected(): String {
        if (!qtPyReady()) {
            return "Connect QT Py first (USB serial) before starting both recordings."
        }
        if (combinedSessionActive.get()) {
            return "Both recording session is already active."
        }
        val result = neonStartFootSessionLikeRecorder()
        if (activeRecordingId.get() != null) {
            combinedSessionActive.set(true)
        }
        return result
    }

    /**
     * Recorder-aligned stop:
     * 1) imu_stream_end marker
     * 2) recording:stop_and_save
     */
    suspend fun stopBothSession(): String {
        val result = neonStopFootSessionLikeRecorder()
        combinedSessionActive.set(false)
        return result
    }

    /**
     * Short human-readable status. Handles:
     * - Older shape: `{ "phone": { "device_name", "battery_level" } }`
     * - Companion shape: `{ "message", "result": [ { "data": { "sensor", "connected", ... } } ] }`
     */
    private fun summarizeStatus(status: JSONObject): String {
        status.optJSONObject("phone")?.let { phone ->
            val name = phone.optString("device_name", "?")
            val bat = phone.opt("battery_level")?.toString() ?: "?"
            return "device_name=$name, battery=$bat%"
        }

        val message = status.optString("message", "")
        val result = status.optJSONArray("result")
        if (result != null) {
            return summarizeResultArray(message, result)
        }

        val keys = status.keys().asSequence().toList()
        return buildString {
            append("API OK (no `phone` / `result[]`). Keys: ")
            append(keys.take(8).joinToString(", "))
            if (keys.size > 8) append("…")
        }
    }

    private fun summarizeResultArray(message: String, result: JSONArray): String {
        var total = 0
        var connected = 0
        val sensors = linkedMapOf<String, Boolean>()
        for (i in 0 until result.length()) {
            val item = result.optJSONObject(i) ?: continue
            val data = item.optJSONObject("data") ?: continue
            total++
            val isConn = data.optBoolean("connected", false)
            if (isConn) connected++
            val sensor = data.optString("sensor", "?")
            if (sensor != "?") {
                sensors[sensor] = (sensors[sensor] == true) || isConn
            }
        }
        val sensorLine = if (sensors.isEmpty()) "" else sensors.entries.joinToString(", ") { (k, v) ->
            "$k=${if (v) "on" else "off"}"
        }
        return buildString {
            if (message.isNotEmpty()) append(message).append(" — ")
            append(total).append(" stream(s), ")
            append(connected).append(" connected")
            if (sensorLine.isNotEmpty()) append(" — ").append(sensorLine)
            if (connected == 0) {
                append(". If glasses are on: open Neon Companion and complete device connection.")
            }
        }
    }
}
