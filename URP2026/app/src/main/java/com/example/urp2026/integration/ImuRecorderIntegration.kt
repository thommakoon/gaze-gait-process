package com.example.urp2026.integration

import android.content.Context
import android.os.Environment
import com.example.urp2026.neon.NeonCompanionApi
import com.example.urp2026.neon.NeonHttpResult
import com.example.urp2026.neon.describe
import com.example.urp2026.pcbridge.PcActionResult
import com.example.urp2026.pcbridge.PcBridgeState
import com.example.urp2026.pcbridge.PcCommandHttpServer
import com.example.urp2026.qtpy.QtPyBridgeState
import com.example.urp2026.qtpy.QtPyFirmwareState
import com.example.urp2026.qtpy.QtPyLineRecord
import com.example.urp2026.qtpy.QtPySerialBridge
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.SupervisorJob
import kotlinx.coroutines.cancel
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.launch
import org.json.JSONArray
import org.json.JSONObject
import java.io.File
import java.util.concurrent.atomic.AtomicReference
import java.util.concurrent.atomic.AtomicBoolean

/**
 * High-level wiring (analogous to Termux `dual_imu_recorder_core.run_recorder` + entry script).
 * QT Py USB serial + Neon REST; optional LAN HTTP bridge for PC wireless CMD:* / record control.
 */
class ImuRecorderIntegration(
    context: Context,
    private val neon: NeonCompanionApi = NeonCompanionApi(),
) {
    private val appContext = context.applicationContext
    private val qtPy = QtPySerialBridge(context)
    private val csvRecorder = DualImuCsvRecorder(rootDir = resolveRecordingRoot())
    private val eventScope = CoroutineScope(SupervisorJob() + Dispatchers.IO)

    /** Last Neon recording id from a successful `recording:start` (for UI / debugging). */
    private val activeRecordingId = AtomicReference<String?>(null)
    private val combinedSessionActive = AtomicBoolean(false)
    private val csvSessionInfo = AtomicReference<String?>(null)

    private val pcBridge = PcCommandHttpServer(
        onQtPyCommand = { cmd -> qtPy.writeCommand(cmd) },
        qtPyConnected = { qtPy.isReady() },
        qtPyFirmwareState = { qtPy.firmwareState().name },
        qtPyStreaming = { qtPy.isStreaming() },
        recordingActive = { combinedSessionActive.get() },
        onRecordStart = { startBothForPc() },
        onRecordStop = { stopBothForPc() },
    )

    init {
        qtPy.setLineRecordListener { record ->
            onQtPyLineRecord(record)
        }
    }

    companion object {
        /**
         * Send a Neon `imu_seq_<seq>` event every N IMU lines while a combined session is active.
         * Provides drift-checkable alignment anchors between QT Py IMU and Neon clocks
         * without flooding the Companion `/api/event` endpoint.
         * At 200 Hz ≈ one event per 5 s; at 100 Hz ≈ one per 10 s (same N).
         */
        const val HEARTBEAT_EVERY_N_LINES: Long = 1000L

        /** Current Unix-epoch nanoseconds; what Neon `/api/event` expects in `timestamp`. */
        private fun utcNanosNow(): Long = System.currentTimeMillis() * 1_000_000L
    }

    fun qtPyReady(): Boolean = qtPy.isReady()
    fun qtPyIsStreaming(): Boolean = qtPy.isStreaming()
    fun qtPyStateFlow(): StateFlow<QtPyBridgeState> = qtPy.state
    fun pcBridgeStateFlow(): StateFlow<PcBridgeState> = pcBridge.state

    suspend fun qtPyConnectResult(): String = qtPy.connect()
    suspend fun qtPyDisconnectResult(): String = qtPy.disconnect()
    /** After USB permission dialog / resume: open if already permitted. Null = nothing to do. */
    suspend fun qtPyTryConnectIfPermitted(): String? = qtPy.tryConnectIfPermitted()
    suspend fun qtPySendCommand(command: String): String = qtPy.writeCommand(command)
    fun qtPyClearMonitor() = qtPy.clearRecentLines()

    fun pcBridgeStart(): String = pcBridge.start()
    fun pcBridgeStop(): String = pcBridge.stop()
    fun pcBridgeRunning(): Boolean = pcBridge.isRunning()

    fun csvSessionInfo(): String? = csvSessionInfo.get()
    fun close() {
        combinedSessionActive.set(false)
        csvRecorder.close()
        qtPy.setLineRecordListener(null)
        pcBridge.close()
        qtPy.close()
        eventScope.cancel()
    }

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
        val r = neon.neonSendEvent("android_probe", utcNanosNow())
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
            val ev = neon.neonSendEvent("android_probe_smoke", utcNanosNow())
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
                val t = utcNanosNow()
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
        val t = utcNanosNow()
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
     * 1) Require QT Py USB connected and firmware STATE:STREAMING.
     * 2) Start Neon recording + imu_stream_start marker.
     */
    suspend fun startBothIfQtPyConnected(): String {
        if (!qtPyReady()) {
            return "Connect QT Py first (USB serial) before starting both recordings."
        }
        val fw = qtPy.firmwareState()
        if (!qtPy.isStreaming()) {
            return "QT Py must be STREAMING before Start both " +
                "(firmware=$fw). Calibrate then CMD Start (or button), then retry."
        }
        if (combinedSessionActive.get()) {
            return "Both recording session is already active."
        }
        val (lfFile, rfFile) = csvRecorder.startNewSession()
        csvSessionInfo.set("CSV: LF=${lfFile.name}, RF=${rfFile.name} @ ${lfFile.parentFile?.absolutePath}")
        val result = neonStartFootSessionLikeRecorder()
        if (activeRecordingId.get() != null) {
            combinedSessionActive.set(true)
            return "$result\n${csvSessionInfo.get()}"
        }
        csvRecorder.close()
        csvSessionInfo.set(null)
        return result
    }

    /** Same checks as [startBothIfQtPyConnected]; structured for PC HTTP bridge. */
    suspend fun startBothForPc(): PcActionResult {
        val message = startBothIfQtPyConnected()
        if (combinedSessionActive.get()) {
            return PcActionResult(ok = true, message = message, httpStatus = 200)
        }
        val status = when {
            !qtPyReady() -> 503
            qtPy.firmwareState() != QtPyFirmwareState.STREAMING -> 409
            message.contains("already active", ignoreCase = true) -> 409
            else -> 502 // Neon / CSV start failed
        }
        return PcActionResult(ok = false, message = message, httpStatus = status)
    }

    /** Same as app Stop both; structured for PC HTTP bridge. */
    suspend fun stopBothForPc(): PcActionResult {
        if (!combinedSessionActive.get()) {
            return PcActionResult(
                ok = false,
                message = "No active both-recording session to stop.",
                httpStatus = 409,
            )
        }
        val message = stopBothSession()
        return PcActionResult(ok = true, message = message, httpStatus = 200)
    }

    /**
     * Recorder-aligned stop:
     * 1) imu_stream_end marker
     * 2) recording:stop_and_save
     */
    suspend fun stopBothSession(): String {
        val result = neonStopFootSessionLikeRecorder()
        combinedSessionActive.set(false)
        csvRecorder.close()
        return buildString {
            append(result)
            csvSessionInfo.get()?.let { append("\n").append(it) }
        }
    }

    private fun onQtPyLineRecord(record: QtPyLineRecord) {
        if (!combinedSessionActive.get()) return
        val parsed = parseDualImuCsvLine(record.rawLine) ?: return
        csvRecorder.appendIcm20(
            seq = parsed.seq,
            timeUs = parsed.timeUs,
            timeUsExtended = record.timeUsExtended,
            recvElapsedNs = record.recvElapsedRealtimeNs,
            tUtcNs = record.recvWallTimeNs,
            lf = parsed.lf,
            rf = parsed.rf,
        )
        maybeSendHeartbeat(parsed.seq, record.recvWallTimeNs)
    }

    /**
     * Fire-and-forget a Neon `imu_seq_<seq>` event every [HEARTBEAT_EVERY_N_LINES] IMU rows.
     * Uses the row's own phone-UTC receive time as the event timestamp so the marker is
     * anchored to a specific IMU sample, not to the (slightly later) HTTP POST instant.
     */
    private fun maybeSendHeartbeat(seq: Long, tUtcNs: Long) {
        if (seq <= 0L) return
        if (seq % HEARTBEAT_EVERY_N_LINES != 0L) return
        eventScope.launch {
            neon.neonSendEvent("imu_seq_$seq", tUtcNs)
        }
    }

    /**
     * Dual IMU CSV from QT Py: **20 fields** (LF/RF accel+gyro+mag) or **14 fields**
     * (accel+gyro only, e.g. sketch_usb_dual_timer / sketch_usb_dual_100).
     * Parsed arrays keep 9 floats per foot for a stable shape; [DualImuCsvRecorder] writes
     * only meta + Acc + Gyr (mag is not persisted to CSV).
     */
    private fun parseDualImuCsvLine(raw: String): ParsedIcm20? {
        val parts = raw.split(',')
        val seq = parts.getOrNull(0)?.toLongOrNull() ?: return null
        val timeUs = parts.getOrNull(1)?.toLongOrNull() ?: return null
        val vals = parts.drop(2).map { it.toDoubleOrNull() ?: return null }
        return when (parts.size) {
            20 -> {
                if (vals.size != 18) return null
                ParsedIcm20(
                    seq = seq,
                    timeUs = timeUs,
                    lf = vals.subList(0, 9).toDoubleArray(),
                    rf = vals.subList(9, 18).toDoubleArray(),
                )
            }
            14 -> {
                if (vals.size != 12) return null
                val lf = DoubleArray(9) { i ->
                    if (i < 6) vals[i] else 0.0
                }
                val rf = DoubleArray(9) { i ->
                    if (i < 6) vals[6 + i] else 0.0
                }
                ParsedIcm20(seq = seq, timeUs = timeUs, lf = lf, rf = rf)
            }
            else -> null
        }
    }

    private data class ParsedIcm20(
        val seq: Long,
        val timeUs: Long,
        val lf: DoubleArray,
        val rf: DoubleArray,
    )

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

    private fun resolveRecordingRoot(): File {
        // Termux-like structure under app external documents: Documents/thom/data
        val externalDocs = appContext.getExternalFilesDir(Environment.DIRECTORY_DOCUMENTS)
        return if (externalDocs != null) {
            File(externalDocs, "thom/data").apply { mkdirs() }
        } else {
            File(appContext.filesDir, "recordings/thom/data").apply { mkdirs() }
        }
    }
}
