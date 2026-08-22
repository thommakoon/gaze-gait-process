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
import com.example.urp2026.qtpy.QtPyUsbIssue
import com.example.urp2026.service.RecordingForegroundService
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
        onHealthSnapshot = { buildPcHealthSnapshot() },
        qtPyConnected = { qtPy.isReady() },
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

        /** If STREAMING but no IMU CSV for this long, report imu_stalled=true. */
        const val IMU_STALL_MS: Long = 500L

        /** Rolling Acc quality window (~2 s at 100 Hz). */
        const val ACC_WINDOW: Int = 200

        /** Known bad fused Acc freeze seen on QT Py dual IMU. */
        const val STUCK_ACC_X: Double = 77.213
        const val STUCK_ACC_Y: Double = 77.227
        const val STUCK_EPS: Double = 0.05

        /** Acc_Z treated as dead if |z| below this (m/s^2). */
        const val ACC_Z_DEAD_EPS: Double = 0.05

        /** Fail Acc quality if stuck or Acc_Z-dead fraction exceeds this. */
        const val ACC_BAD_FRAC: Double = 0.5

        /** Current Unix-epoch nanoseconds; what Neon `/api/event` expects in `timestamp`. */
        private fun utcNanosNow(): Long = System.currentTimeMillis() * 1_000_000L
    }

    /** Rolling Acc quality for LF/RF (updated on every parsed IMU line, even when not recording). */
    private val accQuality = AccQualityTracker(ACC_WINDOW)

    fun qtPyReady(): Boolean = qtPy.isReady()
    fun qtPyIsStreaming(): Boolean = qtPy.isStreaming()
    fun qtPyStateFlow(): StateFlow<QtPyBridgeState> = qtPy.state
    fun pcBridgeStateFlow(): StateFlow<PcBridgeState> = pcBridge.state

    suspend fun qtPyConnectResult(): String {
        val result = qtPy.connect()
        refreshFg()
        return result
    }

    suspend fun qtPyDisconnectResult(): String {
        val result = qtPy.disconnect()
        refreshFg()
        return result
    }
    /** After USB permission dialog / resume: open if already permitted. Null = nothing to do. */
    suspend fun qtPyTryConnectIfPermitted(): String? = qtPy.tryConnectIfPermitted()
    suspend fun qtPySendCommand(command: String): String = qtPy.writeCommand(command)
    fun qtPyClearMonitor() = qtPy.clearRecentLines()

    fun pcBridgeStart(): String {
        RecordingForegroundService.ensureStarted(appContext)
        val msg = pcBridge.start()
        RecordingForegroundService.refresh(appContext)
        return msg
    }

    fun pcBridgeStop(): String {
        val msg = pcBridge.stop()
        RecordingForegroundService.refresh(appContext)
        return msg
    }

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

    private fun refreshFg() {
        RecordingForegroundService.refresh(appContext)
    }
    /**
     * PC monitor snapshot (polled ~1 Hz). Fast local fields + short Neon Companion status probe.
     * Stall: STREAMING but no IMU CSV row for > [IMU_STALL_MS].
     */
    suspend fun buildPcHealthSnapshot(): String {
        val st = qtPy.state.value
        val fw = st.firmwareState.name
        val streaming = st.isStreaming
        val imuAge = qtPy.imuAgeMs()
        val stalled = streaming && (imuAge == null || imuAge > IMU_STALL_MS)
        val recording = combinedSessionActive.get()
        val neonId = activeRecordingId.get()
        val usbDevicePresent = qtPy.usbDevicePresent()
        val usbReconnectMayBeRequired = st.usbIssue in setOf(
            QtPyUsbIssue.DETACHED,
            QtPyUsbIssue.READ_ERROR,
            QtPyUsbIssue.WRITE_ERROR,
            QtPyUsbIssue.OPEN_FAILED,
            QtPyUsbIssue.STALLED,
        ) || stalled
        val usbDiagnosis = when {
            stalled -> "STREAM_STALLED_OR_BOARD_HUNG"
            st.usbIssue != QtPyUsbIssue.NONE -> st.usbIssue.name
            st.connected -> "OK"
            usbDevicePresent -> "DEVICE_PRESENT_NOT_CONNECTED"
            else -> "NO_DEVICE"
        }

        val neonProbe = when (val r = neon.neonStatus()) {
            is NeonHttpResult.Ok -> "neon_reachable=true\nneon_detail=${summarizeStatus(r.value)}"
            is NeonHttpResult.Err -> "neon_reachable=false\nneon_detail=${r.describe("GET /status")}"
        }

        val acc = accQuality.snapshot()
        val accOk = acc.samples > 20 && acc.lfOk && acc.rfOk
        val ready = streaming && !recording && !stalled && accOk

        return buildString {
            append("ok=1\n")
            append("qtpy_connected=").append(st.connected).append('\n')
            append("qtpy_firmware=").append(fw).append('\n')
            append("qtpy_streaming=").append(streaming).append('\n')
            append("usb_device_present=").append(usbDevicePresent).append('\n')
            append("usb_diagnosis=").append(usbDiagnosis).append('\n')
            append("usb_issue=").append(st.usbIssue.name).append('\n')
            append("usb_issue_detail=")
                .append(st.usbIssueDetail.replace('\n', ' ').replace('\r', ' '))
                .append('\n')
            append("usb_issue_wall_ms=")
                .append(st.usbIssueWallTimeMs?.toString() ?: "none")
                .append('\n')
            append("usb_physical_reconnect_may_be_required=")
                .append(usbReconnectMayBeRequired && !usbDevicePresent)
                .append('\n')
            append("usb_want_connected=").append(qtPy.wantsConnection()).append('\n')
            append("usb_auto_reconnect_attempt=")
                .append(qtPy.autoReconnectAttempt())
                .append('\n')
            append("imu_age_ms=").append(imuAge?.toString() ?: "none").append('\n')
            append("imu_stalled=").append(stalled).append('\n')
            append("imu_last_seq=").append(st.lastSeq?.toString() ?: "none").append('\n')
            append("imu_recording=").append(recording).append('\n')
            append("neon_recording_id=").append(neonId ?: "none").append('\n')
            append("neon_recording=").append(neonId != null).append('\n')
            append("recording_active=").append(recording).append('\n')
            append("acc_samples=").append(acc.samples).append('\n')
            append("acc_ok_lf=").append(acc.lfOk).append('\n')
            append("acc_ok_rf=").append(acc.rfOk).append('\n')
            append("acc_ok=").append(accOk).append('\n')
            append("acc_stuck_frac_lf=").append("%.3f".format(acc.lfStuckFrac)).append('\n')
            append("acc_stuck_frac_rf=").append("%.3f".format(acc.rfStuckFrac)).append('\n')
            append("acc_z_dead_frac_lf=").append("%.3f".format(acc.lfZDeadFrac)).append('\n')
            append("acc_z_dead_frac_rf=").append("%.3f".format(acc.rfZDeadFrac)).append('\n')
            append("acc_last_lf=").append(acc.lfLast).append('\n')
            append("acc_last_rf=").append(acc.rfLast).append('\n')
            append("ready_to_record=").append(ready).append('\n')
            append(neonProbe).append('\n')
            csvSessionInfo.get()?.let { append("csv=").append(it).append('\n') }
            append("bridge=").append(pcBridge.state.value.bindHint).append('\n')
            append("cmds=calibrate,start,stop,status,next,record_start,record_stop,watch\n")
        }
    }

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
            refreshFg()
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
        refreshFg()
        return buildString {
            append(result)
            csvSessionInfo.get()?.let { append("\n").append(it) }
        }
    }

    private fun onQtPyLineRecord(record: QtPyLineRecord) {
        val parsed = parseDualImuCsvLine(record.rawLine) ?: return
        // Always update Acc quality while streaming (even before record_start).
        accQuality.push(parsed.lf, parsed.rf)
        if (!combinedSessionActive.get()) return
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

/**
 * Rolling Acc quality for LF/RF. Detects the two failure modes that killed participant50:
 * frozen Acc ≈ (77.213, 77.227, *) and Acc_Z stuck near 0.
 */
internal class AccQualityTracker(
    private val window: Int,
    private val badFrac: Double = ImuRecorderIntegration.ACC_BAD_FRAC,
) {
    data class Snapshot(
        val samples: Int,
        val lfOk: Boolean,
        val rfOk: Boolean,
        val lfStuckFrac: Double,
        val rfStuckFrac: Double,
        val lfZDeadFrac: Double,
        val rfZDeadFrac: Double,
        val lfLast: String,
        val rfLast: String,
    )

    private val lock = Any()
    private val lfStuck = BooleanArray(window)
    private val rfStuck = BooleanArray(window)
    private val lfZDead = BooleanArray(window)
    private val rfZDead = BooleanArray(window)
    private var idx = 0
    private var filled = 0
    private var lfLastX = 0.0
    private var lfLastY = 0.0
    private var lfLastZ = 0.0
    private var rfLastX = 0.0
    private var rfLastY = 0.0
    private var rfLastZ = 0.0

    fun push(lf: DoubleArray, rf: DoubleArray) {
        val lax = lf.getOrElse(0) { 0.0 }
        val lay = lf.getOrElse(1) { 0.0 }
        val laz = lf.getOrElse(2) { 0.0 }
        val rax = rf.getOrElse(0) { 0.0 }
        val ray = rf.getOrElse(1) { 0.0 }
        val raz = rf.getOrElse(2) { 0.0 }
        synchronized(lock) {
            lfStuck[idx] = isStuckAcc(lax, lay)
            rfStuck[idx] = isStuckAcc(rax, ray)
            lfZDead[idx] = kotlin.math.abs(laz) < ImuRecorderIntegration.ACC_Z_DEAD_EPS
            rfZDead[idx] = kotlin.math.abs(raz) < ImuRecorderIntegration.ACC_Z_DEAD_EPS
            lfLastX = lax; lfLastY = lay; lfLastZ = laz
            rfLastX = rax; rfLastY = ray; rfLastZ = raz
            idx = (idx + 1) % window
            if (filled < window) filled++
        }
    }

    fun snapshot(): Snapshot = synchronized(lock) {
        val n = filled
        if (n == 0) {
            return Snapshot(
                samples = 0,
                lfOk = false,
                rfOk = false,
                lfStuckFrac = 0.0,
                rfStuckFrac = 0.0,
                lfZDeadFrac = 0.0,
                rfZDeadFrac = 0.0,
                lfLast = "none",
                rfLast = "none",
            )
        }
        var ls = 0; var rs = 0; var lz = 0; var rz = 0
        for (i in 0 until n) {
            if (lfStuck[i]) ls++
            if (rfStuck[i]) rs++
            if (lfZDead[i]) lz++
            if (rfZDead[i]) rz++
        }
        val lfStuckFrac = ls.toDouble() / n
        val rfStuckFrac = rs.toDouble() / n
        val lfZDeadFrac = lz.toDouble() / n
        val rfZDeadFrac = rz.toDouble() / n
        Snapshot(
            samples = n,
            lfOk = lfStuckFrac < badFrac && lfZDeadFrac < badFrac,
            rfOk = rfStuckFrac < badFrac && rfZDeadFrac < badFrac,
            lfStuckFrac = lfStuckFrac,
            rfStuckFrac = rfStuckFrac,
            lfZDeadFrac = lfZDeadFrac,
            rfZDeadFrac = rfZDeadFrac,
            lfLast = "%.2f,%.2f,%.2f".format(lfLastX, lfLastY, lfLastZ),
            rfLast = "%.2f,%.2f,%.2f".format(rfLastX, rfLastY, rfLastZ),
        )
    }

    private fun isStuckAcc(ax: Double, ay: Double): Boolean {
        return kotlin.math.abs(ax - ImuRecorderIntegration.STUCK_ACC_X) < ImuRecorderIntegration.STUCK_EPS &&
            kotlin.math.abs(ay - ImuRecorderIntegration.STUCK_ACC_Y) < ImuRecorderIntegration.STUCK_EPS
    }
}
