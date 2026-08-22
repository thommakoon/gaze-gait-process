package com.example.urp2026.qtpy

import android.app.PendingIntent
import android.content.BroadcastReceiver
import android.content.Context
import android.content.Intent
import android.content.IntentFilter
import android.hardware.usb.UsbDevice
import android.hardware.usb.UsbManager
import android.os.Build
import android.os.SystemClock
import androidx.core.content.ContextCompat
import com.hoho.android.usbserial.driver.UsbSerialDriver
import com.hoho.android.usbserial.driver.UsbSerialPort
import com.hoho.android.usbserial.driver.UsbSerialProber
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.Job
import kotlinx.coroutines.SupervisorJob
import kotlinx.coroutines.cancel
import kotlinx.coroutines.delay
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.launch
import kotlinx.coroutines.sync.Mutex
import kotlinx.coroutines.sync.withLock
import kotlinx.coroutines.withContext
import java.util.concurrent.atomic.AtomicBoolean
import java.util.concurrent.atomic.AtomicInteger
import kotlin.coroutines.cancellation.CancellationException
import kotlin.coroutines.resume
import kotlinx.coroutines.suspendCancellableCoroutine

data class QtPyLineRecord(
    val rawLine: String,
    val recvElapsedRealtimeNs: Long,
    val recvWallTimeNs: Long,
    val seq: Long? = null,
    val timeUs: Long? = null,
    val timeUsExtended: Long? = null,
)

/** Firmware state machine from sketch_usb_dual_* (parsed from STATE:* lines). */
enum class QtPyFirmwareState {
    UNKNOWN,
    WAITING,
    CALIBRATING,
    READY,
    STREAMING,
}

/** Last USB-level problem observed by the bridge. */
enum class QtPyUsbIssue {
    NONE,
    PERMISSION_DENIED,
    DEVICE_NOT_FOUND,
    OPEN_FAILED,
    READ_ERROR,
    WRITE_ERROR,
    DETACHED,
    /** Soft link died (OEM sleep / host reset) while device may still be enumerated. */
    STALLED,
}

data class QtPyBridgeState(
    val connected: Boolean = false,
    val reading: Boolean = false,
    val firmwareState: QtPyFirmwareState = QtPyFirmwareState.UNKNOWN,
    val usbIssue: QtPyUsbIssue = QtPyUsbIssue.NONE,
    val usbIssueDetail: String = "",
    val usbIssueWallTimeMs: Long? = null,
    val linesReceived: Long = 0,
    val lastLine: String = "",
    val recentLines: List<String> = emptyList(),
    val recentRecords: List<QtPyLineRecord> = emptyList(),
    val lastRecvElapsedRealtimeNs: Long? = null,
    val lastRecvWallTimeNs: Long? = null,
    /** Last recv time for a parsed IMU CSV row (not STATE:/CMD: lines). */
    val lastImuRecvElapsedRealtimeNs: Long? = null,
    val lastSeq: Long? = null,
    val lastTimeUs: Long? = null,
    val lastTimeUsExtended: Long? = null,
    val status: String = "Idle",
) {
    val isStreaming: Boolean get() = connected && firmwareState == QtPyFirmwareState.STREAMING
}

class QtPySerialBridge(
    private val context: Context,
    private val baudRate: Int = 230400,
) {
    private val appContext = context.applicationContext
    private val usbManager = appContext.getSystemService(Context.USB_SERVICE) as UsbManager
    private val ioScope = CoroutineScope(SupervisorJob() + Dispatchers.IO)
    private val connectMutex = Mutex()

    private var activeDriver: UsbSerialDriver? = null
    private var activePort: UsbSerialPort? = null
    private var readJob: Job? = null
    private val closed = AtomicBoolean(false)
    private val timeUsExtender = TimeUsExtender()
    private val permissionReceiverRegistered = AtomicBoolean(false)
    private val detachReceiverRegistered = AtomicBoolean(false)
    private val attachReceiverRegistered = AtomicBoolean(false)
    /** User wants a live USB session; enables soft auto-reconnect until Disconnect. */
    private val wantConnected = AtomicBoolean(false)
    private val reconnectAttempt = AtomicInteger(0)
    private var reconnectJob: Job? = null
    private var stallWatchJob: Job? = null
    @Volatile
    private var pendingPermissionDeviceId: Int? = null
    @Volatile
    private var lineRecordListener: ((QtPyLineRecord) -> Unit)? = null

    private val _state = MutableStateFlow(QtPyBridgeState())
    val state: StateFlow<QtPyBridgeState> = _state.asStateFlow()

    fun wantsConnection(): Boolean = wantConnected.get()
    fun autoReconnectAttempt(): Int = reconnectAttempt.get()

    private val permissionReceiver = object : BroadcastReceiver() {
        override fun onReceive(context: Context, intent: Intent) {
            if (intent.action != ACTION_USB_PERMISSION) return
            val device: UsbDevice? = if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.TIRAMISU) {
                intent.getParcelableExtra(UsbManager.EXTRA_DEVICE, UsbDevice::class.java)
            } else {
                @Suppress("DEPRECATION")
                intent.getParcelableExtra(UsbManager.EXTRA_DEVICE)
            }
            val granted = intent.getBooleanExtra(UsbManager.EXTRA_PERMISSION_GRANTED, false)
            pendingPermissionDeviceId = null
            val permitted = granted ||
                (device != null && usbManager.hasPermission(device)) ||
                (findDriver()?.let { usbManager.hasPermission(it.device) } == true)

            if (!permitted) {
                setUsbFailure(
                    QtPyUsbIssue.PERMISSION_DENIED,
                    "USB permission denied${device?.let { " for ${it.deviceName}" } ?: ""}.",
                )
                return
            }

            // Permission dialog often pauses/cancels the UI connect() coroutine.
            // Always finish open on the bridge scope so one "Yes" is enough.
            ioScope.launch {
                val msg = connectMutex.withLock {
                    if (_state.value.connected) return@withLock "QT Py already connected."
                    val driver = findDriver()
                        ?: return@withLock "No USB serial device found after permission grant."
                    if (!usbManager.hasPermission(driver.device)) {
                        return@withLock "USB permission not active yet; tap Connect again."
                    }
                    openPortLocked(driver)
                }
                if (!_state.value.connected) {
                    setUsbFailure(QtPyUsbIssue.OPEN_FAILED, msg)
                }
            }
        }
    }

    private val detachReceiver = object : BroadcastReceiver() {
        override fun onReceive(context: Context, intent: Intent) {
            if (intent.action != UsbManager.ACTION_USB_DEVICE_DETACHED) return
            val detached = usbDeviceFrom(intent)
            val activeDeviceId = activeDriver?.device?.deviceId
            if (activeDeviceId == null ||
                (detached != null && detached.deviceId != activeDeviceId)
            ) {
                return
            }
            ioScope.launch {
                connectMutex.withLock {
                    cleanupPortLocked()
                    setUsbFailure(
                        QtPyUsbIssue.DETACHED,
                        "QT Py detached. Waiting for USB re-attach (auto-reconnect if permitted)…",
                    )
                }
                // Device is gone; loop will wait until ATTACHED / findDriver succeeds.
                scheduleSoftReconnect("detached")
            }
        }
    }

    private val attachReceiver = object : BroadcastReceiver() {
        override fun onReceive(context: Context, intent: Intent) {
            if (intent.action != UsbManager.ACTION_USB_DEVICE_ATTACHED) return
            if (!wantConnected.get() || _state.value.connected) return
            scheduleSoftReconnect("attached")
        }
    }

    init {
        // Listen even when permission was granted previously, so physical removal is diagnosed.
        ensurePermissionReceiver()
        ensureDetachReceiver()
        ensureAttachReceiver()
    }

    fun isReady(): Boolean = _state.value.connected
    fun isStreaming(): Boolean = _state.value.isStreaming
    fun firmwareState(): QtPyFirmwareState = _state.value.firmwareState
    fun usbDevicePresent(): Boolean = findDriver() != null

    /** Milliseconds since last IMU CSV row; null if none yet. */
    fun imuAgeMs(): Long? {
        val last = _state.value.lastImuRecvElapsedRealtimeNs ?: return null
        return ((SystemClock.elapsedRealtimeNanos() - last) / 1_000_000L).coerceAtLeast(0L)
    }

    fun setLineRecordListener(listener: ((QtPyLineRecord) -> Unit)?) {
        lineRecordListener = listener
    }

    fun clearRecentLines() {
        _state.value = _state.value.copy(
            recentLines = emptyList(),
            recentRecords = emptyList(),
            lastLine = "",
            status = if (_state.value.connected) "Reading lines…" else _state.value.status,
        )
    }

    /**
     * Open serial if a device is present and already permitted.
     * Used after resume when the permission dialog interrupted the first connect().
     * @return status string if an attempt was made; null if nothing to do.
     */
    suspend fun tryConnectIfPermitted(): String? = withContext(Dispatchers.IO) {
        connectMutex.withLock {
            if (_state.value.connected) return@withContext null
            val driver = findDriver() ?: return@withContext null
            if (!usbManager.hasPermission(driver.device)) return@withContext null
            openPortLocked(driver)
        }
    }

    suspend fun connect(): String = withContext(Dispatchers.IO) {
        wantConnected.set(true)
        if (_state.value.connected) return@withContext "QT Py already connected."

        val driver = findDriver()
            ?: return@withContext setUsbFailure(
                QtPyUsbIssue.DEVICE_NOT_FOUND,
                "No USB serial device found. Plug QT Py via OTG.",
            )

        // Do not hold connectMutex while the system permission dialog is up — that
        // would deadlock the standing permissionReceiver that finishes open on grant.
        if (!usbManager.hasPermission(driver.device)) {
            _state.value = _state.value.copy(status = "Waiting for USB permission…")
            val granted = requestUsbPermission(driver.device)
            if (!usbManager.hasPermission(driver.device)) {
                val message = if (granted) {
                    "USB permission granted — opening…"
                } else {
                    "USB permission denied for ${driver.device.deviceName}."
                }
                return@withContext setUsbFailure(
                    if (granted) QtPyUsbIssue.OPEN_FAILED else QtPyUsbIssue.PERMISSION_DENIED,
                    message,
                )
            }
            _state.value = _state.value.copy(status = "Permission granted — opening port…")
        }

        connectMutex.withLock {
            if (_state.value.connected) return@withContext "QT Py already connected."
            val again = findDriver()
                ?: return@withContext setUsbFailure(
                    QtPyUsbIssue.DEVICE_NOT_FOUND,
                    "No USB serial device found. Plug QT Py via OTG.",
                )
            if (!usbManager.hasPermission(again.device)) {
                return@withContext setUsbFailure(
                    QtPyUsbIssue.PERMISSION_DENIED,
                    "USB permission not active; tap Connect again.",
                )
            }
            openPortLocked(again)
        }
    }

    suspend fun disconnect(): String = withContext(Dispatchers.IO) {
        wantConnected.set(false)
        cancelSoftReconnect()
        cancelStallWatch()
        connectMutex.withLock {
            cleanupPortLocked()
            _state.value = _state.value.copy(
                connected = false,
                reading = false,
                firmwareState = QtPyFirmwareState.UNKNOWN,
                usbIssue = QtPyUsbIssue.NONE,
                usbIssueDetail = "",
                usbIssueWallTimeMs = null,
                status = "Disconnected",
            )
            "QT Py disconnected."
        }
    }

    /**
     * Send a newline-framed command to QT Py firmware (sketch_usb_dual_100_cmd).
     * Accepts `CALIBRATE` or `CMD:CALIBRATE` (prefix added if missing).
     */
    suspend fun writeCommand(command: String): String = withContext(Dispatchers.IO) {
        val raw = command.trim()
        if (raw.isEmpty()) return@withContext "Empty command."
        val line = if (raw.startsWith("CMD:", ignoreCase = true)) {
            raw.uppercase()
        } else {
            "CMD:${raw.uppercase()}"
        }
        connectMutex.withLock {
            val port = activePort
                ?: return@withLock "QT Py not connected over USB."
            try {
                val bytes = (line + "\n").toByteArray(Charsets.UTF_8)
                port.write(bytes, WRITE_TIMEOUT_MS)
                _state.value = _state.value.copy(status = "Sent $line")
                "Sent $line"
            } catch (e: Exception) {
                cleanupPortLocked()
                val detail = setUsbFailure(
                    QtPyUsbIssue.WRITE_ERROR,
                    "USB write failed: ${e.message ?: e.javaClass.simpleName}",
                )
                scheduleSoftReconnect("write_error")
                detail
            }
        }
    }

    fun close() {
        if (!closed.compareAndSet(false, true)) return
        wantConnected.set(false)
        cancelSoftReconnect()
        cancelStallWatch()
        unregisterPermissionReceiver()
        unregisterDetachReceiver()
        unregisterAttachReceiver()
        ioScope.launch { disconnect() }
        ioScope.cancel()
    }

    private fun findDriver(): UsbSerialDriver? =
        UsbSerialProber.getDefaultProber().findAllDrivers(usbManager).firstOrNull()

    /** Close port/reader without clearing wantConnected. Caller must hold [connectMutex]. */
    private fun cleanupPortLocked(cancelStall: Boolean = true) {
        stopReaderInternal()
        if (cancelStall) cancelStallWatch()
        val port = activePort
        activePort = null
        activeDriver = null
        try {
            port?.close()
        } catch (_: Exception) {
        }
    }

    /**
     * Soft reopen when the user still wants a session (Connect tapped, not Disconnect).
     * Handles OEM USB host drops, READ/WRITE errors, and re-attach after DETACHED.
     * Does not show the permission dialog — only opens if already permitted.
     */
    private fun scheduleSoftReconnect(reason: String) {
        if (!wantConnected.get() || closed.get()) return
        if (reconnectJob?.isActive == true) return
        reconnectJob = ioScope.launch {
            var delayMs = RECONNECT_INITIAL_DELAY_MS
            while (wantConnected.get() && !closed.get() && !_state.value.connected) {
                val n = reconnectAttempt.incrementAndGet()
                _state.value = _state.value.copy(
                    status = "USB soft-reconnect #$n ($reason)…",
                )
                val outcome = connectMutex.withLock {
                    if (_state.value.connected) return@withLock "ok"
                    cleanupPortLocked()
                    val driver = findDriver()
                        ?: return@withLock "waiting_device"
                    if (!usbManager.hasPermission(driver.device)) {
                        return@withLock "need_permission"
                    }
                    val msg = openPortLocked(driver)
                    if (_state.value.connected) "ok" else msg
                }
                if (outcome == "ok" && _state.value.connected) {
                    _state.value = _state.value.copy(
                        status = "USB soft-reconnected after $reason (attempt $n)",
                    )
                    reconnectAttempt.set(0)
                    return@launch
                }
                if (outcome == "need_permission") {
                    setUsbFailure(
                        QtPyUsbIssue.PERMISSION_DENIED,
                        "USB auto-reconnect needs permission — tap Connect once.",
                    )
                    return@launch
                }
                delay(delayMs)
                delayMs = (delayMs * 2).coerceAtMost(RECONNECT_MAX_DELAY_MS)
            }
        }
    }

    private fun cancelSoftReconnect() {
        reconnectJob?.cancel()
        reconnectJob = null
        reconnectAttempt.set(0)
    }

    private fun startStallWatch() {
        cancelStallWatch()
        stallWatchJob = ioScope.launch {
            while (wantConnected.get() && !closed.get() && activePort != null) {
                delay(STALL_POLL_MS)
                if (!_state.value.isStreaming) continue
                val age = imuAgeMs() ?: continue
                if (age < STALL_RECONNECT_MS) continue
                connectMutex.withLock {
                    if (!_state.value.isStreaming || activePort == null) return@withLock
                    // Don't cancel this stall job from inside itself.
                    cleanupPortLocked(cancelStall = false)
                    setUsbFailure(
                        QtPyUsbIssue.STALLED,
                        "IMU stalled ${age}ms while STREAMING — soft-reconnecting USB…",
                    )
                }
                scheduleSoftReconnect("imu_stall")
                return@launch
            }
        }
    }

    private fun cancelStallWatch() {
        stallWatchJob?.cancel()
        stallWatchJob = null
    }

    private fun openPortLocked(driver: UsbSerialDriver): String {
        if (_state.value.connected) return "QT Py already connected."

        val connection = usbManager.openDevice(driver.device)
            ?: return setUsbFailure(
                QtPyUsbIssue.OPEN_FAILED,
                "openDevice failed (permission, busy USB port, or stale Android USB handle).",
            )

        val port = driver.ports.firstOrNull()
            ?: return setUsbFailure(
                QtPyUsbIssue.OPEN_FAILED,
                "No serial port exposed by USB driver.",
            )

        return try {
            port.open(connection)
            port.setParameters(baudRate, 8, UsbSerialPort.STOPBITS_1, UsbSerialPort.PARITY_NONE)
            port.dtr = true
            port.rts = true
            activeDriver = driver
            activePort = port
            _state.value = _state.value.copy(
                connected = true,
                firmwareState = QtPyFirmwareState.UNKNOWN,
                usbIssue = QtPyUsbIssue.NONE,
                usbIssueDetail = "",
                usbIssueWallTimeMs = null,
                status = "Connected @ ${baudRate}bps to ${driver.device.deviceName}",
            )
            startReader()
            startStallWatch()
            reconnectAttempt.set(0)
            // One-shot: ask firmware to reprint STATE:* (sketch_usb_dual_100_cmd).
            // Does not run on the IMU hot path; ignored by older sketches without CMD:*.
            ioScope.launch {
                writeCommand("STATUS")
            }
            "QT Py connected."
        } catch (e: Exception) {
            try {
                port.close()
            } catch (_: Exception) {
            }
            activePort = null
            activeDriver = null
            setUsbFailure(
                QtPyUsbIssue.OPEN_FAILED,
                "Connect failed: ${e.message ?: e.javaClass.simpleName}",
            )
        }
    }

    private fun startReader() {
        stopReaderInternal()
        val port = activePort ?: return
        _state.value = _state.value.copy(reading = true, status = "Reading lines…")
        timeUsExtender.reset()
        readJob = ioScope.launch {
            val buffer = ByteArray(1024)
            val text = StringBuilder()
            while (activePort === port) {
                try {
                    val len = port.read(buffer, 500)
                    if (len <= 0) continue
                    text.append(String(buffer, 0, len))
                    var newlineIdx = text.indexOf("\n")
                    while (newlineIdx >= 0) {
                        val raw = text.substring(0, newlineIdx).trim('\r')
                        text.delete(0, newlineIdx + 1)
                        if (raw.isNotEmpty()) {
                            val recvElapsedNs = SystemClock.elapsedRealtimeNanos()
                            val recvWallNs = System.currentTimeMillis() * 1_000_000L
                            val fwFromLine = parseFirmwareStateLine(raw)
                            val parsed = parseSeqTimeUs(raw)
                            val extUs = parsed?.timeUs?.let { timeUsExtender.extend(it) }
                            val record = QtPyLineRecord(
                                rawLine = raw,
                                recvElapsedRealtimeNs = recvElapsedNs,
                                recvWallTimeNs = recvWallNs,
                                seq = parsed?.seq,
                                timeUs = parsed?.timeUs,
                                timeUsExtended = extUs,
                            )
                            val updatedRecent = (_state.value.recentLines + raw).takeLast(200)
                            val updatedRecords = (_state.value.recentRecords + record).takeLast(400)
                            val nextFw = fwFromLine ?: _state.value.firmwareState
                            _state.value = _state.value.copy(
                                linesReceived = _state.value.linesReceived + 1,
                                lastLine = raw,
                                recentLines = updatedRecent,
                                recentRecords = updatedRecords,
                                lastRecvElapsedRealtimeNs = recvElapsedNs,
                                lastRecvWallTimeNs = recvWallNs,
                                lastImuRecvElapsedRealtimeNs = if (parsed != null) {
                                    recvElapsedNs
                                } else {
                                    _state.value.lastImuRecvElapsedRealtimeNs
                                },
                                lastSeq = parsed?.seq ?: _state.value.lastSeq,
                                lastTimeUs = parsed?.timeUs ?: _state.value.lastTimeUs,
                                lastTimeUsExtended = extUs ?: _state.value.lastTimeUsExtended,
                                firmwareState = nextFw,
                                status = if (fwFromLine != null) {
                                    "Firmware: ${fwFromLine.name}"
                                } else {
                                    "Reading lines…"
                                },
                            )
                            lineRecordListener?.invoke(record)
                        }
                        newlineIdx = text.indexOf("\n")
                    }
                } catch (e: Exception) {
                    ioScope.launch {
                        connectMutex.withLock {
                            if (activePort !== port && activePort != null) return@withLock
                            cleanupPortLocked()
                            setUsbFailure(
                                QtPyUsbIssue.READ_ERROR,
                                "USB read failed: ${e.message ?: e.javaClass.simpleName}",
                            )
                        }
                        scheduleSoftReconnect("read_error")
                    }
                    break
                }
            }
        }
    }

    private fun stopReaderInternal() {
        readJob?.cancel()
        readJob = null
        if (_state.value.connected) {
            _state.value = _state.value.copy(reading = false)
        }
    }

    private fun ensurePermissionReceiver() {
        if (!permissionReceiverRegistered.compareAndSet(false, true)) return
        ContextCompat.registerReceiver(
            appContext,
            permissionReceiver,
            IntentFilter(ACTION_USB_PERMISSION),
            ContextCompat.RECEIVER_NOT_EXPORTED,
        )
    }

    private fun ensureDetachReceiver() {
        if (!detachReceiverRegistered.compareAndSet(false, true)) return
        ContextCompat.registerReceiver(
            appContext,
            detachReceiver,
            IntentFilter(UsbManager.ACTION_USB_DEVICE_DETACHED),
            ContextCompat.RECEIVER_EXPORTED,
        )
    }

    private fun ensureAttachReceiver() {
        if (!attachReceiverRegistered.compareAndSet(false, true)) return
        ContextCompat.registerReceiver(
            appContext,
            attachReceiver,
            IntentFilter(UsbManager.ACTION_USB_DEVICE_ATTACHED),
            ContextCompat.RECEIVER_EXPORTED,
        )
    }

    private fun setUsbFailure(issue: QtPyUsbIssue, detail: String): String {
        _state.value = _state.value.copy(
            connected = false,
            reading = false,
            firmwareState = QtPyFirmwareState.UNKNOWN,
            usbIssue = issue,
            usbIssueDetail = detail,
            usbIssueWallTimeMs = System.currentTimeMillis(),
            status = detail,
        )
        return detail
    }

    private fun usbDeviceFrom(intent: Intent): UsbDevice? {
        return if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.TIRAMISU) {
            intent.getParcelableExtra(UsbManager.EXTRA_DEVICE, UsbDevice::class.java)
        } else {
            @Suppress("DEPRECATION")
            intent.getParcelableExtra(UsbManager.EXTRA_DEVICE)
        }
    }

    private fun unregisterPermissionReceiver() {
        if (!permissionReceiverRegistered.compareAndSet(true, false)) return
        try {
            appContext.unregisterReceiver(permissionReceiver)
        } catch (_: Exception) {
        }
    }

    private fun unregisterDetachReceiver() {
        if (!detachReceiverRegistered.compareAndSet(true, false)) return
        try {
            appContext.unregisterReceiver(detachReceiver)
        } catch (_: Exception) {
        }
    }

    private fun unregisterAttachReceiver() {
        if (!attachReceiverRegistered.compareAndSet(true, false)) return
        try {
            appContext.unregisterReceiver(attachReceiver)
        } catch (_: Exception) {
        }
    }

    /**
     * Asks for USB permission. Uses a mutable PendingIntent (required so UsbManager can
     * attach grant/deny extras on Android 12+). The standing [permissionReceiver] also
     * opens the port on grant, so a cancelled UI coroutine still finishes connect.
     */
    private suspend fun requestUsbPermission(device: UsbDevice): Boolean {
        ensurePermissionReceiver()
        pendingPermissionDeviceId = device.deviceId

        val piFlags = PendingIntent.FLAG_UPDATE_CURRENT or
            if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.S) {
                PendingIntent.FLAG_MUTABLE
            } else {
                0
            }
        val permissionIntent = PendingIntent.getBroadcast(
            appContext,
            device.deviceId,
            Intent(ACTION_USB_PERMISSION).setPackage(appContext.packageName),
            piFlags,
        )

        return try {
            withContext(Dispatchers.Main) {
                suspendCancellableCoroutine { cont ->
                    // One-shot waiter for this connect() call. Do NOT unregister the
                    // standing permissionReceiver on cancel — that receiver finishes open.
                    val waiter = object : BroadcastReceiver() {
                        override fun onReceive(context: Context, intent: Intent) {
                            if (intent.action != ACTION_USB_PERMISSION) return
                            try {
                                appContext.unregisterReceiver(this)
                            } catch (_: Exception) {
                            }
                            val granted = intent.getBooleanExtra(
                                UsbManager.EXTRA_PERMISSION_GRANTED,
                                false,
                            ) || usbManager.hasPermission(device)
                            if (cont.isActive) cont.resume(granted)
                        }
                    }
                    ContextCompat.registerReceiver(
                        appContext,
                        waiter,
                        IntentFilter(ACTION_USB_PERMISSION),
                        ContextCompat.RECEIVER_NOT_EXPORTED,
                    )
                    usbManager.requestPermission(device, permissionIntent)
                    cont.invokeOnCancellation {
                        try {
                            appContext.unregisterReceiver(waiter)
                        } catch (_: Exception) {
                        }
                        // Standing permissionReceiver remains registered.
                    }
                }
            }
        } catch (_: CancellationException) {
            // UI scope cancelled while dialog was up; standing receiver may still open.
            usbManager.hasPermission(device)
        }
    }

    companion object {
        private const val ACTION_USB_PERMISSION = "com.example.urp2026.USB_PERMISSION"
        private const val WRITE_TIMEOUT_MS = 1000
        private const val RECONNECT_INITIAL_DELAY_MS = 400L
        private const val RECONNECT_MAX_DELAY_MS = 8_000L
        /** Soft-reconnect if STREAMING but no IMU CSV for this long. */
        private const val STALL_RECONNECT_MS = 2_000L
        private const val STALL_POLL_MS = 1_000L
    }
}

private fun parseFirmwareStateLine(line: String): QtPyFirmwareState? {
    if (!line.startsWith("STATE:")) return null
    return when (line.substringAfter("STATE:").trim().uppercase()) {
        "WAITING" -> QtPyFirmwareState.WAITING
        "CALIBRATING" -> QtPyFirmwareState.CALIBRATING
        "READY" -> QtPyFirmwareState.READY
        "STREAMING" -> QtPyFirmwareState.STREAMING
        // CAL_DONE is immediately followed by READY; ignore intermediate.
        else -> null
    }
}

private data class ParsedSeqTimeUs(
    val seq: Long,
    val timeUs: Long,
)

private fun parseSeqTimeUs(line: String): ParsedSeqTimeUs? {
    val parts = line.split(',')
    if (parts.size < 2) return null
    val seq = parts[0].toLongOrNull() ?: return null
    val timeUs = parts[1].toLongOrNull() ?: return null
    return ParsedSeqTimeUs(seq, timeUs)
}

private class TimeUsExtender {
    private var wraps = 0L
    private var lastTimeUs: Long? = null

    fun reset() {
        wraps = 0L
        lastTimeUs = null
    }

    fun extend(timeUs: Long): Long {
        val prev = lastTimeUs
        if (prev != null) {
            // Detect micros() wrap (uint32): large backward jump means overflow.
            if (timeUs < prev && (prev - timeUs) > 1_000_000_000L) {
                wraps += 1L
            }
        }
        lastTimeUs = timeUs
        return timeUs + wraps * 4_294_967_296L
    }
}
