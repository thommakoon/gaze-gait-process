package com.example.urp2026.qtpy

import android.app.PendingIntent
import android.content.BroadcastReceiver
import android.content.Context
import android.content.Intent
import android.content.IntentFilter
import android.hardware.usb.UsbDevice
import android.hardware.usb.UsbManager
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
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import kotlin.coroutines.resume
import java.util.concurrent.atomic.AtomicBoolean

data class QtPyLineRecord(
    val rawLine: String,
    val recvElapsedRealtimeNs: Long,
    val recvWallTimeNs: Long,
    val seq: Long? = null,
    val timeUs: Long? = null,
    val timeUsExtended: Long? = null,
)

data class QtPyBridgeState(
    val connected: Boolean = false,
    val reading: Boolean = false,
    val linesReceived: Long = 0,
    val lastLine: String = "",
    val recentLines: List<String> = emptyList(),
    val recentRecords: List<QtPyLineRecord> = emptyList(),
    val lastRecvElapsedRealtimeNs: Long? = null,
    val lastRecvWallTimeNs: Long? = null,
    val lastSeq: Long? = null,
    val lastTimeUs: Long? = null,
    val lastTimeUsExtended: Long? = null,
    val status: String = "Idle",
)

class QtPySerialBridge(
    private val context: Context,
    private val baudRate: Int = 230400,
) {
    private val appContext = context.applicationContext
    private val usbManager = appContext.getSystemService(Context.USB_SERVICE) as UsbManager
    private val ioScope = CoroutineScope(SupervisorJob() + Dispatchers.IO)

    private var activeDriver: UsbSerialDriver? = null
    private var activePort: UsbSerialPort? = null
    private var readJob: Job? = null
    private val closed = AtomicBoolean(false)
    private val timeUsExtender = TimeUsExtender()
    @Volatile
    private var lineRecordListener: ((QtPyLineRecord) -> Unit)? = null

    private val _state = MutableStateFlow(QtPyBridgeState())
    val state: StateFlow<QtPyBridgeState> = _state.asStateFlow()

    fun isReady(): Boolean = _state.value.connected

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

    suspend fun connect(): String = withContext(Dispatchers.IO) {
        if (_state.value.connected) return@withContext "QT Py already connected."

        val driver = UsbSerialProber.getDefaultProber().findAllDrivers(usbManager).firstOrNull()
            ?: return@withContext "No USB serial device found. Plug QT Py via OTG."

        if (!usbManager.hasPermission(driver.device)) {
            val granted = requestUsbPermission(driver.device)
            if (!granted) {
                return@withContext "USB permission denied for ${driver.device.deviceName}."
            }
        }

        val connection = usbManager.openDevice(driver.device)
            ?: return@withContext "openDevice failed (permission or busy USB port)."

        val port = driver.ports.firstOrNull()
            ?: return@withContext "No serial port exposed by USB driver."

        try {
            port.open(connection)
            port.setParameters(baudRate, 8, UsbSerialPort.STOPBITS_1, UsbSerialPort.PARITY_NONE)
            port.dtr = true
            port.rts = true
            activeDriver = driver
            activePort = port
            _state.value = _state.value.copy(
                connected = true,
                status = "Connected @ ${baudRate}bps to ${driver.device.deviceName}",
            )
            startReader()
            "QT Py connected."
        } catch (e: Exception) {
            try {
                port.close()
            } catch (_: Exception) {
            }
            "Connect failed: ${e.message ?: e.javaClass.simpleName}"
        }
    }

    suspend fun disconnect(): String = withContext(Dispatchers.IO) {
        stopReaderInternal()
        val port = activePort
        activePort = null
        activeDriver = null
        try {
            port?.close()
        } catch (_: Exception) {
        }
        _state.value = _state.value.copy(
            connected = false,
            reading = false,
            status = "Disconnected",
        )
        "QT Py disconnected."
    }

    fun close() {
        if (!closed.compareAndSet(false, true)) return
        ioScope.launch { disconnect() }
        ioScope.cancel()
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
                            _state.value = _state.value.copy(
                                linesReceived = _state.value.linesReceived + 1,
                                lastLine = raw,
                                recentLines = updatedRecent,
                                recentRecords = updatedRecords,
                                lastRecvElapsedRealtimeNs = recvElapsedNs,
                                lastRecvWallTimeNs = recvWallNs,
                                lastSeq = parsed?.seq ?: _state.value.lastSeq,
                                lastTimeUs = parsed?.timeUs ?: _state.value.lastTimeUs,
                                lastTimeUsExtended = extUs ?: _state.value.lastTimeUsExtended,
                                status = "Reading lines…",
                            )
                            lineRecordListener?.invoke(record)
                        }
                        newlineIdx = text.indexOf("\n")
                    }
                } catch (e: Exception) {
                    _state.value = _state.value.copy(
                        reading = false,
                        connected = false,
                        status = "Read error: ${e.message ?: e.javaClass.simpleName}",
                    )
                    activePort = null
                    activeDriver = null
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

    private suspend fun requestUsbPermission(device: UsbDevice): Boolean =
        withContext(Dispatchers.Main) {
            val action = "com.example.urp2026.USB_PERMISSION"
            val permissionIntent = PendingIntent.getBroadcast(
                appContext,
                0,
                Intent(action),
                PendingIntent.FLAG_UPDATE_CURRENT or PendingIntent.FLAG_IMMUTABLE,
            )
            kotlinx.coroutines.suspendCancellableCoroutine { cont ->
                val receiver = object : BroadcastReceiver() {
                    override fun onReceive(context: Context, intent: Intent) {
                        if (intent.action != action) return
                        appContext.unregisterReceiver(this)
                        val granted = intent.getBooleanExtra(UsbManager.EXTRA_PERMISSION_GRANTED, false)
                        if (cont.isActive) cont.resume(granted) {}
                    }
                }
                ContextCompat.registerReceiver(
                    appContext,
                    receiver,
                    IntentFilter(action),
                    ContextCompat.RECEIVER_NOT_EXPORTED,
                )
                usbManager.requestPermission(device, permissionIntent)
                cont.invokeOnCancellation {
                    try {
                        appContext.unregisterReceiver(receiver)
                    } catch (_: Exception) {
                    }
                }
            }
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
