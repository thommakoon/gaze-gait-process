package com.example.urp2026.pcbridge

import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.Job
import kotlinx.coroutines.SupervisorJob
import kotlinx.coroutines.cancel
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.isActive
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import java.io.BufferedReader
import java.io.InputStreamReader
import java.io.OutputStreamWriter
import java.net.Inet4Address
import java.net.NetworkInterface
import java.net.ServerSocket
import java.net.Socket
import java.net.SocketException
import java.util.concurrent.atomic.AtomicBoolean

data class PcBridgeState(
    val running: Boolean = false,
    val port: Int = 8765,
    val bindHint: String = "",
    val lastRequest: String = "",
    val status: String = "Idle",
)

data class PcActionResult(
    val ok: Boolean,
    val message: String,
    /** Suggested HTTP status when !ok (200 when ok). */
    val httpStatus: Int = if (ok) 200 else 409,
)

/**
 * Tiny LAN HTTP server so a PC can drive QT Py + recording over Wi‑Fi:
 *   PC --HTTP--> phone (this) --USB CMD:*--> QT Py
 *                          \-- Neon + CSV (same gates as app "Start both")
 *
 * Endpoints (cleartext HTTP, study LAN only):
 *   GET  /health
 *   GET  /qtpy/calibrate|start|stop|status|next
 *   POST /qtpy/cmd   body: CALIBRATE | START | STOP | STATUS | NEXT
 *   GET|POST /record/start   same as app "Start both recording"
 *   GET|POST /record/stop    same as app "Stop both recording"
 */
class PcCommandHttpServer(
    private val port: Int = DEFAULT_PORT,
    private val onQtPyCommand: suspend (String) -> String,
    private val onHealthSnapshot: suspend () -> String,
    private val qtPyConnected: () -> Boolean,
    private val onRecordStart: suspend () -> PcActionResult,
    private val onRecordStop: suspend () -> PcActionResult,
) {
    private val scope = CoroutineScope(SupervisorJob() + Dispatchers.IO)
    private val started = AtomicBoolean(false)
    private var serverSocket: ServerSocket? = null
    private var acceptJob: Job? = null

    private val _state = MutableStateFlow(PcBridgeState(port = port))
    val state: StateFlow<PcBridgeState> = _state.asStateFlow()

    fun isRunning(): Boolean = started.get()

    fun start(): String {
        if (!started.compareAndSet(false, true)) {
            return "PC bridge already running on :$port"
        }
        return try {
            val ss = ServerSocket(port)
            serverSocket = ss
            val ips = localIpv4Addresses()
            val hint = if (ips.isEmpty()) {
                "http://<phone-ip>:$port"
            } else {
                ips.joinToString(" | ") { "http://$it:$port" }
            }
            _state.value = PcBridgeState(
                running = true,
                port = port,
                bindHint = hint,
                status = "Listening — $hint",
            )
            acceptJob = scope.launch { acceptLoop(ss) }
            "PC bridge listening on $hint"
        } catch (e: Exception) {
            started.set(false)
            serverSocket = null
            val msg = "PC bridge start failed: ${e.message ?: e.javaClass.simpleName}"
            _state.value = PcBridgeState(port = port, status = msg)
            msg
        }
    }

    fun stop(): String {
        if (!started.compareAndSet(true, false)) {
            return "PC bridge not running."
        }
        acceptJob?.cancel()
        acceptJob = null
        try {
            serverSocket?.close()
        } catch (_: Exception) {
        }
        serverSocket = null
        _state.value = PcBridgeState(port = port, status = "Stopped")
        return "PC bridge stopped."
    }

    fun close() {
        stop()
        scope.cancel()
    }

    private suspend fun acceptLoop(ss: ServerSocket) {
        while (scope.isActive && started.get()) {
            val socket = try {
                ss.accept()
            } catch (_: SocketException) {
                break
            } catch (_: Exception) {
                break
            }
            scope.launch { handleClient(socket) }
        }
    }

    private suspend fun handleClient(socket: Socket) = withContext(Dispatchers.IO) {
        try {
            // Record start may wait on Neon localhost HTTP; allow a bit longer.
            socket.soTimeout = 30_000
            val reader = BufferedReader(InputStreamReader(socket.getInputStream(), Charsets.UTF_8))
            val requestLine = reader.readLine() ?: return@withContext
            val parts = requestLine.split(" ")
            if (parts.size < 2) {
                writeHttp(socket, 400, "text/plain", "Bad request\n")
                return@withContext
            }
            val method = parts[0].uppercase()
            val path = parts[1].substringBefore('?')

            var contentLength = 0
            while (true) {
                val header = reader.readLine() ?: break
                if (header.isEmpty()) break
                if (header.startsWith("Content-Length:", ignoreCase = true)) {
                    contentLength = header.substringAfter(':').trim().toIntOrNull() ?: 0
                }
            }
            val body = if (contentLength > 0) {
                val buf = CharArray(contentLength.coerceAtMost(256))
                val n = reader.read(buf, 0, buf.size)
                if (n > 0) String(buf, 0, n).trim() else ""
            } else {
                ""
            }

            _state.value = _state.value.copy(lastRequest = "$method $path")

            when {
                path == "/health" || path == "/" -> {
                    val bodyOut = try {
                        onHealthSnapshot()
                    } catch (e: Exception) {
                        "ok=0\nerror=${e.message ?: e.javaClass.simpleName}\n"
                    }
                    writeHttp(socket, 200, "text/plain; charset=utf-8", bodyOut)
                }
                path.equals("/record/start", ignoreCase = true) -> {
                    val result = try {
                        onRecordStart()
                    } catch (e: Exception) {
                        PcActionResult(
                            ok = false,
                            message = "Error: ${e.message ?: e.javaClass.simpleName}",
                            httpStatus = 500,
                        )
                    }
                    writeAction(socket, result)
                }
                path.equals("/record/stop", ignoreCase = true) -> {
                    val result = try {
                        onRecordStop()
                    } catch (e: Exception) {
                        PcActionResult(
                            ok = false,
                            message = "Error: ${e.message ?: e.javaClass.simpleName}",
                            httpStatus = 500,
                        )
                    }
                    writeAction(socket, result)
                }
                path.startsWith("/qtpy/") -> {
                    val cmd = when {
                        path.equals("/qtpy/calibrate", ignoreCase = true) -> "CALIBRATE"
                        path.equals("/qtpy/start", ignoreCase = true) -> "START"
                        path.equals("/qtpy/stop", ignoreCase = true) -> "STOP"
                        path.equals("/qtpy/status", ignoreCase = true) -> "STATUS"
                        path.equals("/qtpy/next", ignoreCase = true) -> "NEXT"
                        path.equals("/qtpy/cmd", ignoreCase = true) &&
                            (method == "POST" || method == "PUT") -> {
                            body.removePrefix("CMD:").trim().ifEmpty { null }
                        }
                        else -> null
                    }
                    if (cmd == null) {
                        writeHttp(
                            socket,
                            404,
                            "text/plain; charset=utf-8",
                            "Unknown path. Use /qtpy/calibrate|start|stop|status|next or POST /qtpy/cmd\n",
                        )
                    } else if (!qtPyConnected()) {
                        writeHttp(
                            socket,
                            503,
                            "text/plain; charset=utf-8",
                            "QT Py not connected over USB on phone.\n",
                        )
                    } else {
                        val result = try {
                            onQtPyCommand(cmd)
                        } catch (e: Exception) {
                            "Error: ${e.message ?: e.javaClass.simpleName}"
                        }
                        writeHttp(socket, 200, "text/plain; charset=utf-8", result + "\n")
                    }
                }
                else -> writeHttp(socket, 404, "text/plain", "Not found\n")
            }
        } catch (e: Exception) {
            try {
                writeHttp(
                    socket,
                    500,
                    "text/plain",
                    "Server error: ${e.message ?: e.javaClass.simpleName}\n",
                )
            } catch (_: Exception) {
            }
        } finally {
            try {
                socket.close()
            } catch (_: Exception) {
            }
        }
    }

    private fun writeAction(socket: Socket, result: PcActionResult) {
        val code = if (result.ok) 200 else result.httpStatus
        val body = buildString {
            append("ok=").append(if (result.ok) "1" else "0").append('\n')
            append(result.message.trimEnd()).append('\n')
        }
        writeHttp(socket, code, "text/plain; charset=utf-8", body)
    }

    private fun writeHttp(socket: Socket, code: Int, contentType: String, body: String) {
        val bytes = body.toByteArray(Charsets.UTF_8)
        val reason = when (code) {
            200 -> "OK"
            400 -> "Bad Request"
            404 -> "Not Found"
            409 -> "Conflict"
            503 -> "Service Unavailable"
            else -> "Error"
        }
        val out = OutputStreamWriter(socket.getOutputStream(), Charsets.UTF_8)
        out.write("HTTP/1.1 $code $reason\r\n")
        out.write("Content-Type: $contentType\r\n")
        out.write("Content-Length: ${bytes.size}\r\n")
        out.write("Connection: close\r\n")
        out.write("Access-Control-Allow-Origin: *\r\n")
        out.write("\r\n")
        out.flush()
        socket.getOutputStream().write(bytes)
        socket.getOutputStream().flush()
    }

    companion object {
        const val DEFAULT_PORT = 8765

        fun localIpv4Addresses(): List<String> {
            val out = ArrayList<String>()
            try {
                val en = NetworkInterface.getNetworkInterfaces() ?: return out
                while (en.hasMoreElements()) {
                    val nif = en.nextElement()
                    if (!nif.isUp || nif.isLoopback) continue
                    val addrs = nif.inetAddresses
                    while (addrs.hasMoreElements()) {
                        val addr = addrs.nextElement()
                        if (addr is Inet4Address && !addr.isLoopbackAddress) {
                            out.add(addr.hostAddress ?: continue)
                        }
                    }
                }
            } catch (_: Exception) {
            }
            return out
        }
    }
}
