package com.example.urp2026

import android.os.Bundle
import android.widget.Toast
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.activity.enableEdgeToEdge
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.verticalScroll
import androidx.compose.material3.Button
import androidx.compose.material3.Card
import androidx.compose.material3.CardDefaults
import androidx.compose.material3.HorizontalDivider
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Scaffold
import androidx.compose.material3.Switch
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.DisposableEffect
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.rememberCoroutineScope
import androidx.compose.runtime.setValue
import androidx.compose.runtime.collectAsState
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.res.stringResource
import androidx.compose.ui.tooling.preview.Preview
import androidx.compose.ui.unit.dp
import androidx.lifecycle.Lifecycle
import androidx.lifecycle.LifecycleEventObserver
import com.example.urp2026.integration.ImuRecorderIntegration
import com.example.urp2026.qtpy.QtPyLinkLatencyDiagnostics
import com.example.urp2026.ui.theme.URP2026Theme
import kotlinx.coroutines.launch

private const val LOG_MAX_LINES = 48

class MainActivity : ComponentActivity() {
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        enableEdgeToEdge()
        setContent {
            URP2026Theme {
                val integration = remember { ImuRecorderIntegration(applicationContext) }
                Scaffold(modifier = Modifier.fillMaxSize()) { innerPadding ->
                    NeonProbeScreen(
                        integration = integration,
                        modifier = Modifier.padding(innerPadding),
                    )
                }
            }
        }
    }
}

@Composable
fun NeonProbeScreen(
    integration: ImuRecorderIntegration,
    modifier: Modifier = Modifier,
) {
    val context = LocalContext.current
    DisposableEffect(Unit) {
        onDispose { integration.close() }
    }
    val qtPyState by integration.qtPyStateFlow().collectAsState()
    val pcBridgeState by integration.pcBridgeStateFlow().collectAsState()
    val monitorScroll = rememberScrollState()
    var log by remember { mutableStateOf("") }
    var lastResult by remember { mutableStateOf("") }
    var busy by remember { mutableStateOf(false) }
    var recordingIdDisplay by remember { mutableStateOf<String?>(integration.neonActiveRecordingId()) }
    var combinedActive by remember { mutableStateOf(integration.isCombinedSessionActive()) }
    var csvSessionInfo by remember { mutableStateOf<String?>(integration.csvSessionInfo()) }
    val scope = rememberCoroutineScope()
    val scroll = rememberScrollState()
    var verboseSerialMonitor by remember { mutableStateOf(true) }
    var verboseTimingDetails by remember { mutableStateOf(true) }

    fun refreshRecordingLabel() {
        recordingIdDisplay = integration.neonActiveRecordingId()
        combinedActive = integration.isCombinedSessionActive()
        csvSessionInfo = integration.csvSessionInfo()
    }

    fun append(line: String) {
        val merged = "$line\n$log"
        log = merged.lines().take(LOG_MAX_LINES).joinToString("\n")
        refreshRecordingLabel()
    }

    fun runNeon(label: String, block: suspend () -> String) {
        if (busy) return
        busy = true
        scope.launch {
            val result = try {
                block()
            } catch (e: Exception) {
                "${e.javaClass.simpleName}: ${e.message}"
            }
            val line = "$label → $result"
            lastResult = line
            append(line)
            val toastText = result.lines().firstOrNull().orEmpty()
                .let { if (it.length <= 180) it else it.take(177) + "…" }
            Toast.makeText(context, toastText, Toast.LENGTH_LONG).show()
            busy = false
        }
    }

    // After USB permission dialog, finish connect if Android already granted access.
    val activity = context as? ComponentActivity
    DisposableEffect(activity, integration) {
        if (activity == null) return@DisposableEffect onDispose { }
        val observer = LifecycleEventObserver { _, event ->
            if (event != Lifecycle.Event.ON_RESUME) return@LifecycleEventObserver
            if (busy || integration.qtPyReady()) return@LifecycleEventObserver
            scope.launch {
                val result = try {
                    integration.qtPyTryConnectIfPermitted()
                } catch (e: Exception) {
                    "${e.javaClass.simpleName}: ${e.message}"
                } ?: return@launch
                val line = "qtpy_auto_connect → $result"
                lastResult = line
                append(line)
                if (result.contains("connected", ignoreCase = true)) {
                    Toast.makeText(context, result, Toast.LENGTH_SHORT).show()
                }
            }
        }
        activity.lifecycle.addObserver(observer)
        onDispose { activity.lifecycle.removeObserver(observer) }
    }

    Column(
        modifier = modifier
            .fillMaxWidth()
            .fillMaxSize()
            .verticalScroll(scroll)
            .padding(24.dp),
        verticalArrangement = Arrangement.spacedBy(12.dp),
        horizontalAlignment = Alignment.Start,
    ) {
        Text(
            text = stringResource(R.string.neon_section_title),
            style = MaterialTheme.typography.titleLarge,
        )
        Text(
            text = stringResource(R.string.neon_hint),
            style = MaterialTheme.typography.bodyMedium,
        )
        Text(
            text = if (busy) stringResource(R.string.neon_busy) else "QT Py bridge: ${integration.qtPyReady()}",
            style = MaterialTheme.typography.labelLarge,
        )
        Text(
            text = stringResource(
                R.string.qtpy_status_line,
                qtPyState.connected.toString(),
                qtPyState.reading.toString(),
                qtPyState.linesReceived.toString(),
                qtPyState.firmwareState.name,
            ),
            style = MaterialTheme.typography.labelMedium,
        )
        Text(
            text = stringResource(R.string.qtpy_firmware_hint),
            style = MaterialTheme.typography.bodySmall,
        )
        Row(
            verticalAlignment = Alignment.CenterVertically,
            modifier = Modifier.padding(top = 4.dp),
        ) {
            Switch(
                checked = verboseSerialMonitor,
                onCheckedChange = { verboseSerialMonitor = it },
            )
            Text(
                text = stringResource(R.string.qtpy_verbose_serial_switch),
                style = MaterialTheme.typography.labelMedium,
                modifier = Modifier.padding(start = 8.dp),
            )
        }
        if (verboseSerialMonitor) {
            Text(
                text = stringResource(R.string.qtpy_last_line, qtPyState.lastLine.ifBlank { "(none yet)" }),
                style = MaterialTheme.typography.bodySmall,
            )
        }
        Row(
            verticalAlignment = Alignment.CenterVertically,
            modifier = Modifier.padding(top = 2.dp),
        ) {
            Switch(
                checked = verboseTimingDetails,
                onCheckedChange = { verboseTimingDetails = it },
            )
            Text(
                text = stringResource(R.string.qtpy_verbose_timing_switch),
                style = MaterialTheme.typography.labelMedium,
                modifier = Modifier.padding(start = 8.dp),
            )
        }
        if (verboseTimingDetails) {
            Text(
                text = stringResource(
                    R.string.qtpy_timing_line,
                    qtPyState.lastSeq?.toString() ?: "-",
                    qtPyState.lastTimeUs?.toString() ?: "-",
                    qtPyState.lastTimeUsExtended?.toString() ?: "-",
                ),
                style = MaterialTheme.typography.bodySmall,
            )
            Text(
                text = stringResource(
                    R.string.qtpy_recv_time_line,
                    qtPyState.lastRecvElapsedRealtimeNs?.toString() ?: "-",
                    qtPyState.lastRecvWallTimeNs?.toString() ?: "-",
                ),
                style = MaterialTheme.typography.bodySmall,
            )
        }
        Text(
            text = qtPyState.status,
            style = MaterialTheme.typography.bodySmall,
        )
        if (verboseSerialMonitor) {
            Text(
                text = stringResource(R.string.qtpy_monitor_title),
                style = MaterialTheme.typography.titleSmall,
            )
            Card(
                modifier = Modifier
                    .fillMaxWidth()
                    .height(180.dp),
                colors = CardDefaults.cardColors(
                    containerColor = MaterialTheme.colorScheme.surfaceVariant,
                ),
            ) {
                Text(
                    text = if (qtPyState.recentLines.isEmpty()) {
                        stringResource(R.string.qtpy_monitor_empty)
                    } else {
                        qtPyState.recentLines.joinToString(separator = "\n")
                    },
                    style = MaterialTheme.typography.bodySmall,
                    modifier = Modifier
                        .fillMaxSize()
                        .verticalScroll(monitorScroll)
                        .padding(10.dp),
                )
            }
        }

        Button(
            onClick = { runNeon("qtpy_connect") { integration.qtPyConnectResult() } },
            enabled = !busy,
        ) { Text(stringResource(R.string.qtpy_btn_connect)) }
        Button(
            onClick = { runNeon("qtpy_disconnect") { integration.qtPyDisconnectResult() } },
            enabled = !busy,
        ) { Text(stringResource(R.string.qtpy_btn_disconnect)) }
        Button(
            onClick = { integration.qtPyClearMonitor() },
            enabled = !busy,
        ) { Text(stringResource(R.string.qtpy_btn_clear_monitor)) }

        Row(
            modifier = Modifier.fillMaxWidth(),
            horizontalArrangement = Arrangement.spacedBy(8.dp),
        ) {
            Button(
                onClick = { runNeon("qtpy_cmd") { integration.qtPySendCommand("CALIBRATE") } },
                enabled = !busy && qtPyState.connected,
                modifier = Modifier.weight(1f),
            ) { Text(stringResource(R.string.qtpy_btn_calibrate)) }
            Button(
                onClick = { runNeon("qtpy_cmd") { integration.qtPySendCommand("START") } },
                enabled = !busy && qtPyState.connected,
                modifier = Modifier.weight(1f),
            ) { Text(stringResource(R.string.qtpy_btn_start_stream)) }
            Button(
                onClick = { runNeon("qtpy_cmd") { integration.qtPySendCommand("STOP") } },
                enabled = !busy && qtPyState.connected,
                modifier = Modifier.weight(1f),
            ) { Text(stringResource(R.string.qtpy_btn_stop_stream)) }
        }

        HorizontalDivider(modifier = Modifier.padding(vertical = 8.dp))
        Text(
            text = stringResource(R.string.pc_bridge_section_title),
            style = MaterialTheme.typography.titleMedium,
        )
        Text(
            text = stringResource(R.string.pc_bridge_hint),
            style = MaterialTheme.typography.bodySmall,
        )
        Text(
            text = pcBridgeState.status,
            style = MaterialTheme.typography.bodySmall,
        )
        if (pcBridgeState.bindHint.isNotEmpty()) {
            Text(
                text = pcBridgeState.bindHint,
                style = MaterialTheme.typography.labelMedium,
            )
        }
        Button(
            onClick = {
                val msg = integration.pcBridgeStart()
                append("pc_bridge → $msg")
                lastResult = msg
                Toast.makeText(context, msg, Toast.LENGTH_LONG).show()
            },
            enabled = !busy && !pcBridgeState.running,
        ) { Text(stringResource(R.string.pc_bridge_btn_start)) }
        Button(
            onClick = {
                val msg = integration.pcBridgeStop()
                append("pc_bridge → $msg")
                lastResult = msg
                Toast.makeText(context, msg, Toast.LENGTH_SHORT).show()
            },
            enabled = !busy && pcBridgeState.running,
        ) { Text(stringResource(R.string.pc_bridge_btn_stop)) }

        HorizontalDivider(modifier = Modifier.padding(vertical = 8.dp))
        Text(
            text = stringResource(R.string.flow_section_title),
            style = MaterialTheme.typography.titleMedium,
        )
        Button(
            onClick = { runNeon("start_both") { integration.startBothIfQtPyConnected() } },
            enabled = !busy && qtPyState.isStreaming && !combinedActive,
        ) { Text(stringResource(R.string.flow_btn_start_both)) }
        Button(
            onClick = { runNeon("stop_both") { integration.stopBothSession() } },
            enabled = !busy && combinedActive,
        ) { Text(stringResource(R.string.flow_btn_stop_both)) }

        HorizontalDivider(modifier = Modifier.padding(vertical = 8.dp))
        Text(
            text = recordingIdDisplay?.let { stringResource(R.string.neon_active_recording, it) }
                ?: stringResource(R.string.neon_active_recording_none),
            style = MaterialTheme.typography.labelMedium,
        )
        csvSessionInfo?.let {
            Text(
                text = it,
                style = MaterialTheme.typography.bodySmall,
            )
        }

        Card(
            modifier = Modifier.fillMaxWidth(),
            colors = CardDefaults.cardColors(
                containerColor = MaterialTheme.colorScheme.surfaceVariant,
            ),
        ) {
            Column(Modifier.padding(16.dp)) {
                Text(
                    text = stringResource(R.string.neon_last_result_title),
                    style = MaterialTheme.typography.titleSmall,
                )
                Text(
                    text = if (lastResult.isEmpty()) {
                        stringResource(R.string.neon_last_result_empty)
                    } else {
                        lastResult
                    },
                    style = MaterialTheme.typography.bodyLarge,
                    modifier = Modifier.padding(top = 6.dp),
                )
            }
        }

        Text(
            text = stringResource(R.string.neon_section_flow),
            style = MaterialTheme.typography.titleMedium,
            modifier = Modifier.padding(top = 8.dp),
        )
        Button(
            onClick = { runNeon("smoke") { integration.neonSmokeTest() } },
            enabled = !busy,
        ) { Text(stringResource(R.string.neon_btn_smoke)) }
        Button(
            onClick = { runNeon("session_start") { integration.neonStartFootSessionLikeRecorder() } },
            enabled = !busy,
        ) { Text(stringResource(R.string.neon_btn_session_start)) }
        Button(
            onClick = { runNeon("session_stop") { integration.neonStopFootSessionLikeRecorder() } },
            enabled = !busy,
        ) { Text(stringResource(R.string.neon_btn_session_stop)) }

        HorizontalDivider(modifier = Modifier.padding(vertical = 8.dp))

        Text(
            text = stringResource(R.string.neon_section_raw),
            style = MaterialTheme.typography.titleMedium,
        )
        Button(
            onClick = { runNeon("status") { integration.neonStatusSummary() } },
            enabled = !busy,
        ) { Text(stringResource(R.string.neon_btn_status)) }
        Button(
            onClick = { runNeon("start") { integration.neonStartRecordingResult() } },
            enabled = !busy,
        ) { Text(stringResource(R.string.neon_btn_start)) }
        Button(
            onClick = { runNeon("stop") { integration.neonStopRecordingResult() } },
            enabled = !busy,
        ) { Text(stringResource(R.string.neon_btn_stop)) }
        Button(
            onClick = { runNeon("event") { integration.neonSendProbeEvent() } },
            enabled = !busy,
        ) { Text(stringResource(R.string.neon_btn_event)) }

        Text(
            text = if (log.isEmpty()) stringResource(R.string.neon_hint) else log,
            style = MaterialTheme.typography.bodySmall,
            modifier = Modifier.padding(top = 8.dp),
        )

        HorizontalDivider(modifier = Modifier.padding(vertical = 12.dp))
        Text(
            text = stringResource(R.string.qtpy_latency_diag_title),
            style = MaterialTheme.typography.titleSmall,
        )
        Text(
            text = stringResource(R.string.qtpy_latency_diag_hint),
            style = MaterialTheme.typography.bodySmall,
            modifier = Modifier.padding(bottom = 6.dp),
        )
        Card(
            modifier = Modifier.fillMaxWidth(),
            colors = CardDefaults.cardColors(
                containerColor = MaterialTheme.colorScheme.surfaceVariant,
            ),
        ) {
            Text(
                text = QtPyLinkLatencyDiagnostics.summarize(qtPyState.recentRecords),
                style = MaterialTheme.typography.bodySmall,
                modifier = Modifier.padding(12.dp),
            )
        }
    }
}

@Preview(showBackground = true)
@Composable
fun NeonProbeScreenPreview() {
    URP2026Theme {
        val context = LocalContext.current
        NeonProbeScreen(integration = ImuRecorderIntegration(context))
    }
}
