package com.example.urp2026

import android.os.Bundle
import android.widget.Toast
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.activity.enableEdgeToEdge
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
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
import com.example.urp2026.integration.ImuRecorderIntegration
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
    val monitorScroll = rememberScrollState()
    var log by remember { mutableStateOf("") }
    var lastResult by remember { mutableStateOf("") }
    var busy by remember { mutableStateOf(false) }
    var recordingIdDisplay by remember { mutableStateOf<String?>(integration.neonActiveRecordingId()) }
    var combinedActive by remember { mutableStateOf(integration.isCombinedSessionActive()) }
    val scope = rememberCoroutineScope()
    val scroll = rememberScrollState()

    fun refreshRecordingLabel() {
        recordingIdDisplay = integration.neonActiveRecordingId()
        combinedActive = integration.isCombinedSessionActive()
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
            ),
            style = MaterialTheme.typography.labelMedium,
        )
        Text(
            text = stringResource(R.string.qtpy_last_line, qtPyState.lastLine.ifBlank { "(none yet)" }),
            style = MaterialTheme.typography.bodySmall,
        )
        Text(
            text = qtPyState.status,
            style = MaterialTheme.typography.bodySmall,
        )
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

        HorizontalDivider(modifier = Modifier.padding(vertical = 8.dp))
        Text(
            text = stringResource(R.string.flow_section_title),
            style = MaterialTheme.typography.titleMedium,
        )
        Button(
            onClick = { runNeon("start_both") { integration.startBothIfQtPyConnected() } },
            enabled = !busy && qtPyState.connected && !combinedActive,
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
