package com.example.urp2026.qtpy

import kotlin.math.sqrt

/**
 * Bench-style diagnostics: compares **consecutive IMU CSV rows** using
 * `time_us_extended` (QT Py / firmware timeline) vs `recvElapsedRealtimeNs`
 * (Android monotonic receive time).
 *
 * Residual = Δrecv_ns − Δext_us×1000 → mostly **USB + Android scheduling +
 * USB batching** (and ms wall-clock effects if you used wall time — we use
 * elapsed here). **IMU ↔ QT Py I²C latency is not visible** on the phone;
 * measure that on the board with `micros()` around `getEvent`.
 */
object QtPyLinkLatencyDiagnostics {

    private const val MIN_DT_US = 500L
    private const val MAX_DT_US = 25_000L

    fun summarize(records: List<QtPyLineRecord>): String {
        val rows = records
            .filter { it.seq != null && it.timeUsExtended != null }
            .sortedBy { it.seq!! }
        if (rows.size < 2) {
            return "Latency diag: need ≥2 IMU rows with seq + ext_us (got ${rows.size}). Stream firmware CSV while connected."
        }

        val residuals = ArrayList<Long>(256)
        val dtStfList = ArrayList<Long>(256)
        val dtRecvList = ArrayList<Long>(256)
        var skippedGap = 0

        var prev = rows[0]
        for (i in 1 until rows.size) {
            val cur = rows[i]
            val pSeq = prev.seq!!
            val cSeq = cur.seq!!
            if (cSeq != pSeq + 1L) {
                skippedGap++
                prev = cur
                continue
            }
            val dtUs = cur.timeUsExtended!! - prev.timeUsExtended!!
            val dtRecvNs = cur.recvElapsedRealtimeNs - prev.recvElapsedRealtimeNs
            if (dtUs !in MIN_DT_US..MAX_DT_US || dtRecvNs <= 0L) {
                skippedGap++
                prev = cur
                continue
            }
            dtStfList.add(dtUs)
            dtRecvList.add(dtRecvNs)
            residuals.add(dtRecvNs - dtUs * 1000L)
            prev = cur
        }

        if (residuals.isEmpty()) {
            return "Latency diag: no valid consecutive seq pairs (gaps/skips=$skippedGap). Clear monitor and stream again."
        }

        val meanResNs = residuals.sum().toDouble() / residuals.size
        val varRes = residuals.sumOf { val d = it - meanResNs; d * d } / residuals.size
        val stdevResNs = sqrt(varRes)
        val maxAbsResNs = residuals.maxOf { kotlin.math.abs(it) }

        val medianStfUs = medianLong(dtStfList)
        val medianRecvMs = medianLong(dtRecvList) / 1_000_000.0

        return buildString {
            append("Pairs: ${residuals.size} (seq+1 only, Δext_us in ${MIN_DT_US}…${MAX_DT_US} µs)\n")
            append("Median ΔSampleTimeFine(ext): ${medianStfUs} µs\n")
            append("Median Δrecv_elapsed: ${"%.3f".format(medianRecvMs)} ms\n")
            append("Residual Δrecv − Δext: mean ${"%.3f".format(meanResNs / 1_000_000.0)} ms, ")
            append("σ ${"%.3f".format(stdevResNs / 1_000_000.0)} ms, ")
            append("max|·| ${"%.3f".format(maxAbsResNs / 1_000_000.0)} ms\n")
            append("Interpret: residual ≈ host/USB jitter; I²C on QT Py needs firmware timing.")
        }
    }

    private fun medianLong(sortedInput: List<Long>): Long {
        if (sortedInput.isEmpty()) return 0L
        val s = sortedInput.sorted()
        val mid = s.size / 2
        return if (s.size % 2 == 1) s[mid] else (s[mid - 1] + s[mid]) / 2
    }
}
