package com.example.urp2026.integration

import java.io.BufferedWriter
import java.io.File
import java.time.LocalDateTime
import java.time.format.DateTimeFormatter
import kotlin.math.max

private const val STATUS_ICM_RAW = 1

/**
 * Writes two CSV files (LF/RF) compatible with the Python dual recorder OUTPUT_COLUMNS layout.
 */
class DualImuCsvRecorder(private val rootDir: File) {
    private var lfWriter: BufferedWriter? = null
    private var rfWriter: BufferedWriter? = null
    private var lfFile: File? = null
    private var rfFile: File? = null
    private var prevTimeUs: Long? = null

    @Synchronized
    fun startNewSession(): Pair<File, File> {
        close()
        val ts = LocalDateTime.now().format(DateTimeFormatter.ofPattern("yyyyMMdd_HHmmss"))
        val sessionDir = File(rootDir, "imu_sessions/$ts").apply { mkdirs() }
        lfFile = File(sessionDir, "LF_imu_fused_$ts.csv")
        rfFile = File(sessionDir, "RF_imu_fused_$ts.csv")
        lfWriter = lfFile!!.bufferedWriter()
        rfWriter = rfFile!!.bufferedWriter()
        writeHeader(lfWriter!!)
        writeHeader(rfWriter!!)
        prevTimeUs = null
        return lfFile!! to rfFile!!
    }

    @Synchronized
    fun appendIcm20(
        seq: Long,
        timeUs: Long,
        timeUsExtended: Long?,
        recvElapsedNs: Long?,
        tUtcNs: Long,
        lf: DoubleArray,
        rf: DoubleArray,
    ) {
        val lfW = lfWriter ?: return
        val rfW = rfWriter ?: return
        val dt = computeDtSec(timeUs)

        val lfRow = buildRow(
            seq = seq,
            timeUs = timeUs,
            timeUsExtended = timeUsExtended,
            recvElapsedNs = recvElapsedNs,
            tUtcNs = tUtcNs,
            acc = lf.copyOfRange(0, 3),
            gyr = lf.copyOfRange(3, 6),
            mag = lf.copyOfRange(6, 9),
            dtSec = dt,
        )
        val rfRow = buildRow(
            seq = seq,
            timeUs = timeUs,
            timeUsExtended = timeUsExtended,
            recvElapsedNs = recvElapsedNs,
            tUtcNs = tUtcNs,
            acc = rf.copyOfRange(0, 3),
            gyr = rf.copyOfRange(3, 6),
            mag = rf.copyOfRange(6, 9),
            dtSec = dt,
        )

        lfW.write(lfRow)
        lfW.newLine()
        rfW.write(rfRow)
        rfW.newLine()
    }

    @Synchronized
    fun close() {
        try {
            lfWriter?.flush()
            rfWriter?.flush()
            lfWriter?.close()
            rfWriter?.close()
        } catch (_: Exception) {
        } finally {
            lfWriter = null
            rfWriter = null
            prevTimeUs = null
        }
    }

    private fun computeDtSec(timeUs: Long): Double {
        val prev = prevTimeUs
        prevTimeUs = timeUs
        if (prev == null) return 0.005
        val dt = (timeUs - prev).toDouble() / 1_000_000.0
        return if (dt <= 0.0 || dt > 0.5) 0.005 else dt
    }

    private fun writeHeader(writer: BufferedWriter) {
        repeat(7) { writer.newLine() }
        writer.write(OUTPUT_COLUMNS.joinToString(","))
        writer.newLine()
    }

    private fun buildRow(
        seq: Long,
        timeUs: Long,
        timeUsExtended: Long?,
        recvElapsedNs: Long?,
        tUtcNs: Long,
        acc: DoubleArray,
        gyr: DoubleArray,
        mag: DoubleArray,
        dtSec: Double,
    ): String {
        val zq = listOf(0.0, 0.0, 0.0, 0.0) // Quat
        val zdq = listOf(0.0, 0.0, 0.0, 0.0) // dQuat
        val dv = listOf(acc[0] * dtSec, acc[1] * dtSec, acc[2] * dtSec)
        val values = mutableListOf<String>()
        values += seq.toString()
        values += max(0L, timeUs).toString()
        values += (timeUsExtended ?: -1L).toString()
        values += (recvElapsedNs ?: -1L).toString()
        values += tUtcNs.toString()
        values += zq.map(::fmt)
        values += zdq.map(::fmt)
        values += dv.map(::fmt)
        values += acc.map(::fmt)
        values += gyr.map(::fmt)
        values += mag.map(::fmt)
        values += STATUS_ICM_RAW.toString()
        return values.joinToString(",")
    }

    private fun fmt(v: Double): String = "%.6f".format(v)

    companion object {
        private val OUTPUT_COLUMNS = listOf(
            "PacketCounter", "SampleTimeFine", "time_us_extended", "recv_elapsed_ns", "t_utc_ns",
            "Quat_W", "Quat_X", "Quat_Y", "Quat_Z",
            "dq_W", "dq_X", "dq_Y", "dq_Z",
            "dv[1]", "dv[2]", "dv[3]",
            "Acc_X", "Acc_Y", "Acc_Z",
            "Gyr_X", "Gyr_Y", "Gyr_Z",
            "Mag_X", "Mag_Y", "Mag_Z",
            "Status",
        )
    }
}

