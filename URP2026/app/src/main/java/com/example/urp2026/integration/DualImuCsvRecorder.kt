package com.example.urp2026.integration

import java.io.BufferedWriter
import java.io.File
import java.time.LocalDateTime
import java.time.format.DateTimeFormatter
import kotlin.math.max

/**
 * Writes two CSV files (LF / RF), one foot per file.
 * Each row is **11 columns**: timing/meta + Acc + Gyr (no mag — matches accel/gyro-only QT Py firmware).
 */
class DualImuCsvRecorder(private val rootDir: File) {
    private var lfWriter: BufferedWriter? = null
    private var rfWriter: BufferedWriter? = null
    private var lfFile: File? = null
    private var rfFile: File? = null

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

        val lfRow = buildRow(
            seq = seq,
            timeUs = timeUs,
            timeUsExtended = timeUsExtended,
            recvElapsedNs = recvElapsedNs,
            tUtcNs = tUtcNs,
            acc = lf.copyOfRange(0, 3),
            gyr = lf.copyOfRange(3, 6),
        )
        val rfRow = buildRow(
            seq = seq,
            timeUs = timeUs,
            timeUsExtended = timeUsExtended,
            recvElapsedNs = recvElapsedNs,
            tUtcNs = tUtcNs,
            acc = rf.copyOfRange(0, 3),
            gyr = rf.copyOfRange(3, 6),
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
        }
    }

    private fun writeHeader(writer: BufferedWriter) {
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
    ): String {
        val values = mutableListOf<String>()
        values += seq.toString()
        values += max(0L, timeUs).toString()
        values += (timeUsExtended ?: -1L).toString()
        values += (recvElapsedNs ?: -1L).toString()
        values += tUtcNs.toString()
        values += acc.map(::fmt)
        values += gyr.map(::fmt)
        return values.joinToString(",")
    }

    private fun fmt(v: Double): String = "%.6f".format(v)

    companion object {
        private val OUTPUT_COLUMNS = listOf(
            "PacketCounter",
            "SampleTimeFine",
            "time_us_extended",
            "recv_elapsed_ns",
            "t_utc_ns",
            "Acc_X", "Acc_Y", "Acc_Z",
            "Gyr_X", "Gyr_Y", "Gyr_Z",
        )
    }
}
