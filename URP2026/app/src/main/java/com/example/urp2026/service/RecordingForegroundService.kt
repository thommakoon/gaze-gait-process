package com.example.urp2026.service

import android.app.Notification
import android.app.NotificationChannel
import android.app.NotificationManager
import android.app.PendingIntent
import android.app.Service
import android.content.Context
import android.content.Intent
import android.content.pm.ServiceInfo
import android.os.Build
import android.os.IBinder
import android.os.PowerManager
import android.util.Log
import androidx.core.app.NotificationCompat
import androidx.core.content.ContextCompat
import com.example.urp2026.MainActivity
import com.example.urp2026.R
import com.example.urp2026.integration.RecorderRuntime

/**
 * Keeps QT Py USB reader, CSV session, and PC bridge alive while the Activity is backgrounded
 * or the screen is off (partial wake lock — step 3).
 *
 * Uses [dataSync] FGS type (targetSdk 34+): recording IMU CSV + Neon/PC session.
 */
class RecordingForegroundService : Service() {

    private var wakeLock: PowerManager.WakeLock? = null

    override fun onBind(intent: Intent?): IBinder? = null

    override fun onCreate() {
        super.onCreate()
        ensureChannel()
        // Call startForeground BEFORE any heavy init — Android kills the app otherwise.
        promoteToForeground(buildNotification())
        acquireWakeLock()
        try {
            RecorderRuntime.get(this)
        } catch (t: Throwable) {
            Log.e(TAG, "RecorderRuntime init failed", t)
        }
        isRunning = true
    }

    override fun onStartCommand(intent: Intent?, flags: Int, startId: Int): Int {
        when (intent?.action) {
            ACTION_STOP -> {
                isRunning = false
                releaseWakeLock()
                stopForeground(STOP_FOREGROUND_REMOVE)
                stopSelf()
                return START_NOT_STICKY
            }
            else -> {
                // START / REFRESH / null (system restart)
                acquireWakeLock()
                promoteToForeground(buildNotification())
            }
        }
        return START_STICKY
    }

    override fun onDestroy() {
        isRunning = false
        releaseWakeLock()
        super.onDestroy()
    }

    private fun acquireWakeLock() {
        val existing = wakeLock
        if (existing?.isHeld == true) return
        try {
            val pm = getSystemService(PowerManager::class.java) ?: return
            val lock = existing ?: pm.newWakeLock(
                PowerManager.PARTIAL_WAKE_LOCK,
                WAKE_LOCK_TAG,
            ).also {
                it.setReferenceCounted(false)
                wakeLock = it
            }
            // No timeout: held for the lifetime of this FG service (study sessions).
            // Released on Stop service / onDestroy.
            @Suppress("DEPRECATION")
            lock.acquire()
            Log.i(TAG, "PARTIAL_WAKE_LOCK acquired")
        } catch (t: Throwable) {
            Log.e(TAG, "acquireWakeLock failed", t)
        }
    }

    private fun releaseWakeLock() {
        try {
            val lock = wakeLock
            if (lock?.isHeld == true) {
                lock.release()
                Log.i(TAG, "PARTIAL_WAKE_LOCK released")
            }
        } catch (t: Throwable) {
            Log.e(TAG, "releaseWakeLock failed", t)
        } finally {
            wakeLock = null
        }
    }

    private fun ensureChannel() {
        if (Build.VERSION.SDK_INT < Build.VERSION_CODES.O) return
        val mgr = getSystemService(NotificationManager::class.java) ?: return
        val channel = NotificationChannel(
            CHANNEL_ID,
            getString(R.string.fg_channel_name),
            NotificationManager.IMPORTANCE_LOW,
        ).apply {
            description = getString(R.string.fg_channel_desc)
            setShowBadge(false)
        }
        mgr.createNotificationChannel(channel)
    }

    private fun promoteToForeground(notification: Notification) {
        try {
            if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.UPSIDE_DOWN_CAKE) {
                startForeground(
                    NOTIFICATION_ID,
                    notification,
                    ServiceInfo.FOREGROUND_SERVICE_TYPE_DATA_SYNC,
                )
            } else if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.Q) {
                startForeground(
                    NOTIFICATION_ID,
                    notification,
                    ServiceInfo.FOREGROUND_SERVICE_TYPE_DATA_SYNC,
                )
            } else {
                startForeground(NOTIFICATION_ID, notification)
            }
        } catch (t: Throwable) {
            // Last-resort: type-less promote so Activity can still open.
            Log.e(TAG, "startForeground(dataSync) failed; retrying without type", t)
            try {
                startForeground(NOTIFICATION_ID, notification)
            } catch (t2: Throwable) {
                Log.e(TAG, "startForeground failed", t2)
            }
        }
    }

    private fun buildNotification(): Notification {
        val openApp = PendingIntent.getActivity(
            this,
            0,
            Intent(this, MainActivity::class.java).addFlags(Intent.FLAG_ACTIVITY_SINGLE_TOP),
            PendingIntent.FLAG_UPDATE_CURRENT or PendingIntent.FLAG_IMMUTABLE,
        )
        val stopSelf = PendingIntent.getService(
            this,
            1,
            Intent(this, RecordingForegroundService::class.java).setAction(ACTION_STOP),
            PendingIntent.FLAG_UPDATE_CURRENT or PendingIntent.FLAG_IMMUTABLE,
        )

        val runtime = RecorderRuntime.peek()
        val connected = runtime?.qtPyReady() == true
        val streaming = runtime?.qtPyIsStreaming() == true
        val recording = runtime?.isCombinedSessionActive() == true
        val bridge = runtime?.pcBridgeRunning() == true
        val neonId = runtime?.neonActiveRecordingId()
        val wakeHeld = wakeLock?.isHeld == true

        val title = when {
            recording -> getString(R.string.fg_title_recording)
            streaming -> getString(R.string.fg_title_streaming)
            connected -> getString(R.string.fg_title_connected)
            bridge -> getString(R.string.fg_title_bridge)
            else -> getString(R.string.fg_title_idle)
        }
        val text = buildString {
            append(if (connected) "USB on" else "USB off")
            append(" · ")
            append(if (streaming) "STREAMING" else "idle")
            append(" · ")
            append(if (recording) "REC" else "not recording")
            if (bridge) append(" · PC bridge")
            if (wakeHeld) append(" · wake")
            neonId?.let { append(" · neon=").append(it) }
        }

        return NotificationCompat.Builder(this, CHANNEL_ID)
            .setContentTitle(title)
            .setContentText(text)
            .setSmallIcon(R.drawable.ic_stat_fg)
            .setContentIntent(openApp)
            .setOngoing(true)
            .setOnlyAlertOnce(true)
            .setCategory(NotificationCompat.CATEGORY_SERVICE)
            .addAction(0, getString(R.string.fg_action_stop), stopSelf)
            .build()
    }

    companion object {
        private const val TAG = "UrpFgService"
        private const val WAKE_LOCK_TAG = "URP2026:PartialWake"
        const val CHANNEL_ID = "urp2026_recording"
        const val NOTIFICATION_ID = 2026
        const val ACTION_START = "com.example.urp2026.action.FG_START"
        const val ACTION_REFRESH = "com.example.urp2026.action.FG_REFRESH"
        const val ACTION_STOP = "com.example.urp2026.action.FG_STOP"

        @Volatile
        private var isRunning: Boolean = false

        fun ensureStarted(context: Context) {
            val app = context.applicationContext
            try {
                RecorderRuntime.get(app)
            } catch (t: Throwable) {
                Log.e(TAG, "ensureStarted: runtime failed", t)
            }
            if (isRunning) {
                refresh(app)
                return
            }
            val intent = Intent(app, RecordingForegroundService::class.java).setAction(ACTION_START)
            try {
                ContextCompat.startForegroundService(app, intent)
            } catch (t: Throwable) {
                Log.e(TAG, "startForegroundService failed", t)
            }
        }

        fun refresh(context: Context) {
            val app = context.applicationContext
            if (!isRunning && RecorderRuntime.peek() == null) return
            val intent = Intent(app, RecordingForegroundService::class.java).setAction(ACTION_REFRESH)
            try {
                // Prefer startService once already foreground — avoids repeated FGS start timeouts.
                if (isRunning) {
                    app.startService(intent)
                } else {
                    ContextCompat.startForegroundService(app, intent)
                }
            } catch (t: Throwable) {
                Log.e(TAG, "refresh failed", t)
            }
        }

        fun stop(context: Context) {
            val app = context.applicationContext
            val intent = Intent(app, RecordingForegroundService::class.java).setAction(ACTION_STOP)
            try {
                app.startService(intent)
            } catch (_: Exception) {
                try {
                    app.stopService(Intent(app, RecordingForegroundService::class.java))
                } catch (_: Exception) {
                }
            }
        }
    }
}
