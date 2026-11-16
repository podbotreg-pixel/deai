package dev.dinet.app

import android.content.Context
import android.content.Intent
import android.content.IntentFilter
import android.net.ConnectivityManager
import android.net.NetworkCapabilities
import android.os.BatteryManager
import android.util.Log
import androidx.work.Constraints
import androidx.work.CoroutineWorker
import androidx.work.ExistingPeriodicWorkPolicy
import androidx.work.ExistingWorkPolicy
import androidx.work.NetworkType
import androidx.work.OneTimeWorkRequestBuilder
import androidx.work.PeriodicWorkRequestBuilder
import androidx.work.WorkManager
import androidx.work.WorkerParameters
import java.io.File
import java.time.Duration

/**
 * Фоновый вклад телефона (M3 шаг 2): WorkManager будит воркер периодически;
 * воркер проходит ГЕЙТЫ (батарея ≥70%, термал <44°C, Wi-Fi unmetered) и
 * гонит эру роя по новому материалу через Rust-ядро (`Core.tick()`).
 *
 * - РАСПИСАНИЕ держит WorkManager: constraint UNMETERED (Wi-Fi без лимита)
 *   на уровне планировщика, плюс battery-not-low; ДОД «батарея ≥70%» и
 *   термал WorkManager выразить нельзя — проверяются внутри воркера.
 * - Батарея читается из sticky-интента ACTION_BATTERY_CHANGED: путь
 *   BATTERY_PROPERTY_CAPACITY игнорирует сервисную подмену (эмулятор),
 *   а интент — ровно то, что видит система.
 * - Воркер может стартовать в свежем процессе (без UI): `tick` поднимает
 *   ядро сам по запомненному каталогу KV.
 * - Вклад пока ЛОКАЛЬНЫЙ (эры роя на своём материале); сетевой dial роя —
 *   следующий шаг. Параллельные процессы против KV-лога: воркер ходит в
 *   том же процессе приложения (WorkManager по умолчанию), конфликт
 *   двух процессов на append-only KV разберём вместе с сетевым шагом.
 */
class TickWorker(ctx: Context, params: WorkerParameters) : CoroutineWorker(ctx, params) {

    override suspend fun doWork(): Result {
        val level = batteryLevel()
        val temp = thermalMax()
        val wifi = wifiUnmetered()
        log(
            "гейты: батарея=${level ?: "не читается"}%, " +
                "термал=${temp?.let { "${it / 1000.0}°C" } ?: "нет зон"}, " +
                "wifi-unmetered=$wifi"
        )

        // батарея: ≥70%; нечитаемая (эмулятор без батареи) — гейт не держим
        if (level != null && level in 1..69) {
            log("батарея $level% < 70% — вклад пропущен")
            return Result.success()
        }
        if (temp != null && temp >= HOT_MILLICELSIUS) {
            log("термал ${temp / 1000.0}°C ≥ 44.0°C — вклад пропущен")
            return Result.success()
        }
        if (!wifi) {
            log("сеть не Wi-Fi-unmetered — вклад пропущен")
            return Result.success()
        }

        val dir = File(applicationContext.filesDir, "dinet").absolutePath
        val opened = try {
            Core.open(dir)
        } catch (e: Throwable) {
            false
        }
        if (!opened) {
            log("ядро не открылось в фоне")
            return Result.retry()
        }
        val note = try {
            Core.tick()
        } catch (e: Throwable) {
            "ошибка: ${e.message}"
        }
        log("фоновый вклад: $note")
        return Result.success()
    }

    private fun log(msg: String) {
        Log.i(TAG, msg)
    }

    /// Заряд батареи в % из sticky-интента (то, что видит система);
    /// null — батарея не симулируется.
    private fun batteryLevel(): Int? {
        val intent = try {
            applicationContext.registerReceiver(
                null,
                IntentFilter(Intent.ACTION_BATTERY_CHANGED),
            )
        } catch (e: Throwable) {
            null
        } ?: return null
        val lvl = intent.getIntExtra(BatteryManager.EXTRA_LEVEL, -1)
        val scale = intent.getIntExtra(BatteryManager.EXTRA_SCALE, -1)
        if (lvl < 0 || scale <= 0) return null
        return lvl * 100 / scale
    }

    /// Самая горячая thermal-зона (милли°C); null — зон нет/не читаются.
    private fun thermalMax(): Int? {
        var max: Int? = null
        val zones = File("/sys/class/thermal")
            .listFiles { f -> f.name.startsWith("thermal_zone") } ?: return null
        for (z in zones) {
            val t = File(z, "temp").readText().trim().toIntOrNull() ?: continue
            if (t <= 0) continue // пропущенные зоны отдают 0/минус — не грейт
            if (max == null || t > max) max = t
        }
        return max
    }

    /// Wi-Fi без лимита (дубль констрейнта UNMETERED — для честного лога).
    private fun wifiUnmetered(): Boolean {
        val cm = applicationContext.getSystemService(ConnectivityManager::class.java)
            ?: return false
        val net = cm.activeNetwork ?: return false
        val caps = cm.getNetworkCapabilities(net) ?: return false
        return caps.hasTransport(NetworkCapabilities.TRANSPORT_WIFI) &&
            caps.hasCapability(NetworkCapabilities.NET_CAPABILITY_NOT_METERED)
    }

    companion object {
        const val TAG = "dinet-bg"
        /// Порог остановки: 44°C (милли°C) — телефон горячий, отдаём прохладу.
        const val HOT_MILLICELSIUS = 44_000
        const val PERIODIC = "dinet-bg-periodic"
        const val ONCE = "dinet-bg-once"

        /** Гейты планировщика: unmetered Wi-Fi + не низкая батарея. */
        private fun gates(): Constraints = Constraints.Builder()
            .setRequiredNetworkType(NetworkType.UNMETERED)
            .setRequiresBatteryNotLow(true)
            .build()

        /// Периодический вклад (6 часов, KEEP — не перетираем существующий).
        fun schedulePeriodic(context: Context) {
            WorkManager.getInstance(context).enqueueUniquePeriodicWork(
                PERIODIC,
                ExistingPeriodicWorkPolicy.KEEP,
                PeriodicWorkRequestBuilder<TickWorker>(Duration.ofHours(6))
                    .setConstraints(gates())
                    .build(),
            )
        }

        /// «Вклад сейчас» (гейты те же; запускается, если сеть подходит).
        fun scheduleOnce(context: Context) {
            WorkManager.getInstance(context).enqueueUniqueWork(
                ONCE,
                ExistingWorkPolicy.KEEP,
                OneTimeWorkRequestBuilder<TickWorker>()
                    .setConstraints(gates())
                    .build(),
            )
        }
    }
}
