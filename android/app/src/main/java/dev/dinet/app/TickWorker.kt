package dev.dinet.app

import android.content.Context
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
 * воркер проходит ГЕЙТЫ (см. [Gates]) и гонит эру роя по новому материалу
 * через Rust-ядро (`Core.tick()`).
 *
 * - РАСПИСАНИЕ держит WorkManager: constraint UNMETERED (Wi-Fi без лимита)
 *   на уровне планировщика, плюс battery-not-low; пороги ДОДа («батарея
 *   ≥70%», термал) WorkManager выразить нельзя — проверяются внутри.
 * - Батарея читается из sticky-интента ACTION_BATTERY_CHANGED: путь
 *   BATTERY_PROPERTY_CAPACITY игнорирует сервисную подмену (эмулятор),
 *   а интент — ровно то, что видит система.
 * - Воркер может стартовать в свежем процессе (без UI): `tick` поднимает
 *   ядро сам по запомненному каталогу KV.
 * - Вклад ЛОКАЛЬНЫЙ (эры роя на своём материале); сетевой вклад —
 *   [NetWorker] (шаг 3). Воркеры ходят в том же процессе приложения
 *   (WorkManager по умолчанию) — append-only KV пишется из одного места.
 */
class TickWorker(ctx: Context, params: WorkerParameters) : CoroutineWorker(ctx, params) {

    override suspend fun doWork(): Result {
        val (ok, line) = Gates.check(applicationContext)
        log(line)
        if (!ok) return Result.success()

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

    companion object {
        const val TAG = "dinet-bg"
        const val PERIODIC = "dinet-bg-periodic"
        const val ONCE = "dinet-bg-once"

        /** Гейты планировщика: unmetered Wi-Fi + не низкая батарея. */
        fun gates(): Constraints = Constraints.Builder()
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
