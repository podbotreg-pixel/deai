package dev.dinet.app

import android.content.Context
import android.util.Log
import androidx.work.CoroutineWorker
import androidx.work.ExistingPeriodicWorkPolicy
import androidx.work.ExistingWorkPolicy
import androidx.work.OneTimeWorkRequestBuilder
import androidx.work.PeriodicWorkRequestBuilder
import androidx.work.WorkManager
import androidx.work.WorkerParameters
import java.io.File
import java.time.Duration

/**
 * СЕТЕВОЙ вклад телефона (M3 шаг 3): WorkManager воркер проходит гейты
 * (см. [Gates]) и гонит СЕТЬ signature участия — dial роя из
 * «рой:peers» (адреса ставит deep-link `dinet://cmd/peer?addr=ip:порт`),
 * волна состава, серия эр роя на реальном TCP+gossipsub до чек-пойнта
 * REVEAL (≤4 эры), чек-пойнт применяется к реплике чата.
 *
 * - Смотритель роя (`cargo run -p dinet-p2p --bin swarmkeeper`) держит
 *   порты и перечитывает состав перед каждой эрой — телефон входит в
 *   любой момент и выходит после чек-пойнта (сессия ≤ ~40с).
 * - Роя нет/адреса не отвечают → честный отказ в логе; локальный
 *   материал тем временем тянет [TickWorker].
 * - Воркер ходит в том же процессе приложения: append-only KV и
 *   реестр сетевого вклада пишутся из одного места.
 */
class NetWorker(ctx: Context, params: WorkerParameters) : CoroutineWorker(ctx, params) {

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
            Core.net()
        } catch (e: Throwable) {
            "ошибка: ${e.message}"
        }
        log("сетевой вклад: $note")
        return Result.success()
    }

    private fun log(msg: String) {
        Log.i(TAG, msg)
    }

    companion object {
        const val TAG = "dinet-net"
        const val PERIODIC = "dinet-net-periodic"
        const val ONCE = "dinet-net-once"

        /// Периодический сетевой вклад (6 часов, KEEP).
        fun schedulePeriodic(context: Context) {
            WorkManager.getInstance(context).enqueueUniquePeriodicWork(
                PERIODIC,
                ExistingPeriodicWorkPolicy.KEEP,
                PeriodicWorkRequestBuilder<NetWorker>(Duration.ofHours(6))
                    .setConstraints(TickWorker.gates())
                    .build(),
            )
        }

        /// «Сетевой вклад сейчас» (deep-link dinet://cmd/net).
        fun scheduleOnce(context: Context) {
            WorkManager.getInstance(context).enqueueUniqueWork(
                ONCE,
                ExistingWorkPolicy.KEEP,
                OneTimeWorkRequestBuilder<NetWorker>()
                    .setConstraints(TickWorker.gates())
                    .build(),
            )
        }
    }
}
