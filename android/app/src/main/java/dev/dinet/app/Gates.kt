package dev.dinet.app

import android.content.Context
import android.content.Intent
import android.content.IntentFilter
import android.net.ConnectivityManager
import android.net.NetworkCapabilities
import java.io.File

/**
 * Гейты вклада (DoD M3): батарея ≥70%, термал <44°C, Wi-Fi unmetered.
 * Планировщик WorkManager держит UNMETERED + battery-not-low; пороги
 * ДОДа проверяются здесь (внутри воркера) — общие для локального
 * (TickWorker) и сетевого (NetWorker) вкладов.
 */
object Gates {

    /** Порог остановки по заряду. */
    const val MIN_LEVEL = 70
    /** Порог остановки: 44°C (милли°C) — телефон горячий, отдаём прохладу. */
    const val HOT_MILLICELSIUS = 44_000

    /** Одна строка о гейтах + вердикт: true — вклад можно гнать. */
    fun check(ctx: Context): Pair<Boolean, String> {
        val level = batteryLevel(ctx)
        val temp = thermalMax()
        val wifi = wifiUnmetered(ctx)
        val line = "гейты: батарея=${level ?: "не читается"}%, " +
            "термал=${temp?.let { "${it / 1000.0}°C" } ?: "нет зон"}, " +
            "wifi-unmetered=$wifi"
        // батарея: ≥70%; нечитаемая (эмулятор без батареи) — гейт не держим
        if (level != null && level < MIN_LEVEL) {
            return false to "$line; батарея $level% < $MIN_LEVEL% — вклад пропущен"
        }
        if (temp != null && temp >= HOT_MILLICELSIUS) {
            return false to "$line; термал ${temp / 1000.0}°C ≥ 44.0°C — вклад пропущен"
        }
        if (!wifi) {
            return false to "$line; сеть не Wi-Fi-unmetered — вклад пропущен"
        }
        return true to "$line; вклад разрешён"
    }

    /// Заряд батареи в % из sticky-интента (то, что видит система);
    /// null — батарея не симулируется. Путь BATTERY_PROPERTY_CAPACITY
    /// игнорирует сервисную подмену (`dumpsys battery set`) — ловушка
    /// эмулятора, интент честнее.
    private fun batteryLevel(ctx: Context): Int? {
        val intent = try {
            ctx.registerReceiver(null, IntentFilter(Intent.ACTION_BATTERY_CHANGED))
        } catch (e: Throwable) {
            null
        } ?: return null
        val lvl = intent.getIntExtra(android.os.BatteryManager.EXTRA_LEVEL, -1)
        val scale = intent.getIntExtra(android.os.BatteryManager.EXTRA_SCALE, -1)
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
    private fun wifiUnmetered(ctx: Context): Boolean {
        val cm = ctx.getSystemService(ConnectivityManager::class.java) ?: return false
        val net = cm.activeNetwork ?: return false
        val caps = cm.getNetworkCapabilities(net) ?: return false
        return caps.hasTransport(NetworkCapabilities.TRANSPORT_WIFI) &&
            caps.hasCapability(NetworkCapabilities.NET_CAPABILITY_NOT_METERED)
    }
}
