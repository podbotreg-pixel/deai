package dev.dinet.app

import android.content.Intent
import android.net.Uri
import android.os.Bundle
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.statusBarsPadding
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.items
import androidx.compose.foundation.lazy.rememberLazyListState
import androidx.compose.material3.Button
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedTextField
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.setValue
import androidx.compose.ui.Modifier
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import java.io.File
import java.util.concurrent.Executors

/** Ход ленты: кто сказал и что. */
data class Msg(val who: String, val text: String)

/**
 * Чат DI-NET: Compose-UI поверх Rust-ядра через рукописный JNI-мост.
 *
 * ВСЕ JNI-вызовы — на фоновом исполнителе (один поток: Rust-ядро и так
 * под Mutex): эры и сеть идут секунды, UI-поток не блокируется; итоги
 * возвращаются runOnUiThread'ом. Кнопки «Учить» нет — знание течёт
 * через ходы чата (автопары микроязыка валидны); учительские пары —
 * deep-link `dinet://cmd/teach?p=…&a=…` (тесты/скрипты).
 *
 * Скриптовое управление (adb, без клавиатуры — для живых проверок):
 *   am start -a android.intent.action.VIEW -d "dinet://cmd/turn?text=…"  dev.dinet.app
 *   am start -a android.intent.action.VIEW -d "dinet://cmd/teach?p=…&a=…" dev.dinet.app
 */
class MainActivity : ComponentActivity() {
    /** Один фоновый исполнитель под ВСЕ JNI-вызовы (UI не ждёт Rust). */
    private val exec = Executors.newSingleThreadExecutor()
    @Volatile private var opened = false
    val msgs = mutableStateOf(listOf<Msg>())
    val stats = mutableStateOf("нет ядра")
    val input = mutableStateOf("")
    /** Вкладка: 0 — чат, 1 — рой (статистика). */
    val tab = mutableStateOf(0)
    val swarm = mutableStateOf("рой: …")

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContent { DinetApp(this) }
        // ядро открывается в ФОНЕ (KV-файлы <filesDir>/dinet и <filesDir>/dinet.рой)
        val dir = File(filesDir, "dinet").absolutePath
        bg {
            val ok = try {
                Core.open(dir)
            } catch (e: Throwable) {
                false
            }
            runOnUiThread {
                opened = ok
                msgs.value = listOf(
                    if (ok) {
                        Msg(
                            "система",
                            "ядро открыто; знание течёт через ходы чата — ход " +
                                "микроязыком «ай би ви ги» и ответ реплики сами " +
                                "становятся парой; статистика роя — экран «Рой»",
                        )
                    } else {
                        Msg("система", "ядро НЕ открылось — смотреть logcat (страж JNI или KV)")
                    },
                )
                // M3 шаги 2+3: периодический вклад (WorkManager, 6 ч; гейты
                // батарея≥70% / Wi-Fi unmetered / термал<44°C — внутри воркеров):
                // локальный по новому материалу + сетевой (dial роя, если адреса есть)
                if (ok) {
                    TickWorker.schedulePeriodic(this)
                    NetWorker.schedulePeriodic(this)
                }
                refreshStats()
                handleUri(intent?.data)
            }
        }
    }

    override fun onStart() {
        super.onStart()
        // вернулись в форграунд после фона — счётчики могли уехать
        refreshStats()
    }

    override fun onNewIntent(intent: Intent) {
        super.onNewIntent(intent)
        handleUri(intent.data)
    }

    /** Работа в фоне: один исполнитель на все JNI (порядок сохраняется). */
    private fun bg(f: () -> Unit) {
        exec.execute(f)
    }

    private fun handleUri(uri: Uri?) {
        if (uri == null) return
        when (uri.pathSegments.firstOrNull()) {
            "bg" -> {
                // «вклад сейчас»: воркер с теми же гейтами (жёсткая проверка
                // сети-гейтов планировщика всё равно случится перед запуском)
                if (opened) {
                    TickWorker.scheduleOnce(this)
                    push(Msg("система", "фоновый вклад запрошен; лог — logcat (dinet-bg)"))
                }
                return
            }
            "net" -> {
                // «сетевой вклад сейчас»: dial роя, серия эр до чек-пойнта
                if (opened) {
                    NetWorker.scheduleOnce(this)
                    push(Msg("система", "сетевой вклад запрошен; лог — logcat (dinet-net)"))
                }
                return
            }
            "peer" -> {
                // адрес роя: ip:порт (смотрителя) — в рой-KV «рой:peers»
                val addr = uri.getQueryParameter("addr").orEmpty()
                if (addr.isEmpty()) {
                    push(Msg("система", "нет addr= — формат dinet://cmd/peer?addr=ip:порт"))
                    return
                }
                bg {
                    val ok = try {
                        Core.peer(addr)
                    } catch (e: Throwable) {
                        false
                    }
                    push(
                        Msg(
                            "система",
                            if (ok) "адрес роя записан: $addr; сетевой вклад — dinet://cmd/net"
                            else "адрес НЕ записан: $addr (нужен формат ip:порт)",
                        ),
                    )
                }
                return
            }
        }
        when (uri.pathSegments.firstOrNull()) {
            "turn" -> doTurn(uri.getQueryParameter("text").orEmpty())
            "teach" -> doTeach(
                uri.getQueryParameter("p").orEmpty(),
                uri.getQueryParameter("a").orEmpty(),
            )
        }
    }

    private fun push(m: Msg) {
        runOnUiThread { msgs.value = msgs.value + m }
    }

    private fun refreshStats() {
        bg {
            val s = try {
                Core.stats()
            } catch (e: Throwable) {
                "?"
            }
            runOnUiThread { stats.value = s }
        }
    }

    fun doTurn(prompt: String) {
        val p = prompt.trim()
        if (p.isEmpty()) return
        push(Msg("ты", p))
        bg {
            val reply = try {
                Core.turn(p)
            } catch (e: Throwable) {
                "ошибка: ${e.message}"
            }
            push(Msg("рой", reply))
            refreshStats()
        }
    }

    /// Учительская пара — только deep-link (`dinet://cmd/teach?p=…&a=…`):
    /// кнопки «Учить» в чате больше нет.
    fun doTeach(prompt: String, answerPair: String) {
        val p = prompt.trim()
        val a = answerPair.trim()
        if (p.isEmpty() || a.isEmpty()) return
        push(Msg("учитель", "$p → $a"))
        bg {
            val note = try {
                Core.teach(p, a)
            } catch (e: Throwable) {
                "ошибка: ${e.message}"
            }
            push(Msg("система", note))
            refreshStats()
        }
    }

    /// Открыть вклад (0 — чат, 1 — рой); «Рой» сразу показывает статистику.
    fun openTab(which: Int) {
        tab.value = which
        if (which == 1) loadSwarm()
    }

    /// Статистика роя из рой-KV + статус гейтов вклада (Kotlin) — фон.
    fun loadSwarm() {
        bg {
            val line = try {
                Core.swarmStats()
            } catch (e: Throwable) {
                "ошибка: ${e.message}"
            }
            val (_, gates) = Gates.check(this)
            runOnUiThread { swarm.value = "$line\n$gates" }
        }
    }

    /// «Обновить состав»: лёгкий probe роя (dial + волна состава, без эр)
    /// → свежий состав + статистика — фон; UI не ждёт сеть (~6с).
    fun probe() {
        bg {
            val note = try {
                Core.probe()
            } catch (e: Throwable) {
                "ошибка: ${e.message}"
            }
            val line = try {
                Core.swarmStats()
            } catch (e: Throwable) {
                ""
            }
            val (_, gates) = Gates.check(this)
            runOnUiThread {
                swarm.value = "$note\n$line\n$gates"
                refreshStats()
            }
        }
    }
}

@Composable
fun DinetApp(activity: MainActivity) {
    var input by activity.input
    val msgs by activity.msgs
    val stats by activity.stats
    val swarm by activity.swarm
    val tab by activity.tab
    val list = rememberLazyListState()

    MaterialTheme {
        // API 35 рисует edge-to-edge: без паддинга верхний ряд (переключа-
        // тель Чат/Рой) уезжает ПОД прозрачный статус-бар и тапы съедаются
        Column(Modifier.fillMaxSize().statusBarsPadding().padding(12.dp)) {
            Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                Text(
                    "DI-NET",
                    fontSize = 22.sp,
                    fontWeight = FontWeight.Bold,
                    modifier = Modifier.weight(1f),
                )
                Button(onClick = { activity.openTab(0) }) { Text("Чат") }
                Button(onClick = { activity.openTab(1) }) { Text("Рой") }
            }
            if (tab == 0) {
                Text(stats, fontSize = 12.sp)
                LazyColumn(
                    Modifier.fillMaxWidth().weight(1f).padding(vertical = 8.dp),
                    state = list,
                    verticalArrangement = Arrangement.spacedBy(6.dp),
                ) {
                    items(msgs) { m ->
                        Column {
                            Text(m.who, fontSize = 11.sp, fontWeight = FontWeight.Bold)
                            Text(m.text, fontSize = 15.sp)
                        }
                    }
                }
                Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                    OutlinedTextField(
                        value = input,
                        onValueChange = { input = it },
                        modifier = Modifier.weight(1f),
                        label = { Text("ход или контекст (4 слова)") },
                        singleLine = true,
                    )
                    Button(onClick = { activity.doTurn(input) }) { Text("Спросить") }
                }
            } else {
                Text(
                    swarm,
                    fontSize = 13.sp,
                    modifier = Modifier.weight(1f).padding(vertical = 8.dp),
                )
                Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                    Button(onClick = { activity.probe() }) { Text("Обновить состав") }
                }
            }
        }
    }
}
