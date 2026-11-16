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

/** Ход ленты: кто сказал и что. */
data class Msg(val who: String, val text: String)

/**
 * Чат DI-NET: Compose-UI поверх Rust-ядра через рукописный JNI-мост.
 *
 * Скриптовое управление (adb, без клавиатуры — для живых проверок):
 *   am start -a android.intent.action.VIEW -d "dinet://cmd/turn?text=…"  dev.dinet.app
 *   am start -a android.intent.action.VIEW -d "dinet://cmd/teach?p=…&a=…" dev.dinet.app
 */
class MainActivity : ComponentActivity() {
    private var opened = false
    val msgs = mutableStateOf(listOf<Msg>())
    val stats = mutableStateOf("нет ядра")
    val input = mutableStateOf("")
    val answer = mutableStateOf("")

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        // ядро открывается ДО UI: KV-файлы <filesDir>/dinet и <filesDir>/dinet.рой
        val dir = File(filesDir, "dinet").absolutePath
        opened = try {
            Core.open(dir)
        } catch (e: Throwable) {
            false
        }
        msgs.value = listOf(
            if (opened) {
                Msg(
                    "система",
                    "ядро открыто; учи парами «4 слова → 1 слово» (ай би ви ги ди ей жи зи)",
                )
            } else {
                Msg("система", "ядро НЕ открылось — смотреть logcat (страж JNI или KV)")
            }
        )
        refreshStats()
        // M3 шаги 2+3: периодический вклад (WorkManager, 6 ч; гейты
        // батарея≥70% / Wi-Fi unmetered / термал<44°C — внутри воркеров):
        // локальный по новому материалу + сетевой (dial роя, если адреса есть)
        if (opened) {
            TickWorker.schedulePeriodic(this)
            NetWorker.schedulePeriodic(this)
        }
        handleUri(intent?.data)
        setContent { DinetApp(this) }
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
                val ok = if (opened) Core.peer(addr) else false
                push(
                    Msg(
                        "система",
                        if (ok) "адрес роя записан: $addr; сетевой вклад — dinet://cmd/net"
                        else "адрес НЕ записан: $addr (нужен формат ip:порт)",
                    ),
                )
                return
            }
        }
        if (!opened) return
        when (uri.pathSegments.firstOrNull()) {
            "turn" -> doTurn(uri.getQueryParameter("text").orEmpty())
            "teach" -> doTeach(
                uri.getQueryParameter("p").orEmpty(),
                uri.getQueryParameter("a").orEmpty(),
            )
        }
    }

    private fun push(m: Msg) {
        msgs.value = msgs.value + m
    }

    private fun refreshStats() {
        stats.value = try {
            Core.stats()
        } catch (e: Throwable) {
            "?"
        }
    }

    fun doTurn(prompt: String) {
        val p = prompt.trim()
        if (!opened || p.isEmpty()) return
        push(Msg("ты", p))
        val reply = try {
            Core.turn(p)
        } catch (e: Throwable) {
            "ошибка: ${e.message}"
        }
        push(Msg("рой", reply))
        refreshStats()
    }

    fun doTeach(prompt: String, answerPair: String) {
        val p = prompt.trim()
        val a = answerPair.trim()
        if (!opened || p.isEmpty() || a.isEmpty()) return
        push(Msg("учитель", "$p → $a"))
        val note = try {
            Core.teach(p, a)
        } catch (e: Throwable) {
            "ошибка: ${e.message}"
        }
        push(Msg("система", note))
        refreshStats()
    }
}

@Composable
fun DinetApp(activity: MainActivity) {
    var input by activity.input
    var answer by activity.answer
    val msgs by activity.msgs
    val stats by activity.stats
    val list = rememberLazyListState()

    MaterialTheme {
        Column(Modifier.fillMaxSize().padding(12.dp)) {
            Text("DI-NET", fontSize = 22.sp, fontWeight = FontWeight.Bold)
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
            Row(
                Modifier.padding(top = 8.dp),
                horizontalArrangement = Arrangement.spacedBy(8.dp),
            ) {
                OutlinedTextField(
                    value = answer,
                    onValueChange = { answer = it },
                    modifier = Modifier.weight(1f),
                    label = { Text("ответ учителя (1 слово)") },
                    singleLine = true,
                )
                Button(onClick = { activity.doTeach(input, answer) }) { Text("Учить") }
            }
        }
    }
}
