package dev.dinet.app

/**
 * Мост в Rust-ядро (крейт dinet-android, libdinet_phone.so).
 * JNI-ABI рукописный: таблица функций берётся по спец-индексам,
 * страж ABI проверяется при open() — при несходстве Java получает
 * false/пустую строку + RuntimeException.
 */
object Core {
    init {
        System.loadLibrary("dinet_phone")
    }

    /** Открыть/создать ядро в каталоге данных приложения. */
    external fun open(dir: String): Boolean

    /** Ход чата: ответ реплики роя (микроязык) или шаблон RAG. */
    external fun turn(prompt: String): String

    /** Учительская пара: 4 слова контекста + 1 слово-ответ из LEX. */
    external fun teach(prompt: String, answer: String): String

    /** Счётчики одной строкой: модель/ходы/пары/память/эры/чек-пойнты. */
    external fun stats(): String

    /** Фоновый вклад (WorkManager): эра роя по новому материалу; ядро
     *  поднимает себя сам (воркер стартует без UI). */
    external fun tick(): String

    /** Запомнить адрес роя (deep-link dinet://cmd/peer?addr=ip:порт). */
    external fun peer(addr: String): Boolean

    /** Сетевой вклад (WorkManager): dial роя, серия эр на реальном TCP
     *  до чек-пойнта REVEAL; ядро поднимает себя сам. */
    external fun net(): String

    /** Лёгкий probe роя (экран «Рой», «Обновить состав»): dial адресов,
     *  волна состава до тишины — без эр; состав персистит. */
    external fun probe(): String

    /** Статистика роя (экран «Рой»): адреса, последний состав,
     *  «видели N мин назад», вклад (эры/чек-пойнты/прокси/записи сети). */
    external fun swarmStats(): String
}
