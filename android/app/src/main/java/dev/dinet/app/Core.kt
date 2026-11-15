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
}
