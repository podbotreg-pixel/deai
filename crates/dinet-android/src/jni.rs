//! Рукописный JNI-мост (БЕЗ крейта `jni`): JNI-ABI — такой же протокол,
//! как и всё в DI-NET, поэтому пишется руками, std-only.
//! cdylib `libdinet_phone.so` (имя таргета dinet_phone — не «dinet_core»,
//! чтобы не столкнуться с lib таргетом крейта dinet-core).
//!
//! ## Таблица по СПЕЦ-ИНДЕКСАМ
//! `JNIEnv*` — указатель на указатель на таблицу функций `JNINativeInterface_`.
//! Первые 4 слота — reserved, далее фиксированный порядок jni.h. Берём
//! только то, что нужно мосту, по спецификационным индексам:
//! GetVersion=4, FindClass=6, ThrowNew=14, NewStringUTF=167,
//! GetStringUTFLength=168, GetStringUTFChars=169,
//! ReleaseStringUTFChars=170, ExceptionCheck=228.
//!
//! ## Страж ABI (`open`)
//! 1) `GetVersion() == 0x00010006` (JNI 1.6) — если спец-индексы
//!    разъехались, таблица прочитается мусором и версия не сойдётся;
//! 2) self-roundtrip «ок»: `NewStringUTF("ок") → GetStringUTFChars` —
//!    строка обязана совпасть. Несходство = паника → экспорт отдаёт
//!    Java `false` + `RuntimeException` (приложение живо, сбой виден).
//!
//! ## Строки: MUTF-8 ≈ UTF-8 для BMP
//! Кириллица и микроязык — BMP, кодировки совпадают байт-в-байт.
//! Дополнение юнирода (эмодзи) в MUTF-8 кодируется суррогатными парами —
//! читается lossy (U+FFFD); в v1 допустимо, ядро на таких ходах
//! отвечает шаблоном RAG.
//!
//! ## Экспорты (`Java_dev_dinet_app_Core_*`)
//! `open(dir): Bool` · `turn(prompt): String` ·
//! `teach(prompt, answer): String` · `stats(): String` ·
//! `tick(): String` · `peer(addr): Bool` · `net(): String` ·
//! `probe(): String` (волна состава без эр) · `swarmStats(): String`
//! (экран «Рой»).
//! Паника внутри любого экспорта ловится: Java получает `false`/пустую
//! строку + `RuntimeException` с причиной.

use std::ffi::{c_char, c_void, CStr, CString};
use std::panic::{catch_unwind, AssertUnwindSafe};
use std::ptr;
use std::sync::Mutex;

use crate::PhoneCore;

/// Ядро телефона — одно на процесс; открывается `open(dir)`.
static CORE: Mutex<Option<PhoneCore>> = Mutex::new(None);
/// Каталог KV последнего успешного `open` — фон (WorkManager) поднимает
/// ядро через `tick()` без UI: если ядро не открыто, оно открывается
/// отсюда само.
static CORE_DIR: Mutex<Option<String>> = Mutex::new(None);

/// `JNIEnv*`: указатель на указатель на таблицу функций.
type Env = *mut *mut c_void;

// Сигнатуры функций таблицы (abi, как в jni.h; первый аргумент — Env).
type GetVersionFn = unsafe extern "C" fn(Env) -> i32;
type FindClassFn = unsafe extern "C" fn(Env, *const c_char) -> *mut c_void;
type ThrowNewFn = unsafe extern "C" fn(Env, *mut c_void, *const c_char) -> i32;
type NewStringUTFFn = unsafe extern "C" fn(Env, *const c_char) -> *mut c_void;
type GetStringUTFCharsFn = unsafe extern "C" fn(Env, *mut c_void, *mut u8) -> *const c_char;
type ReleaseStringUTFCharsFn = unsafe extern "C" fn(Env, *mut c_void, *const c_char);
type ExceptionCheckFn = unsafe extern "C" fn(Env) -> u8;

// Спец-индексы таблицы (4 reserved-слота, далее порядок jni.h).
const IDX_GET_VERSION: usize = 4;
const IDX_FIND_CLASS: usize = 6;
const IDX_THROW_NEW: usize = 14;
const IDX_NEW_STRING_UTF: usize = 167;
const IDX_GET_STRING_UTF_CHARS: usize = 169;
const IDX_RELEASE_STRING_UTF_CHARS: usize = 170;
const IDX_EXCEPTION_CHECK: usize = 228;

/// JNI_VERSION_1_6 — ожидаемая версия таблицы.
const JNI_VERSION_1_6: i32 = 0x0001_0006;

/// Слот таблицы №idx (Env → таблица → слот).
unsafe fn slot(env: Env, idx: usize) -> *mut c_void {
    let table = *env as *mut *mut c_void;
    *table.add(idx)
}

/// Слот №idx как указатель функции нужной сигнатуры.
unsafe fn func<T: Copy>(env: Env, idx: usize) -> T {
    debug_assert_eq!(std::mem::size_of::<T>(), std::mem::size_of::<*mut c_void>());
    std::mem::transmute_copy(&slot(env, idx))
}

/// Java-строка → Rust (MUTF-8 ≈ UTF-8 для BMP; NUL-терминированная).
unsafe fn read_string(env: Env, js: *mut c_void) -> String {
    if js.is_null() {
        return String::new();
    }
    let get: GetStringUTFCharsFn = func(env, IDX_GET_STRING_UTF_CHARS);
    let release: ReleaseStringUTFCharsFn = func(env, IDX_RELEASE_STRING_UTF_CHARS);
    let chars = get(env, js, ptr::null_mut());
    if chars.is_null() {
        return String::new();
    }
    let s = CStr::from_ptr(chars).to_string_lossy().into_owned();
    release(env, js, chars);
    s
}

/// Rust → Java-строка (NewStringUTF); при висящем исключении — null.
/// ВАЖНО: вызывать NewStringUTF при уже висящем исключении ЗАПРЕЩЕНО —
/// ART убивает процесс (`JNI DETECTED ERROR: NewStringUTF called with
/// pending exception`). `safe()` вешает RuntimeException на ошибках ядра,
/// поэтому эта проверка обязательна на входе.
unsafe fn write_string(env: Env, s: &str) -> *mut c_void {
    let check: ExceptionCheckFn = func(env, IDX_EXCEPTION_CHECK);
    if check(env) != 0 {
        return ptr::null_mut();
    }
    let new_string: NewStringUTFFn = func(env, IDX_NEW_STRING_UTF);
    let c = CString::new(s).unwrap_or_default();
    let js = new_string(env, c.as_ptr());
    if check(env) != 0 {
        // исключение уже на Java-стороне — она его увидит после возврата
        return ptr::null_mut();
    }
    js
}

/// Повесить RuntimeException с причиной на Java-стороне.
unsafe fn throw(env: Env, msg: &str) {
    let find: FindClassFn = func(env, IDX_FIND_CLASS);
    let throw_new: ThrowNewFn = func(env, IDX_THROW_NEW);
    let class = find(env, c"java/lang/RuntimeException".as_ptr());
    if class.is_null() {
        return;
    }
    let c = CString::new(msg).unwrap_or_default();
    let _ = throw_new(env, class, c.as_ptr());
}

/// Страж ABI: версия таблицы = JNI 1.6 И self-roundtrip «ок».
/// Паникует при несходстве — экспорт `open` ловит и отдаёт Java причину.
unsafe fn guard(env: Env) {
    let version: GetVersionFn = func(env, IDX_GET_VERSION);
    assert_eq!(
        version(env),
        JNI_VERSION_1_6,
        "таблица JNI не читается: GetVersion != 0x00010006 — спец-индексы разъехались"
    );
    let round = write_string(env, "ок");
    assert!(!round.is_null(), "NewStringUTF вернул null — ABI не тот");
    assert_eq!(
        read_string(env, round),
        "ок",
        "self-roundtrip «ок» не совпал — NewStringUTF/GetStringUTFChars не на своих индексах"
    );
}

fn core<T>(f: impl FnOnce(&mut PhoneCore) -> Result<T, String>, err: &str) -> Result<T, String> {
    match CORE.lock() {
        Ok(mut g) => match g.as_mut() {
            Some(c) => f(c),
            None => Err(format!("{err}: ядро не открыто (сначала open)")),
        },
        Err(_) => Err(format!("{err}: блокировка ядра отравлена")),
    }
}

/// Фон без UI: поднять ядро по запомненному каталогу, если оно не открыто.
fn ensure_open() -> Result<(), String> {
    let empty = CORE
        .lock()
        .map_err(|_| "блокировка ядра отравлена".to_string())?
        .is_none();
    if !empty {
        return Ok(());
    }
    let dir = CORE_DIR
        .lock()
        .map_err(|_| "блокировка каталога отравлена".to_string())?
        .clone()
        .ok_or_else(|| "tick: каталог ядра неизвестен (сначала open)".to_string())?;
    let c = PhoneCore::open(std::path::Path::new(&dir))
        .map_err(|e| format!("tick: KV не открылся: {e}"))?;
    *CORE
        .lock()
        .map_err(|_| "блокировка ядра отравлена".to_string())? = Some(c);
    Ok(())
}

/// Обвязка экспорта: паника Rust-ядра ловится → Java получает
/// `false`/пустую строку + RuntimeException с причиной.
fn safe<T>(env: Env, tag: &str, fallback: T, f: impl FnOnce() -> Result<T, String>) -> T {
    match catch_unwind(AssertUnwindSafe(f)) {
        Ok(Ok(v)) => v,
        Ok(Err(msg)) => unsafe {
            throw(env, &format!("{tag}: {msg}"));
            fallback
        },
        Err(_) => unsafe {
            throw(env, &format!("{tag}: паника Rust-ядра (страж JNI не прошёл?)"));
            fallback
        },
    }
}

#[no_mangle]
pub extern "C" fn Java_dev_dinet_app_Core_open(
    env: Env,
    _class: *mut c_void,
    jdir: *mut c_void,
) -> u8 {
    safe(env, "open", false, || {
        unsafe { guard(env) };
        let dir = unsafe { read_string(env, jdir) };
        if dir.trim().is_empty() {
            return Err("пустой путь каталога".into());
        }
        match PhoneCore::open(std::path::Path::new(&dir)) {
            Ok(c) => {
                *CORE
                    .lock()
                    .map_err(|_| "блокировка ядра отравлена".to_string())? = Some(c);
                if let Ok(mut d) = CORE_DIR.lock() {
                    *d = Some(dir);
                }
                Ok(true)
            }
            Err(e) => Err(format!("KV не открылся: {e}")),
        }
    }) as u8
}

#[no_mangle]
pub extern "C" fn Java_dev_dinet_app_Core_turn(
    env: Env,
    _class: *mut c_void,
    jprompt: *mut c_void,
) -> *mut c_void {
    let answer = safe(env, "turn", String::new(), || {
        let prompt = unsafe { read_string(env, jprompt) };
        if prompt.trim().is_empty() {
            return Err("пустой ход".into());
        }
        core(|c| c.turn(&prompt).map_err(|e| e.to_string()), "turn")
    });
    unsafe { write_string(env, &answer) }
}

#[no_mangle]
pub extern "C" fn Java_dev_dinet_app_Core_teach(
    env: Env,
    _class: *mut c_void,
    jp: *mut c_void,
    ja: *mut c_void,
) -> *mut c_void {
    let note = safe(env, "teach", String::new(), || {
        let p = unsafe { read_string(env, jp) };
        let a = unsafe { read_string(env, ja) };
        core(|c| c.teach(&p, &a), "teach")
    });
    unsafe { write_string(env, &note) }
}

#[no_mangle]
pub extern "C" fn Java_dev_dinet_app_Core_stats(env: Env, _class: *mut c_void) -> *mut c_void {
    let stats = safe(env, "stats", String::new(), || core(|c| Ok(c.stats()), "stats"));
    unsafe { write_string(env, &stats) }
}

/// Фоновый вклад (WorkManager, шаг 2): эра роя по новому материалу.
/// Ядро поднимает себя сам (фоновый процесс без UI); строка-итог для логов.
#[no_mangle]
pub extern "C" fn Java_dev_dinet_app_Core_tick(env: Env, _class: *mut c_void) -> *mut c_void {
    let note = safe(env, "tick", String::new(), || {
        ensure_open()?;
        core(|c| Ok(c.bg_tick()), "tick")
            .map(|o| o.unwrap_or_else(|| "нет нового материала".into()))
    });
    unsafe { write_string(env, &note) }
}

/// Запомнить адрес роя (deep-link `dinet://cmd/peer?addr=ip:порт`).
/// 0 — адрес не парсится (GUI покажет; ядро живо).
#[no_mangle]
pub extern "C" fn Java_dev_dinet_app_Core_peer(
    env: Env,
    _class: *mut c_void,
    jaddr: *mut c_void,
) -> u8 {
    safe(env, "peer", false, || {
        let addr = unsafe { read_string(env, jaddr) };
        core(|c| c.add_peer(&addr).map(|_| true), "peer")
    }) as u8
}

/// Адреса роя одной строкой (по одному на строку) — экран «Сеть».
#[no_mangle]
pub extern "C" fn Java_dev_dinet_app_Core_peers(env: Env, _class: *mut c_void) -> *mut c_void {
    let list = safe(env, "peers", String::new(), || {
        core(|c| Ok(c.peers().join("\n")), "peers")
    });
    unsafe { write_string(env, &list) }
}

/// Сетевой вклад (WorkManager, шаг 3): dial роя («рой:peers»), волна
/// состава, серия эр роя на реальном TCP до чек-пойнта REVEAL (≤4 эры).
/// Ядро поднимает себя сам; строка-итог для логов.
#[no_mangle]
pub extern "C" fn Java_dev_dinet_app_Core_net(env: Env, _class: *mut c_void) -> *mut c_void {
    let note = safe(env, "net", String::new(), || {
        ensure_open()?;
        core(
            |c| {
                Ok(match c.net_tick(4) {
                    Ok(n) => n,
                    Err(e) => format!("не вышло: {e}"),
                })
            },
            "net",
        )
    });
    unsafe { write_string(env, &note) }
}

/// Лёгкий probe роя (экран «Рой», «Обновить состав»): dial адресов,
/// волна состава до тишины — без эр. Состав персистит (swarmStats).
#[no_mangle]
pub extern "C" fn Java_dev_dinet_app_Core_probe(env: Env, _class: *mut c_void) -> *mut c_void {
    let note = safe(env, "probe", String::new(), || {
        ensure_open()?;
        core(|c| c.net_probe(), "probe")
    });
    unsafe { write_string(env, &note) }
}

/// Статистика роя из рой-KV (экран «Рой»): адреса, последний состав,
/// «видели N мин назад», вклад (эры/чек-пойнты/прокси/записи сети).
#[no_mangle]
pub extern "C" fn Java_dev_dinet_app_Core_swarmStats(
    env: Env,
    _class: *mut c_void,
) -> *mut c_void {
    let s = safe(env, "swarmStats", String::new(), || {
        ensure_open()?;
        core(|c| Ok(c.swarm_stats()), "swarmStats")
    });
    unsafe { write_string(env, &s) }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn core_guard_fails_without_open() {
        // вызовы до open — честная ошибка, без паники
        assert!(core(|_| Ok(()), "turn").is_err());
    }
}
