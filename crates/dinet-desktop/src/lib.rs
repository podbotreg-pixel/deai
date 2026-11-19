//! C-ABI мост настольного Qt-клиента DI-NET (M0/M4 GUI): те же функции,
//! что JNI у телефона, но для `ctypes`/Qt — открыть ядро, ход, пара,
//! статистика, роевые эры, probe, сеть. Все строки — UTF-8 CString,
//! вызывающий ОБЯЗАН освобождать их через `dinet_free`.
//!
//! Правило GUI проекта: Linux/Windows — Qt (здесь PyQt6 поверх этого
//! моста); Android — Compose; webchat остаётся lightweight-слоем.
//!
//! Явная индексация/длинные API-функции — намеренный стиль C-моста.
#![allow(clippy::needless_range_loop, clippy::too_many_arguments)]
// C-ABI: все extern-функции принимают сырые указатели по контракту ABI —
// вызывающий (Qt/ctypes) обязан передавать валидные C-строки.
#![allow(clippy::not_unsafe_ptr_arg_deref)]

use dinet_phone::PhoneCore;
use std::ffi::{c_char, CStr, CString};
use std::panic::{catch_unwind, AssertUnwindSafe};
use std::path::Path;
use std::sync::Mutex;

/// Ядро процесса (один экземпляр, как у JNI-моста телефона).
static CORE: Mutex<Option<PhoneCore>> = Mutex::new(None);
/// Каталог KV — для повторного открытия из фонового тика.
static CORE_DIR: Mutex<Option<String>> = Mutex::new(None);

/// Безопасный указатель → Rust String (невалидный → пусто).
unsafe fn read_str(p: *const c_char) -> String {
    if p.is_null() {
        return String::new();
    }
    CStr::from_ptr(p).to_string_lossy().into_owned()
}

/// Rust String → валидный C-указатель (NULL при внутреннем NUL).
fn to_c(s: String) -> *mut c_char {
    match CString::new(s) {
        Ok(c) => c.into_raw(),
        Err(_) => std::ptr::null_mut(),
    }
}

/// Общий вызов: паника Rust НЕ пересекает FFI-границу.
fn with_core(f: impl FnOnce(&mut PhoneCore) -> Result<String, String>) -> *mut c_char {
    let out = catch_unwind(AssertUnwindSafe(|| {
        let mut guard = match CORE.lock() {
            Ok(g) => g,
            Err(_) => return Err("блокировка ядра отравлена".into()),
        };
        match guard.as_mut() {
            Some(core) => f(core),
            None => Err("ядро не открыто (dinet_open)".into()),
        }
    }));
    to_c(match out {
        Ok(Ok(s)) => s,
        Ok(Err(e)) => format!("ошибка: {e}"),
        Err(_) => "ошибка: паника Rust-ядра".into(),
    })
}

/// Открыть/создать ядро в каталоге `dir` (KV: `<dir>`, `<dir>.рой`).
/// 1 — успех, 0 — ошибка.
#[no_mangle]
pub extern "C" fn dinet_open(dir: *const c_char) -> i32 {
    let dir = unsafe { read_str(dir) };
    let ok = catch_unwind(AssertUnwindSafe(|| {
        if dir.trim().is_empty() {
            return false;
        }
        match PhoneCore::open(Path::new(&dir)) {
            Ok(core) => {
                *CORE.lock().unwrap() = Some(core);
                *CORE_DIR.lock().unwrap() = Some(dir);
                true
            }
            Err(_) => false,
        }
    }))
    .unwrap_or(false);
    ok as i32
}

/// Ход чата → ответ реплики (RAG + toy-модель).
#[no_mangle]
pub extern "C" fn dinet_turn(prompt: *const c_char) -> *mut c_char {
    let prompt = unsafe { read_str(prompt) };
    with_core(|c| c.turn(&prompt).map_err(|e| e.to_string()))
}

/// Учительская пара; каждые 4 валидных пар — локальная эра роя.
#[no_mangle]
pub extern "C" fn dinet_teach(p: *const c_char, a: *const c_char) -> *mut c_char {
    let (p, a) = unsafe { (read_str(p), read_str(a)) };
    with_core(|c| c.teach(&p, &a))
}

/// Счётчики чата одной строкой.
#[no_mangle]
pub extern "C" fn dinet_stats() -> *mut c_char {
    with_core(|c| Ok(c.stats()))
}

/// Статистика экрана «Рой»: адреса, состав, эры, прокси, знак, гейты.
#[no_mangle]
pub extern "C" fn dinet_swarm_stats() -> *mut c_char {
    with_core(|c| Ok(c.swarm_stats()))
}

/// Запомнить адрес смотрителя `ip:порт`. 1 — OK.
#[no_mangle]
pub extern "C" fn dinet_add_peer(addr: *const c_char) -> i32 {
    let addr = unsafe { read_str(addr) };
    let ok = catch_unwind(AssertUnwindSafe(|| {
        match CORE.lock() {
            Ok(mut g) => match g.as_mut() {
                Some(c) => c.add_peer(&addr).is_ok(),
                None => false,
            },
            Err(_) => false,
        }
    }))
    .unwrap_or(false);
    ok as i32
}

/// Адреса роя, по одному на строку.
#[no_mangle]
pub extern "C" fn dinet_peers() -> *mut c_char {
    with_core(|c| Ok(c.peers().join("\n")))
}

/// Лёгкий probe роя (волна состава без эр).
#[no_mangle]
pub extern "C" fn dinet_probe() -> *mut c_char {
    with_core(|c| c.net_probe())
}

/// Фоновый вклад: эра по новому материалу.
#[no_mangle]
pub extern "C" fn dinet_bg() -> *mut c_char {
    with_core(|c| Ok(c.bg_tick().unwrap_or_else(|| "нет нового материала".into())))
}

/// Сетевой вклад: до `max_eras` эр поверх роя до чек-пойнта.
#[no_mangle]
pub extern "C" fn dinet_net(max_eras: u32) -> *mut c_char {
    with_core(|c| c.net_tick(max_eras.max(1)))
}

/// camoMark-знак происхождения: 1 — подтверждён.
#[no_mangle]
pub extern "C" fn dinet_watermark_ok() -> i32 {
    let ok = catch_unwind(AssertUnwindSafe(|| {
        CORE.lock()
            .ok()
            .and_then(|g| g.as_ref().map(|c| c.watermark_ok()))
            .unwrap_or(false)
    }))
    .unwrap_or(false);
    ok as i32
}

/// Освободить строку, возвращённую любым `dinet_*`-вызовом.
///
/// # Safety
/// Указатель должен быть получен из `dinet_*`-функции и ещё не освобождён.
#[no_mangle]
pub unsafe extern "C" fn dinet_free(p: *mut c_char) {
    if !p.is_null() {
        drop(unsafe { CString::from_raw(p) });
    }
}

/// Служебный указатель для Qt (не используется вызывающим).
#[no_mangle]
pub extern "C" fn dinet_abi_version() -> u32 {
    1
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Один тест на всё: глобальное ядро процесса, параллельные тесты
    /// конфликтовали бы за него.
    #[test]
    fn c_abi_roundtrip() {
        let dir = std::env::temp_dir().join(format!("dinet-desktop-{}", std::process::id()));
        let _ = std::fs::remove_file(&dir);
        let _ = std::fs::remove_file(dir.with_extension("рой"));
        let cdir = CString::new(dir.to_str().unwrap()).unwrap();
        assert_eq!(dinet_open(cdir.as_ptr()), 1);
        assert_eq!(dinet_abi_version(), 1);
        let stats = dinet_stats();
        assert!(!stats.is_null());
        let s = unsafe { CStr::from_ptr(stats).to_string_lossy().into_owned() };
        assert!(s.contains("модель:"), "stats: {s}");
        unsafe { dinet_free(stats) };

        let p = CString::new("ай би ви ги").unwrap();
        let a = CString::new("ди").unwrap();
        let note = dinet_teach(p.as_ptr(), a.as_ptr());
        assert!(!note.is_null());
        unsafe { dinet_free(note) };
        let turn = dinet_turn(p.as_ptr());
        assert!(!turn.is_null());
        unsafe { dinet_free(turn) };
        let swarm = dinet_swarm_stats();
        assert!(!swarm.is_null());
        unsafe { dinet_free(swarm) };
        let peers = dinet_peers();
        assert!(!peers.is_null());
        unsafe { dinet_free(peers) };
        let bad = CString::new("not-an-address").unwrap();
        assert_eq!(dinet_add_peer(bad.as_ptr()), 0);
    }
}
