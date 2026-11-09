// ---------------------------------------------------------------------------
// PROBE-2 (train-cpp v1): raw-FFI путь для ВНУТРИСЕТЕВОГО LoRA.
//
// llama.cpp — forward-only: градиенты сквозь граф не идут, а имена узлов
// графа в этой версии не проставляются. Здесь мы подключаем eval-колбэк
// (llama_context_params::cb_eval) и наблюдаем узлы по СТРУКТУРЕ:
//   • голова   — mul_mat с выходом [n_vocab × n_tok]; её вход t_embd
//                (src[1]) — скрытое состояние ПОСЛЕ финальной нормы, а
//                t_embd->src[0] — RMS-норма, её src[0] — h_pre (до нормы);
//   • ffn_down — mul_mat, чей вес (src[0]) называется "blk.N.ffn_down"
//                и N — последний слой; его вход (src[1]) — выход swiGLU.
//
// Тяжёлый forward остаётся llama.cpp; тренажёр (dinet-train) учит дельту
// по короткому хвосту графа (резидуал → финальная норма → голова).
// ---------------------------------------------------------------------------

#![allow(unknown_lints)]
#![allow(clippy::missing_safety_doc)]
#![allow(clippy::cast_sign_loss)]

use llama_cpp_sys_2 as sys;
use std::ffi::{CStr, CString, c_void};
use std::os::raw::{c_char, c_int};
use std::path::Path;

/// Материал одного raw-прохода для шага layer-LoRA.
#[derive(Debug, Clone)]
pub struct L2Probe {
    pub n_tokens: usize,
    pub n_embd: usize,
    pub n_vocab: usize,
    /// Выход swiGLU (вход ffn_down последнего слоя): n_tok строк по n_ff.
    pub x_down: Option<Vec<Vec<f32>>>,
    /// Скрытое состояние ДО финальной RMS-нормы: n_tok строк по n_embd.
    pub h_pre: Option<Vec<Vec<f32>>>,
    /// После финальной нормы и веса: то, что идёт в голову (n_tok × n_embd).
    pub hidden: Option<Vec<Vec<f32>>>,
    /// Базовые логиты llama.cpp: n_tok строк по n_vocab.
    pub base_logits: Vec<Vec<f32>>,
}

/// Загруженный один раз raw-модель; повторный проход — только decode.
pub struct Probe2 {
    model: *mut sys::llama_model,
    pub n_embd: usize,
    pub n_vocab: usize,
    pub n_layer: usize,
}

unsafe impl Send for Probe2 {}
unsafe impl Sync for Probe2 {}

impl Drop for Probe2 {
    fn drop(&mut self) {
        if !self.model.is_null() {
            unsafe { sys::llama_free_model(self.model) };
        }
    }
}

impl std::fmt::Debug for Probe2 {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Probe2")
            .field("n_embd", &self.n_embd)
            .field("n_vocab", &self.n_vocab)
            .field("n_layer", &self.n_layer)
            .finish()
    }
}

impl Probe2 {
    /// Загрузка GGUF в raw-API (llama_backend инициализируем заранее — v0).
    pub fn load(path: &Path) -> Result<Self, String> {
        let cpath = CString::new(path.to_string_lossy().as_bytes())
            .map_err(|e| format!("probe2: путь не C-совместим: {e}"))?;
        let mut mp = unsafe { sys::llama_model_default_params() };
        mp.n_gpu_layers = 0;
        mp.load_mode = sys::LLAMA_LOAD_MODE_MMAP;
        let model = unsafe { sys::llama_model_load_from_file(cpath.as_ptr(), mp) };
        if model.is_null() {
            return Err("probe2: модель не загрузилась".into());
        }
        let n_embd = unsafe { sys::llama_model_n_embd(model).max(1) as usize };
        let n_vocab =
            unsafe { sys::llama_vocab_n_tokens(sys::llama_model_get_vocab(model)).max(1) as usize };
        let n_layer = unsafe { sys::llama_model_n_layer(model).max(1) as usize };
        Ok(Probe2 { model, n_embd, n_vocab, n_layer })
    }

    /// Один raw-pass; tokens — llama_token ids (i32).
    /// dump=true включает дамп ВСЕХ узлов графа в stderr (разведка).
    pub fn run(&self, tokens: &[i32], dump: bool) -> Result<L2Probe, String> {
        self.run_target_lora(tokens, dump, &format!("ffn_swiglu-{}", self.n_layer - 1), None)
    }

    /// Как run, но захват ДОП-цели другим узлом графа (для отладки —
    /// например «ffn_out-23» — выход down-проекции последнего слоя).
    pub fn run_target(&self, tokens: &[i32], dump: bool, x_name: &str) -> Result<L2Probe, String> {
        self.run_target_lora(tokens, dump, x_name, None)
    }

    /// Захват цели + ММА LoRA-адаптера llama.cpp (мост для проверки «что
    /// llama.cpp реально делает» против предсказаний тренера).
    pub fn run_target_lora(
        &self,
        tokens: &[i32],
        dump: bool,
        x_name: &str,
        lora_path: Option<&Path>,
    ) -> Result<L2Probe, String> {
        let n = tokens.len();
        if n == 0 || n > 128 {
            return Err(format!("probe2: некорректное число позиций {n}"));
        }
        #[allow(unused_mut)]
        let mut st = Box::new(CaptureState {
            n_tokens: n as i64,
            x_name: x_name.to_string(),
            dump,
            x_down: None,
            h_pre: None,
            hidden: None,
            err: None,
        });
        let mut params = unsafe { sys::llama_context_default_params() };
        params.n_ctx = n as u32;
        params.n_batch = n as u32;
        params.n_ubatch = n as u32;
        params.n_seq_max = 1;
        params.n_threads = 2;
        params.n_threads_batch = 2;
        params.pooling_type = sys::LLAMA_POOLING_TYPE_NONE;
        params.embeddings = true;
        params.cb_eval = Some(capture_cb);
        params.cb_eval_user_data = (&*st) as *const CaptureState as *mut c_void;
        let ctx = unsafe { sys::llama_new_context_with_model(self.model, params) };
        if ctx.is_null() {
            return Err("probe2: контекст не создался".into());
        }
        let mut lora_stack: Option<*mut sys::llama_adapter_lora> = None;
        if let Some(lp) = lora_path {
            let cl = CString::new(lp.to_string_lossy().as_bytes())
                .map_err(|e| format!("probe2: путь lora: {e}"))?;
            let ad = unsafe { sys::llama_adapter_lora_init(self.model, cl.as_ptr()) };
            if ad.is_null() {
                unsafe { sys::llama_free(ctx) };
                return Err("probe2: lora не загрузилась (capture)".into());
            }
            unsafe {
                let mut adapters = [ad];
                let mut scales = [1.0f32];
                sys::llama_set_adapters_lora(ctx, adapters.as_mut_ptr(), 1, scales.as_mut_ptr());
            }
            lora_stack = Some(ad);
        }

        let mut batch = unsafe { sys::llama_batch_init(n as c_int, 0, 1) };
        unsafe {
            batch.n_tokens = n as c_int;
            for (i, &tok) in tokens.iter().enumerate() {
                *batch.token.add(i) = tok;
                *batch.pos.add(i) = i as c_int;
                let slot = *batch.seq_id.add(i);
                *batch.n_seq_id.add(i) = 1;
                *slot = 0;
                *batch.logits.add(i) = 1;
            }
        }

        let rc = unsafe { sys::llama_decode(ctx, batch) };
        let mut decode_err = if rc != 0 {
            Some(format!("probe2: decode rc={rc}"))
        } else {
            None
        };

        // Логиты llama.cpp — честный базовый forward (тот же, что у головы).
        let mut base_logits = Vec::with_capacity(n);
        if decode_err.is_none() {
            for i in 0..n as c_int {
                let lp = unsafe { sys::llama_get_logits_ith(ctx, i) };
                if lp.is_null() {
                    decode_err.replace(format!("probe2: логитов для позиции {i} нет"));
                    break;
                }
                let row = unsafe { std::slice::from_raw_parts(lp, self.n_vocab) };
                base_logits.push(row.to_vec());
            }
        }

        unsafe { sys::llama_free(ctx) };
        unsafe { sys::llama_batch_free(batch) };
        if let Some(ad) = lora_stack {
            unsafe { sys::llama_adapter_lora_free(ad) };
        }

        let err = decode_err.or(st.err.clone());
        let mut probe = L2Probe {
            n_tokens: n,
            n_embd: self.n_embd,
            n_vocab: self.n_vocab,
            x_down: st.x_down.clone(),
            h_pre: st.h_pre.clone(),
            hidden: st.hidden.clone(),
            base_logits,
        };
        if let Some(e) = err {
            probe.x_down = None;
            probe.h_pre = None;
            probe.hidden = None;
            return Err(e);
        }
        Ok(probe)
    }

    /// raw-pass С НАЛОЖЕННЫМ LoRA-адаптером llama.cpp (как worker после
    /// apply_lora_from_file): возвращает только логиты по позициям.
    /// scale = 1.0 — как задаёт Клиент-владелец, alpha в самом файле.
    pub fn run_with_lora(&self, tokens: &[i32], lora_path: &Path) -> Result<Vec<Vec<f32>>, String> {
        let n = tokens.len();
        if n == 0 || n > 128 {
            return Err(format!("probe2: некорректное число позиций {n}"));
        }
        let clora = CString::new(lora_path.to_string_lossy().as_bytes())
            .map_err(|e| format!("probe2: путь лоры не C-совместим: {e}"))?;
        let lora = unsafe { sys::llama_adapter_lora_init(self.model, clora.as_ptr()) };
        if lora.is_null() {
            return Err("probe2: lora не загрузилась llama.cpp".into());
        }
        let mut params = unsafe { sys::llama_context_default_params() };
        params.n_ctx = n as u32;
        params.n_batch = n as u32;
        params.n_ubatch = n as u32;
        params.n_seq_max = 1;
        params.n_threads = 2;
        params.n_threads_batch = 2;
        params.pooling_type = sys::LLAMA_POOLING_TYPE_NONE;
        params.embeddings = true;
        let ctx = unsafe { sys::llama_new_context_with_model(self.model, params) };
        if ctx.is_null() {
            unsafe { sys::llama_adapter_lora_free(lora) };
            return Err("probe2: контекст (lora) не создался".into());
        }
        unsafe {
            // 1 адаптер, scale 1.0
            let mut adapters = [lora];
            let mut scales = [1.0f32];
            let rc = sys::llama_set_adapters_lora(
                ctx,
                adapters.as_mut_ptr(),
                1,
                scales.as_mut_ptr(),
            );
            if rc != 0 {
                return Err(format!("probe2: set_adapter rc={rc}"));
            }
        }
        let mut batch = unsafe { sys::llama_batch_init(n as c_int, 0, 1) };
        unsafe {
            batch.n_tokens = n as c_int;
            for (i, &tok) in tokens.iter().enumerate() {
                *batch.token.add(i) = tok;
                *batch.pos.add(i) = i as c_int;
                let slot = *batch.seq_id.add(i);
                *batch.n_seq_id.add(i) = 1;
                *slot = 0;
                *batch.logits.add(i) = 1;
            }
        }
        let rc = unsafe { sys::llama_decode(ctx, batch) };
        let mut out = Vec::with_capacity(n);
        if rc == 0 {
            for i in 0..n as c_int {
                let lp = unsafe { sys::llama_get_logits_ith(ctx, i) };
                if lp.is_null() {
                    break;
                }
                let row = unsafe { std::slice::from_raw_parts(lp, self.n_vocab) };
                out.push(row.to_vec());
            }
        }
        unsafe { sys::llama_free(ctx) };
        unsafe { sys::llama_batch_free(batch) };
        unsafe { sys::llama_adapter_lora_free(lora) };
        if rc != 0 {
            return Err(format!("probe2: decode(lora) rc={rc}"));
        }
        Ok(out)
    }
}

struct CaptureState {
    n_tokens: i64,
    x_name: String,
    dump: bool,

    x_down: Option<Vec<Vec<f32>>>,
    h_pre: Option<Vec<Vec<f32>>>,
    hidden: Option<Vec<Vec<f32>>>,

    err: Option<String>,
}

unsafe extern "C" fn capture_cb(
    t: *mut sys::ggml_tensor,
    ask: bool,
    ud: *mut c_void,
) -> bool {
    let st = unsafe { &mut *(ud as *mut CaptureState) };
    if t.is_null() {
        return true; // не прерываем вычисление
    }
    let tt = unsafe { &*t };
    if ask {
        return st.dump || is_target(tt, st);
    }
    collect(tt, st);
    true
}

/// Целевые узлы (по имени, эта версия llama.cpp имена ставит):
/// • "ffn_swiglu-{L}"  — besie вход ffn_down последнего слоя (L = n_layer-1);
/// • "norm"            — финальная RMS-норма (src[0] = h_pre);
/// • "result_embd_pooled" — скрытое после нормы (вход головы).
fn is_target(t: &sys::ggml_tensor, st: &CaptureState) -> bool {
    if t.src.is_empty() || t.ne.is_empty() {
        return false;
    }
    let name = cstr(t.name.as_ptr());
    if name == st.x_name {
        return true;
    }
    if name == "norm" && t.op == sys::GGML_OP_RMS_NORM {
        // h_pre = src[0] финальной нормы; нужен и сам вход
        let s0 = t.src[0];
        return !s0.is_null();
    }
    if name == "result_embd_pooled" {
        return true;
    }
    false
}

fn cstr(p: *const c_char) -> String {
    if p.is_null() {
        return String::new();
    }
    unsafe { CStr::from_ptr(p).to_string_lossy().into_owned() }
}

/// Копировать n_tok строк тензора (F32, шаг nb[1] байт) в Vec строк.
///
/// SAFETY: вызывается из колбэка после вычисления узла, тем же потоком;
/// данные именно f32, шаг строки nb[1] не меньше ne[0]*4.
fn copy_rows(t: &sys::ggml_tensor, n_tok: usize, what: &str) -> Result<Vec<Vec<f32>>, String> {
    if t.type_ != sys::GGML_TYPE_F32 {
        return Err(format!("{what}: узел не F32 (type={})", t.type_ as c_int));
    }
    if t.data.is_null() {
        return Err(format!("{what}: данные узла null"));
    }
    let cols = t.ne[0] as usize;
    if cols == 0 || cols > (1 << 26) {
        return Err(format!("{what}: подозрительная ширина {cols}"));
    }
    if (t.nb[1] as usize) < cols * 4 {
        return Err(format!("{what}: шаг строки {} < размера строки", t.nb[1]));
    }
    if (t.ne[1] as usize) < n_tok {
        return Err(format!(
            "{what}: строк {} меньше позиций {}",
            t.ne[1], n_tok
        ));
    }
    let base = t.data as *const u8;
    let step = t.nb[1];
    let mut out = Vec::with_capacity(n_tok);
    for p in 0..n_tok {
        let row = unsafe { std::slice::from_raw_parts(base.add(p * step) as *const f32, cols) };
        if !row.iter().all(|x| x.is_finite()) {
            return Err(format!("{what}: не-конечные значения в позиции {p}"));
        }
        out.push(row.to_vec());
    }
    Ok(out)
}

/// ask=false: узел вычислен — читаем данные целевых узлов.
fn collect(t: &sys::ggml_tensor, st: &mut CaptureState) {
    if st.err.is_some() {
        return;
    }
    let name = cstr(t.name.as_ptr());
    if st.dump {
        let _ = std::io::Write::write_fmt(
            &mut std::io::stderr(),
            format_args!(
                "[probe2] узел op={:>2} type={:>2} ne=[{:>6},{:>6},{:>6}] name={}\n",
                t.op,
                t.type_ as c_int,
                t.ne[0],
                t.ne[1],
                t.ne[2],
                name.trim_end()
            ),
        );
        // весовые входы интересных узлов — имена базовых GGUF-тензоров
        let x_src_name = st.x_name.replace("ffn_swiglu", "ffn_out");
        if name == "result_output" || name.starts_with("result_") || name == x_src_name
            || (name == "norm" && t.op == sys::GGML_OP_RMS_NORM)
        {
            for (k, s) in t.src.iter().enumerate() {
                if let Some(s) = unsafe { s.cast_const().as_ref() } {
                    let wn = cstr(s.name.as_ptr());
                    let _ = std::io::Write::write_fmt(
                        &mut std::io::stderr(),
                        format_args!(
                            "[probe2]   {name} src{k}: op={} type={} ne=[{},{}] name='{}'\n",
                            s.op,
                            s.type_ as c_int,
                            s.ne[0],
                            s.ne[1],
                            wn.trim_end()
                        ),
                    );
                }
            }
        }
        // дальше — обычный захват (в дамп-режиме тоже собираем данные)
    }
    let name = cstr(t.name.as_ptr());
    if name == st.x_name {
        // swiGLU-выход последнего слоя = вход ffn_down
        let x = t as *const sys::ggml_tensor;
        match copy_rows(unsafe { &*x }, st.n_tokens as usize, "x_down") {
            Ok(v) => st.x_down = Some(v),
            Err(e) => st.err = Some(e),
        }
        return;
    }
    if st.dump && (name == "result_output" || name.starts_with("result_")) {
        for (k, s) in t.src.iter().enumerate() {
            if let Some(s) = unsafe { s.cast_const().as_ref() } {
                if !s.src.is_empty() {
                    let wn = cstr(s.name.as_ptr());
                    let _ = std::io::Write::write_fmt(
                        &mut std::io::stderr(),
                        format_args!(
                            "[probe2]   {name} src{k}: op={} type={} ne=[{},{}] name='{}'\n",
                            s.op,
                            s.type_ as c_int,
                            s.ne[0],
                            s.ne[1],
                            wn.trim_end()
                        ),
                    );
                }
            }
        }
    }
    if name == "norm" && t.op == sys::GGML_OP_RMS_NORM {
        let hpre = t.src[0];
        if hpre.is_null() {
            st.err = Some("probe2: нет входа у финальной нормы".into());
            return;
        }
        match copy_rows(unsafe { &*hpre }, st.n_tokens as usize, "h_pre") {
            Ok(hp) => st.h_pre = Some(hp),
            Err(e) => st.err = Some(e),
        }
        if st.dump {
            for (k, s) in t.src.iter().enumerate() {
                if let Some(s) = unsafe { s.cast_const().as_ref() } {
                    let wn = cstr(s.name.as_ptr());
                    let _ = std::io::Write::write_fmt(
                        &mut std::io::stderr(),
                        format_args!(
                            "[probe2]   norm src{k}: op={} type={} ne=[{},{}] name='{}'\n",
                            s.op,
                            s.type_ as c_int,
                            s.ne[0],
                            s.ne[1],
                            wn.trim_end()
                        ),
                    );
                }
            }
        }
        return;
    }
    if name == "result_embd_pooled" {
        let node = t as *const sys::ggml_tensor;
        match copy_rows(unsafe { &*node }, st.n_tokens as usize, "hidden") {
            Ok(hp) => st.hidden = Some(hp),
            Err(e) => st.err = Some(e),
        }
    }
}

// ---------------------------------------------------------------------------
// Тесты (живой GGUF через env DI_NET_GGUF).
// ---------------------------------------------------------------------------

#[cfg(test)]
mod impl_probe2 {
    use super::*;
    use crate::{LlamaModelParams, probe_tokens};
    use std::path::Path;

    fn tokenize_gated(path: &Path, text: &str) -> Vec<i32> {
        let be = crate::llama_backend().expect("backend");
        let model = crate::LlamaModel::load_from_file(&be, path, &LlamaModelParams::default())
            .expect("safe-модель");
        model
            .vocab()
            .tokenize(text.as_bytes(), false, true)
            .into_iter()
            .map(|t| t.0)
            .collect()
    }

    /// Разведка графа: полный дамп узлов под DI_NET_PROBE_DUMP=1.
    /// В CI с моделью: утверждаем форму; без неё — тест молча пропуск.
    #[test]
    fn graph_walk_gated() {
        let Some(path) = std::env::var("DI_NET_GGUF").ok() else {
            return;
        };
        let dump = std::env::var("DI_NET_PROBE_DUMP").is_ok();
        if dump {
            eprintln!("graph_walk: дамп ниже");
        }
        let eng = Probe2::load(Path::new(&path)).expect("модель");
        let tok = tokenize_gated(Path::new(&path), "вопрос привет ответ");
        let p = eng.run(&tok, dump).expect("проход");
        assert_eq!(p.base_logits.len(), tok.len());
        assert_eq!(p.n_tokens, tok.len());
        for r in &p.base_logits {
            assert_eq!(r.len(), p.n_vocab);
            assert!(r.iter().all(|x| x.is_finite()));
        }
    }

    /// raw-выход согласован с безопасным probe (слушаемый кандидатом v0):
    /// hidden (после нормы) совпадает попозиционно, логиты полные.
    #[test]
    fn probe2_matches_safe_probe_gated() {
        let Some(path) = std::env::var("DI_NET_GGUF").ok() else {
            return;
        };
        let be = crate::llama_backend().expect("backend");
        let model = crate::LlamaModel::load_from_file(
            &be,
            Path::new(&path),
            &LlamaModelParams::default(),
        )
        .expect("safe-модель");
        let toks = model
            .vocab()
            .tokenize("Вопрос: привет\nОтвет: привет мир".as_bytes(), false, true);
        let tokv: Vec<i32> = toks.iter().map(|t| t.0).collect();

        let probe = probe_tokens(&model, &be, &toks).expect("safe probe");
        let eng = Probe2::load(Path::new(&path)).expect("raw модель");
        let mut pr = eng.run(&tokv, false).expect("raw probe");

        // hidden: второй кандидат на пост-норм (данные с узла графа).
        let hid = pr.hidden.clone().expect("hidden captured");
        let compare_rows = hid.len() == tokv.len();
        let mut maxdiff: f32 = 0.0;
        if compare_rows {
            for (p, row) in hid.iter().enumerate() {
                for (i, a) in row.iter().enumerate() {
                    let b = probe.hidden[p * probe.n_embd + i];
                    if (a - b).abs() > maxdiff {
                        maxdiff = (a - b).abs();
                    }
                }
            }
        }
        assert!(compare_rows, "hidden собран для всех позиций");
        assert!(
            maxdiff < 1e-3,
            "hidden расходится с safe-probe: maxdiff={maxdiff}"
        );

        // x_down и h_pre НЕ пустые и конечные
        let xd = pr.x_down.clone().expect("x_down captured");
        for row in &xd {
            assert!(!row.is_empty());
            assert!(row.iter().all(|x| x.is_finite()));
        }
        let hp = pr.h_pre.clone().expect("h_pre captured");
        for row in &hp {
            assert_eq!(row.len(), probe.n_embd);
            assert!(row.iter().all(|x| x.is_finite()));
        }
        pr.x_down = None;
        pr.h_pre = None;
        pr.hidden = None;
    }

    /// Cross-check де-кванта: голова из СВОЕГО GGUF-ридера
    /// (`output.weight`, Q8_0 → f32) · hidden (result_embd_pooled)
    /// против логитов llama.cpp. ggml перед matmul квантизует скрытые
    /// активации в Q8_0 — мы симулируем ту же процедуру; после неё
    /// расхождения обязаны быть шумом ≤2e-2.
    #[test]
    fn head_dequant_matches_llama_logits_gated() {
        let Some(path) = std::env::var("DI_NET_GGUF").ok() else {
            return;
        };
        let gg = dinet_train::gguf::GgufModel::open(Path::new(&path)).expect("gguf-ридер");
        let head = match gg.read_f32("output.weight") {
            Ok(h) => h,
            Err(_) => {
                // tied-модель (SmolLM2, gemma-3 ≤1b): головы-тензора нет,
                // де-квант головы не о чем — честная остановка
                eprintln!("tied-модель (output.weight нет): head-часть пропущена");
                return;
            }
        };
        let n_embd = gg.num("embedding_length").expect("n_embd в kv") as usize;
        assert_eq!(head.row, n_embd, "строка головы = n_embd");

        let eng = Probe2::load(Path::new(&path)).expect("модель");
        let tok = tokenize_gated(Path::new(&path), "вопрос привет ответ");
        let p = eng.run(&tok, false).expect("проход");
        let hidden = p.hidden.clone().expect("hidden captured");
        assert_eq!(hidden.len(), tok.len());

        // активационная квантизация Q8_0 (как делает ggml перед vec_dot
        // для Q8_0-веса): блоки по 32 со своим масштабом d
        fn quant8(v: &[f32]) -> Vec<f32> {
            let mut out = Vec::with_capacity(v.len());
            assert_eq!(v.len() % 32, 0);
            for b in 0..v.len() / 32 {
                let blk = &v[b * 32..b * 32 + 32];
                let amax = blk.iter().fold(0f32, |m, x| m.max(x.abs()));
                let d = amax / 127.0;
                for x in blk {
                    let q = if d == 0.0 {
                        0i8
                    } else {
                        (x / d).round().clamp(-127.0, 127.0) as i8
                    };
                    out.push(d * q as f32);
                }
            }
            out
        }

        // сверка head q8/float по первой и последней позиции
        let d = n_embd;
        let nv = p.n_vocab as usize;
        for pos in [0usize, tok.len() - 1] {
            let h_q = quant8(&hidden[pos]);
            let h_f = &hidden[pos];
            let rows = &head.data[..];
            let mut diff_q = 0f32;
            let mut diff_f = 0f32;
            for j in 0..nv {
                let row = &rows[j * d..(j + 1) * d];
                let mut wq = 0f32;
                let mut wf = 0f32;
                for (k, wr) in row.iter().enumerate() {
                    wq += wr * h_q[k];
                    wf += wr * h_f[k];
                }
                let base = p.base_logits[pos][j];
                diff_q = diff_q.max((wq - base).abs());
                diff_f = diff_f.max((wf - base).abs());
            }
            eprintln!("pos={pos}: maxdiff(точный f32)={diff_f:.4} maxdiff(Q8_0-акт.)={diff_q:.4}");
            // активационная квантизация обязана попасть в шум ggml
            assert!(
                diff_q < 2e-2,
                "логиты расходятся после Q8_0-активаций: {diff_q}"
            );
            // "сырой" f32-dot — ровно на уровень квантизационного шума ниже
            assert!(
                diff_f > diff_q,
                "f32-dot неожиданно точнее квантованного — странно"
            );
        }
    }

    /// Завершающий cross-check v1: обученный СОСТАВНОЙ адаптер (голова +
    /// ffn_down последнего слоя) читается из файла СВОИМ ридером, и
    /// предсказание логит-дельты тренера сравнивается с тем, что РЕАЛЬНО
    /// даёт llama.cpp при наложении адаптера.
    /// Env: DI_NET_GGUF (модель) + DI_NET_ADAPTER (файл .lora.gguf).
    #[test]
    fn lora_delta_parity_gated() {
        let (model_path, adapter_path) =
            match (std::env::var("DI_NET_GGUF").ok(), std::env::var("DI_NET_ADAPTER").ok()) {
                (Some(m), Some(a)) => (m, a),
                _ => return,
            };
        use dinet_train::AdapterPiece;
        let gg = dinet_train::gguf::GgufModel::open(Path::new(&model_path)).expect("gguf");
        // размеры БЕЗ префикса арха: поиск по суффиксу (qwen2/llama/...)
        let d = gg.num("embedding_length").expect("n_embd в gguf") as usize;
        let n_layer = gg.num("block_count").unwrap_or(0) as usize;
        assert!(d > 0 && n_layer > 0, "размеры модели читаются");
        // арх берётся из МОДЕЛИ: llama.cpp сверяет arch адаптера с моделью
        let arch_m = gg.architecture().as_deref().unwrap_or("llama").to_string();

        // адаптер — целиком СВОЙ ридер (пары lora_a/lora_b, F32)
        let ad = dinet_train::gguf::GgufModel::open(Path::new(&adapter_path)).expect("adapter");
        // голова ОПЦИОНАЛЬНА: tied-модели (SmolLM2, gemma-3 ≤1b и др.) пишут
        // только слой — head-часть паритета тогда пропускается
        let head_pr =
            ad.read_f32("output.weight.lora_a").ok().zip(ad.read_f32("output.weight.lora_b").ok());
        let (base_name, a_l, b_l) = match ad
            .tensors
            .keys()
            .find(|k| k.ends_with(".ffn_down.weight.lora_a"))
            .map(|k| k.trim_end_matches(".lora_a").to_string())
        {
            Some(b) => (
                Some(b.clone()),
                Some(ad.read_f32(&format!("{b}.lora_a")).expect("layer lora_a")),
                Some(ad.read_f32(&format!("{b}.lora_b")).expect("layer lora_b")),
            ),
            None => (None, None, None),
        };
        let eng = Probe2::load(Path::new(&model_path)).expect("модель");
        let tok = tokenize_gated(Path::new(&model_path), "вопрос привет ответ");
        let base = eng.run(&tok, false).expect("базовый проход");
        let with = eng
            .run_with_lora(&tok, Path::new(&adapter_path))
            .expect("проход с адаптером");
        assert_eq!(with.len(), tok.len());
        let _ = &with;

        let tmpd = std::env::temp_dir();

        // --- изоляция БЕЗ слойной лоры: голова на выходе norm Ф32 → сравнение точное.
        // Пропускается для layer-only адаптера (tied-модели: SmolLM2 и др.).
        if let Some((a_h, b_h)) = head_pr.as_ref() {
            assert_eq!(a_h.row as usize, d, "строка головы = n_embd");
            let head_only = tmpd.join("dinet-parity-head.gguf");
            dinet_train::save_adapter_pieces(&arch_m, &head_only, &[AdapterPiece {
                base: "output.weight".into(),
                a: a_h.data.clone(),
                a_ne0: a_h.row as u64,
                b: b_h.data.clone(),
                b_ne1: b_h.rows as u64,
                r: a_h.rows as u64,
            }])
            .expect("head-only пишется");
            let obs_head = eng.run_with_lora(&tok, &head_only).expect("head-only проход");
            let mut mh = 0f32;
            {
                let pos = 0usize;
                let hid = base.hidden.as_ref().expect("hidden")[pos].clone();
                for j in 0..eng.n_vocab {
                    let mut z_h = vec![0f32; a_h.rows];
                    for (i2, zz) in z_h.iter_mut().enumerate() {
                        let row = &a_h.data[i2 * d..(i2 + 1) * d];
                        *zz = row.iter().zip(hid.iter()).map(|(w, v)| w * v).sum();
                    }
                    let brow = &b_h.data[j * b_h.row..(j + 1) * b_h.row];
                    let pred: f32 = brow.iter().zip(z_h.iter()).map(|(w, v)| w * v).sum();
                    let ob = obs_head[pos][j] - base.base_logits[pos][j];
                    mh = mh.max((pred - ob).abs());
                }
            }
            eprintln!("head-only: max|pred−obs|={mh:.4}");
            assert!(mh < 1e-3, "head-паритет сломался: {mh}");
            let _ = std::fs::remove_file(&head_only);
        } else {
            eprintln!("layer-only адаптер (tied): head-часть паритета пропущена");
        }

        // --- слой-only: llama вычисляет δ на ЛИНЕЙНОЙ точке l_out-{il} =>
        // obsδ = l_out(с лорой) − l_out(база) обязан равняться B·(A·xsw) точно.
        // v1 тренирует только ffn_down ПОСЛЕДНЕГО слоя; слой парсится из имени
        // (qwen2 → blk.23, llama-арх (SmolLM2-135M) → blk.29 — динамично).
        if let (Some(bn), Some(al), Some(bl)) = (base_name, a_l, b_l) {
            let il: usize = bn
                .split('.')
                .nth(1)
                .and_then(|s| s.parse().ok())
                .expect("номер слоя в имени тензора");
            assert_eq!(il + 1, n_layer, "v1 тренирует ПОСЛЕДНИЙ слой (blk.{il})");
            let lay_only = tmpd.join("dinet-parity-layer.gguf");
            dinet_train::save_adapter_pieces(&arch_m, &lay_only, &[AdapterPiece {
                base: bn.clone(),
                a: al.data.clone(),
                a_ne0: al.row as u64,
                b: bl.data.clone(),
                b_ne1: bl.rows as u64,
                r: al.rows as u64,
            }])
            .expect("layer-only пишется");
            // вход ffn_down = выход swiGLU слоя llama (ffn_silu — запасной
            // вариант naming для архов без PAR-сплит swiGLU)
            let psx = match eng.run_target(&tok, false, &format!("ffn_swiglu-{il}")) {
                Ok(pp) => pp,
                Err(_) => eng
                    .run_target(&tok, false, &format!("ffn_silu-{il}"))
                    .expect("вход ffn_down (ffn_swiglu/ffn_silu)"),
            };
            let xs = psx.x_down.expect("вход ffn_down не захвачен");
            let plo = eng
                .run_target_lora(&tok, false, &format!("l_out-{il}"), Some(&lay_only))
                .expect("l_out лора проход");
            let pbo = eng
                .run_target(&tok, false, &format!("l_out-{il}"))
                .expect("l_out баз проход");
            let yl = plo.x_down.as_ref().expect("l_out лора");
            let yb = pbo.x_down.as_ref().expect("l_out база");
            let n_in = al.row as usize;
            let r_l = al.rows as usize;
            let mut ml = 0f32;
            for pos in 0..tok.len() {
                // z(q, pos) = Σ_j A[q][j]·xsw[pos][j]
                let mut z = vec![0f32; r_l];
                for (q, zz) in z.iter_mut().enumerate() {
                    let row = &al.data[q * n_in..(q + 1) * n_in];
                    *zz = row.iter().zip(xs[pos].iter()).map(|(w, v)| w * v).sum();
                }
                for (k, (lo, bo)) in yl[pos].iter().zip(yb[pos].iter()).enumerate() {
                    let brow = &bl.data[k * r_l..(k + 1) * r_l];
                    let pred: f32 = brow.iter().zip(z.iter()).map(|(w, v)| w * v).sum();
                    ml = ml.max((pred - (lo - bo)).abs());
                }
            }
            eprintln!("layer-only: max|pred−obs| ({bn}) = {ml:.6}");
            assert!(ml < 1e-3, "layer-паритет сломался: {ml}");
            let _ = std::fs::remove_file(&lay_only);
        }
    }

    /// ЖИВОЙ ЦИКЛ v1 end-to-end: capture llama → обучение layer-LoRA НАШИМ
    /// ходом (AdamW, как в webchat) → сохранённый адаптер ЧИТАЕТСЯ llama.cpp
    /// → ЦЕЛЕВОЙ токен растёт против старого argmax в ЖИВОЙ модели.
    /// Гейт: DI_NET_GGUF=/путь.gguf (арх любой: qwen2, llama-tied SmolLM2 —
    /// голова лосса = output.weight / token_embd.weight, как в webchat).
    #[test]
    fn layer_train_moves_logits_gated() {
        let Some(model_path) = std::env::var("DI_NET_GGUF").ok() else {
            return;
        };
        use dinet_train::layer::{HeadWeights, LayerBatch, LayerLora, LayerSpec};

        let gg = dinet_train::gguf::GgufModel::open(Path::new(&model_path)).expect("gguf");
        let arch = gg.architecture().as_deref().unwrap_or("llama").to_string();
        let n_ff = gg.num("feed_forward_length").expect("n_ff в gguf") as usize;
        let eps = gg.f32val("layer_norm_rms_epsilon").unwrap_or(1e-5);
        // голова для ЛОССА: output.weight; tied-модель — token_embd.weight
        let head_w = gg
            .read_f32("output.weight")
            .ok()
            .or_else(|| gg.read_f32("token_embd.weight").ok())
            .expect("голова лосса (output/token_embd)");
        let norm_w = gg.read_f32("output_norm.weight").expect("финальная норма");

        let eng = Probe2::load(Path::new(&model_path)).expect("модель");
        assert_eq!(head_w.row as usize, eng.n_embd, "строка головы = n_embd");
        // пара учителя: p = 4 слова микро-языка, a = 1 слово
        let toks = tokenize_gated(Path::new(&model_path), "Вопрос: ай би ви ги\nОтвет: ди");
        let targets: Vec<i32> = toks.iter().skip(1).copied().collect();
        let n_pos = targets.len();

        let probe = eng.run(&toks, false).expect("базовый проход");
        // ХВОСТ: последние 4 позиции (там живёт ответ «ди»); позиции
        // независимы — CE на них тот же, шаг дешевле
        let tail4 = n_pos.saturating_sub(4);
        let batch = LayerBatch {
            x: probe.x_down.as_ref().expect("x down")[tail4..n_pos].to_vec(),
            h_pre: probe.h_pre.as_ref().expect("h_pre")[tail4..n_pos].to_vec(),
            hidden: probe.hidden.as_ref().expect("hidden")[tail4..n_pos].to_vec(),
            base_logits: probe.base_logits[tail4..n_pos].to_vec(),
            target: targets[tail4..n_pos].to_vec(),
        };

        let mut ly = LayerLora::new(
            LayerSpec { n_in: n_ff as u32, n_out: eng.n_embd as u32, r: 8 },
            0x0BAB1E as u64,
        );
        let hw = HeadWeights {
            w: head_w.data.clone(),
            n_vocab: eng.n_vocab,
            d: eng.n_embd,
        };
        let lr = 9e-4f32; // как в webchat: lr 1.5e-3 · 0.6
        let mut loss_first = f32::NAN;
        let mut loss_last = f32::NAN;
        for _ in 0..24 {
            if let Some(ls) = ly.train_step(&batch, &hw, &norm_w.data, eps, lr, 0.0) {
                if loss_first.is_nan() {
                    loss_first = ls.loss;
                }
                loss_last = ls.loss;
            }
        }
        eprintln!("конвейер: loss {loss_first:.3} → {loss_last:.3} за 24 шага");
        assert!(loss_last < loss_first, "лосс слоя не падает: {loss_first} → {loss_last}");

        // сохранение в llama-формат (arch из МОДЕЛИ) и живая проверка
        let base_name = format!("blk.{}.ffn_down.weight", eng.n_layer - 1);
        let piece = ly.to_piece(&base_name);
        let tmp = std::env::temp_dir().join(format!("dinet-moves-{}.gguf", std::process::id()));
        dinet_train::save_adapter_pieces(&arch, &tmp, &[piece]).expect("adapter пишется");
        let with = eng.run_with_lora(&toks, &tmp).expect("проход с адаптером");
        let _ = std::fs::remove_file(&tmp);

        // позиция p предсказывает токен p+1: «ди» — ПОСЛЕДНЯЯ цель —
        // сравниваем её CE (до/с адаптером) и сдвиг против старого argmax
        let fn_ce = |logits: &[f32], t: usize| {
            let mx = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let lse = mx + logits.iter().map(|v| (v - mx).exp()).sum::<f32>().ln();
            lse - logits[t as usize]
        };
        let p = n_pos - 1;
        let t = targets[p] as usize;
        let ce_b = fn_ce(&probe.base_logits[p], t);
        let ce_w = fn_ce(&with[p], t);
        eprintln!("CE целевого токена: base {ce_b:.3} → с адаптером {ce_w:.3}");
        assert!(
            ce_w < ce_b,
            "адаптер не снизил CE целевого токена в ЖИВОЙ llama: {ce_b} → {ce_w}"
        );
        // старый argmax (без целевого токена) — руками, f32 без Ord
        let mut lm = usize::MAX;
        let mut lm_v = f32::NEG_INFINITY;
        for (j, v) in probe.base_logits[p].iter().enumerate() {
            if j != t && *v > lm_v {
                lm_v = *v;
                lm = j;
            }
        }
        assert!(lm != usize::MAX, "нет argmax");
        let d_t = with[p][t] - probe.base_logits[p][t];
        let d_m = with[p][lm] - probe.base_logits[p][lm];
        eprintln!("сдвиги: цель {:+.3}, старый argmax {:+.3}", d_t, d_m);
        assert!(
            d_t > d_m + 0.1,
            "целевой токен не обогнал старый argmax: цель {:+.3} argmax {:+.3}",
            d_t,
            d_m
        );
    }
}
