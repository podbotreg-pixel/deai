// ---------------------------------------------------------------------------
// GGUF-ЧИТАТЕЛЬ (train-cpp v1): нужен, чтобы ТРЕНАЖЁР мог читать веса модели
// (голова `output.weight` = Q8_0, норма `output_norm.weight` = F32) прямо из
// файла GGUF — llama.cpp их наружу не отдаёт (нет llama_get_model_tensor).
//
// Только ЧТЕНИЕ v3, типы: F32, F16, Q8_0 (самые простые и в точности те,
// что llama.cpp хранит у норм/головы маленьких моделей). K-кванты (Q4_K,
// Q6_K…) сознательно не читаются: внутренние матрицы нам не нужны — для
// layer-LoRA нужны ТОЛЬКО норма и голова.
// ---------------------------------------------------------------------------

#![allow(dead_code)]

use std::collections::BTreeMap;
use std::io::{Read, Seek, SeekFrom};

/// Значение kv GGUF (не всё — только то, что может нам пригодиться).
#[derive(Debug, Clone, PartialEq)]
pub enum GVal {
    Str(String),
    F32(f32),
    I32(i32),
    U32(u32),
    U64(u64),
    Bool(bool),
    Other,
}

/// Информация о тензоре из таблицы GGUF.
#[derive(Debug, Clone)]
pub struct TensorInfo {
    pub name: String,
    /// размеры в порядке GGUF (ne[0] — самый быстрый)
    pub ne: Vec<u64>,
    pub ggml_type: u32,
    pub offset: u64,
}

/// Открытый GGUF: метаданные + таблица тензоров; данные читаются seek'ом.
pub struct GgufModel {
    path: std::path::PathBuf,
    pub kv: BTreeMap<String, GVal>,
    pub tensors: BTreeMap<String, TensorInfo>,
    data_start: u64,
}

/// Загруженный тензор: строки по ne[0], rows по остальным размерам.
#[derive(Debug, Clone)]
pub struct LoadedTensor {
    pub data: Vec<f32>,
    /// ширина строки (ne[0])
    pub row: usize,
    /// число строк (произведение ne[1..])
    pub rows: usize,
}

impl GgufModel {
    /// Открыть и распарсить заголовок+kv+таблицу тензоров (без данных).
    pub fn open(path: &std::path::Path) -> Result<Self, String> {
        let mut f = std::fs::File::open(path).map_err(|e| format!("gguf: open: {e}"))?;
        let mut magic = [0u8; 4];
        read(&mut f, &mut magic)?;
        if &magic != b"GGUF" {
            return Err("gguf: не GGUF".into());
        }
        let ver = rd_u32(&mut f)?;
        if ver < 2 || ver > 3 {
            return Err(format!("gguf: версия {ver} не поддерживается"));
        }
        let n_tensors = rd_u64(&mut f)?;
        let n_kvs = rd_u64(&mut f)?;
        if n_tensors > 1 << 17 || n_kvs > 1 << 17 {
            return Err("gguf: подозрительное число записей".into());
        }

        let mut kv = BTreeMap::new();
        for _ in 0..n_kvs {
            let key = rd_string(&mut f)?;
            let vt = rd_u32(&mut f)?;
            let val = read_kv_val(&mut f, vt)?;
            kv.insert(key, val);
        }

        let mut tensors = BTreeMap::new();
        for _ in 0..n_tensors {
            let name = rd_string(&mut f)?;
            let nd = rd_u32(&mut f)? as usize;
            if nd == 0 || nd > 4 {
                return Err(format!("gguf: тензор '{name}' с nd={nd}"));
            }
            let mut ne = Vec::with_capacity(nd);
            let mut total: u64 = 1;
            for _ in 0..nd {
                let d = rd_u64(&mut f)?;
                if d == 0 || d > (1u64 << 31) {
                    return Err(format!("gguf: тензор '{name}' с размером {d}"));
                }
                total = total.saturating_mul(d);
                ne.push(d);
            }
            let t = rd_u32(&mut f)?;
            let off = rd_u64(&mut f)?;
            tensors.insert(name.clone(), TensorInfo { name, ne, ggml_type: t, offset: off });
        }
        let align = match kv.get("general.alignment") {
            Some(GVal::U32(a)) => (*a as u64).max(1),
            _ => 32,
        };
        let pos = f.stream_position().map_err(|e| format!("gguf: pos: {e}"))?;
        let pad = (align - pos % align) % align;
        f.seek(SeekFrom::Current(pad as i64))
            .map_err(|e| format!("gguf: seek data: {e}"))?;
        let data_start = f.stream_position().map_err(|e| format!("gguf: pos: {e}"))?;
        Ok(GgufModel { path: path.to_path_buf(), kv, tensors, data_start })
    }

    /// Значение kv по ИМЕННОМУ ключу.
    pub fn val(&self, key: &str) -> Option<&GVal> {
        self.kv.get(key)
    }

    /// Значение kv по суффиксу (ключи llama.cpp имеют префикс архитектуры:
    /// и `llama.block_count`, и `qwen2.block_count`).
    pub fn val_by_suffix(&self, suffix: &str) -> Option<&GVal> {
        self.kv
            .iter()
            .rev()
            .find(|(k, _)| k.ends_with(suffix))
            .map(|(_, v)| v)
    }

    pub fn architecture(&self) -> Option<String> {
        match self.kv.get("general.architecture") {
            Some(GVal::Str(s)) => Some(s.clone()),
            _ => None,
        }
    }

    /// Обезличенный по суффиксу численный параметр.
    pub fn num(&self, suffix: &str) -> Option<i64> {
        match self.val_by_suffix(suffix) {
            Some(GVal::I32(v)) => Some(*v as i64),
            Some(GVal::U32(v)) => Some(*v as i64),
            Some(GVal::U64(v)) => Some(*v as i64),
            Some(GVal::F32(v)) => Some(*v as i64),
            _ => None,
        }
    }

    pub fn f32val(&self, suffix: &str) -> Option<f32> {
        match self.val_by_suffix(suffix) {
            Some(GVal::F32(v)) => Some(*v),
            _ => None,
        }
    }

    pub fn tensor(&self, name: &str) -> Option<&TensorInfo> {
        self.tensors.get(name)
    }

    /// Прочитать тензор целиком в f32 (де-квант F32/F16/Q8_0).
    pub fn read_f32(&self, name: &str) -> Result<LoadedTensor, String> {
        let Some(ti) = self.tensor(name) else {
            return Err(format!("gguf: тензора '{name}' нет"));
        };
        let mut total: u64 = 1;
        for d in &ti.ne {
            total = total.saturating_mul(*d);
        }
        if total == 0 || total > (1u64 << 31) {
            return Err(format!("gguf: '{name}' слишком велик ({total})"));
        }
        let bytes = byte_size_q(ti.ggml_type, total)
            .ok_or_else(|| format!("gguf: '{name}': тип {} не читается", ti.ggml_type))?;
        let mut f = std::fs::File::open(&self.path).map_err(|e| format!("gguf: open: {e}"))?;
        f.seek(SeekFrom::Start(self.data_start + ti.offset))
            .map_err(|e| format!("gguf: seek '{name}': {e}"))?;
        let mut raw = vec![0u8; bytes as usize];
        read(&mut f, &mut raw)?;

        let data = dequant(&raw, ti.ggml_type, total as usize)?;
        if data.iter().any(|x| !x.is_finite()) {
            return Err(format!("gguf: '{name}': не-конечные значения"));
        }
        Ok(LoadedTensor { data, row: ti.ne[0] as usize, rows: total as usize / ti.ne[0] as usize })
    }
}

// ---- чтение заголовка -------------------------------------------------------

fn read(f: &mut std::fs::File, buf: &mut [u8]) -> Result<(), String> {
    f.read_exact(buf).map_err(|e| format!("gguf: read: {e}"))
}

fn rd_u32(f: &mut std::fs::File) -> Result<u32, String> {
    let mut b = [0u8; 4];
    read(f, &mut b)?;
    Ok(u32::from_le_bytes(b))
}

fn rd_u64(f: &mut std::fs::File) -> Result<u64, String> {
    let mut b = [0u8; 8];
    read(f, &mut b)?;
    Ok(u64::from_le_bytes(b))
}

fn rd_string(f: &mut std::fs::File) -> Result<String, String> {
    let len = rd_u64(f)? as usize;
    if len > (1 << 24) {
        return Err(format!("gguf: строка длиной {len}"));
    }
    let mut buf = vec![0u8; len];
    read(f, &mut buf)?;
    Ok(String::from_utf8_lossy(&buf).into_owned())
}

fn skip_n(f: &mut std::fs::File, n: u64, sz: u64) -> Result<(), String> {
    let total = n.saturating_mul(sz);
    if total > (1 << 31) {
        return Err("gguf: слишком большое значение kv".into());
    }
    let mut left = total;
    let mut buf = [0u8; 8192];
    while left > 0 {
        let chunk = left.min(8192);
        read(f, &mut buf[..chunk as usize])?;
        left -= chunk;
    }
    Ok(())
}

fn skip_string(f: &mut std::fs::File) -> Result<(), String> {
    let len = rd_u64(f)?;
    if len > (1 << 24) {
        return Err(format!("gguf: строка {len}"));
    }
    let mut buf = vec![0u8; len as usize];
    read(f, &mut buf)
}

fn maybe_string(f: &mut std::fs::File) -> Result<GVal, String> {
    Ok(GVal::Str(rd_string(f)?))
}

/// Прочитать или пропустить значение kv по типу.
fn read_kv_val(f: &mut std::fs::File, vt: u32) -> Result<GVal, String> {
    Ok(match vt {
        0 => GVal::U32(rd_u32(f)? as u32), // u8 — читаем как u32 по байту
        1 => { let mut b=[0u8;1]; read(f,&mut b)?; GVal::I32(b[0] as i8 as i32) }
        2 | 3 => { let mut b=[0u8;2]; read(f,&mut b)?; GVal::Other }
        4 => GVal::U32(rd_u32(f)?),
        5 => { let mut b=[0u8;4]; read(f,&mut b)?; GVal::I32(i32::from_le_bytes(b)) }
        6 => { let mut b=[0u8;4]; read(f,&mut b)?; GVal::F32(f32::from_le_bytes(b)) }
        7 => { let mut b=[0u8;1]; read(f,&mut b)?; GVal::Bool(b[0] != 0) }
        8 => maybe_string(f)?, // STRING
        9 => {
            // ARRAY: тип элементов, число, значения (не читаем — пропускаем)
            let et = rd_u32(f)?;
            let n = rd_u64(f)?;
            let sz = match et {
                0 | 1 | 7 => 1, 2 | 3 => 2, 4 | 5 | 6 => 4,
                10 | 11 | 12 => 8, 8 => 0, 9 => return Err("gguf: вложенные массивы".into()),
                _ => return Err(format!("gguf: неизвестный тип {et}")),
            };
            if sz == 0 {
                // массив строк — построчно
                for _ in 0..n {
                    skip_string(f)?;
                }
            } else {
                skip_n(f, n, sz)?;
            }
            GVal::Other
        }
        10 | 11 | 12 => { let _ = rd_u64(f)?; GVal::Other }
        _ => return Err(format!("gguf: неизвестный kv-тип {vt}")),
    })
}

// ---- де-квант ---------------------------------------------------------------

/// Число БАЙТ данных тензора (по типу и числу значений).
pub fn byte_size_q(ggml_type: u32, n_values: u64) -> Option<u64> {
    match ggml_type {
        0 => Some(n_values.checked_mul(4)?), // F32
        1 => Some(n_values.checked_mul(2)?), // F16
        8 => {
            // Q8_0: блок 32 значения, 34 байта
            if n_values % 32 != 0 {
                return None;
            }
            Some((n_values / 32).checked_mul(34)?)
        }
        _ => None,
    }
}

fn f16_to_f32(h: u16) -> f32 {
    let s = ((h >> 15) as u32) << 31;
    let e = ((h >> 10) & 0x1f) as i32;
    let m = (h & 0x3ff) as u32;
    match e {
        0 => {
            if m == 0 {
                f32::from_bits(s)
            } else {
                let v = (m as f32) * 2f32.powi(-24);
                if s == 0 { v } else { -v }
            }
        }
        31 => {
            if m == 0 {
                f32::from_bits(s | 0x7f80_0000)
            } else {
                f32::from_bits(s | 0x7fc0_0000 | (m << 13))
            }
        }
        _ => f32::from_bits(s | (((e - 15 + 127) as u32) << 23) | (m << 13)),
    }
}

/// Де-квант данных тензора в f32.
pub fn dequant(raw: &[u8], ggml_type: u32, n_values: usize) -> Result<Vec<f32>, String> {
    let expect = byte_size_q(ggml_type, n_values as u64)
        .ok_or_else(|| format!("де-квант: тип {ggml_type} не поддерживается"))?
        as usize;
    if raw.len() < expect {
        return Err(format!(
            "де-квант: данных {} < нужно {expect}",
            raw.len()
        ));
    }
    let mut out = vec![0f32; n_values];
    let raw = &raw[..expect];
    match ggml_type {
        0 => {
            for (i, slot) in out.iter_mut().enumerate() {
                *slot = f32::from_le_bytes(raw[i * 4..i * 4 + 4].try_into().unwrap());
            }
        }
        1 => {
            for (i, slot) in out.iter_mut().enumerate() {
                *slot = f16_to_f32(u16::from_le_bytes(raw[i * 2..i * 2 + 2].try_into().unwrap()));
            }
        }
        8 => {
            // Q8_0: блок = [f16 d][32 × i8]; value = d × q
            let blocks = n_values / 32;
            for bi in 0..blocks {
                let off = bi * 34;
                let h = u16::from_le_bytes(raw[off..off + 2].try_into().unwrap());
                let d = f16_to_f32(h);
                for j in 0..32 {
                    let q = raw[off + 2 + j] as i8;
                    out[bi * 32 + j] = d * q as f32;
                }
            }
        }
        _ => return Err(format!("де-квант: тип {ggml_type} не поддерживается")),
    }
    Ok(out)
}

// ---- тесты ------------------------------------------------------------------

#[cfg(test)]
mod gguf_tests {
    use super::*;

    /// Живая модель: какие kv-ключи реально лежат в GGUF и как они
    /// читаются через suffix-поиск (DI_NET_GGUF).
    #[test]
    fn kv_probe_gated() {
        let Some(path) = std::env::var("DI_NET_GGUF").ok() else {
            return;
        };
        let m = GgufModel::open(std::path::Path::new(&path)).expect("gguf");
        let keys: Vec<&String> = m.kv.keys().collect();
        for k in keys {
            eprintln!("kv: {k} = {:?}", m.kv.get(k).unwrap());
        }
        eprintln!(
            "num(intermediate_size)={:?} num(block_count)={:?} eps={:?}",
            m.num("intermediate_size"),
            m.num("block_count"),
            m.f32val("layer_norm_rms_epsilon")
        );
        eprintln!("всего kv: {}", m.kv.len());
    }

    /// Синтетический GGUF (строка-ключ, f32-kv, тензоры F32/Q8_0) — читается
    /// назад: имена, размеры, de-квант до известных величин.
    #[test]
    fn synthetic_gguf_roundtrip() {
        fn pu64(f: &mut Vec<u8>, v: u64) {
            f.extend_from_slice(&v.to_le_bytes());
        }
        fn pu32(f: &mut Vec<u8>, v: u32) {
            f.extend_from_slice(&v.to_le_bytes());
        }
        fn pf32(f: &mut Vec<u8>, v: f32) {
            f.extend_from_slice(&v.to_le_bytes());
        }
        fn pstr(f: &mut Vec<u8>, v: &[u8]) {
            f.extend_from_slice(v);
        }
        fn add_str(f: &mut Vec<u8>, key: &[u8], val: &[u8]) {
            pu64(f, key.len() as u64);
            pstr(f, key);
            pu32(f, 8); // STRING
            pu64(f, val.len() as u64);
            pstr(f, val);
        }
        fn add_f32(f: &mut Vec<u8>, key: &[u8], v: f32) {
            pu64(f, key.len() as u64);
            pstr(f, key);
            pu32(f, 6); // FLOAT32
            pf32(f, v);
        }

        let mut f: Vec<u8> = Vec::new();
        pstr(&mut f, b"GGUF");
        pu32(&mut f, 3); // версия
        pu64(&mut f, 2); // тензор count
        pu64(&mut f, 3); // kv count
        add_str(&mut f, b"general.architecture", b"qwen2");
        add_f32(&mut f, b"qwen2.attention.layer_norm_rms_epsilon", 1e-5);
        add_f32(&mut f, b"general.alignment", 32.0);
        // тензор 1: F32 [4,5]; тензор 2: Q8_0 [64]
        let ti = |f: &mut Vec<u8>, name: &[u8], ne: &[u64], t: u32, off: u64| {
            pu64(f, name.len() as u64);
            pstr(f, name);
            pu32(f, ne.len() as u32);
            for d in ne {
                pu64(f, *d);
            }
            pu32(f, t);
            pu64(f, off);
        };
        let off_a = 0u64;
        let off_b = (4 * 5 * 4 + 31) / 32 * 32;
        ti(&mut f, b"output_norm.weight", &[4, 5], 0, off_a);
        ti(&mut f, b"output.weight", &[64], 8, off_b);
        // выравнивание до 32
        let hdr = f.len();
        let pad = (32 - hdr % 32) % 32;
        f.extend(std::iter::repeat(0u8).take(pad));
        // данные F32 (5 строк по 4): значения 0..19
        for i in 0..20usize {
            pf32(&mut f, i as f32);
        }
        // паддинг до off_b (96) — выравнивание GGUF между данными тензоров
        let pad2 = off_b as usize - 20 * 4;
        f.extend(std::iter::repeat(0u8).take(pad2));
        // данные Q8_0: 2 блока (f16 d + 32×i8)
        let mut qb: Vec<u8> = Vec::new();
        for blo in 0..2u16 {
            qb.extend_from_slice(&half_bits(0.125f32).to_le_bytes());
            for j in 0..32u16 {
                let q = (blo * 32 + j) as i8;
                qb.extend_from_slice(&q.to_le_bytes());
            }
        }
        f.extend_from_slice(&qb);

        let tmp = std::env::temp_dir().join(format!("dinet-gguf-test-{}.bin", std::process::id()));
        std::fs::write(&tmp, &f).unwrap();
        let m = GgufModel::open(&tmp).unwrap();
        assert_eq!(m.architecture().as_deref(), Some("qwen2"));
        assert_eq!(m.f32val("layer_norm_rms_epsilon"), Some(1e-5));
        let norm = m.read_f32("output_norm.weight").unwrap();
        assert_eq!(norm.data.len(), 20);
        assert_eq!(norm.row, 4);
        assert_eq!(norm.rows, 5);
        assert_eq!(norm.data[7], 7.0);
        // Q8_0: d=0.125, q=0..63 → 0.125*q
        let q = m.read_f32("output.weight").unwrap();
        assert_eq!(q.data.len(), 64);
        assert_eq!(q.data[0], 0.0);
        assert_eq!(q.data[33], 0.125 * 33.0);
        assert_eq!(q.data[63], 0.125 * 63.0);
        let _ = std::fs::remove_file(&tmp);
    }

    /// f16_to_f32: известные значения.
    #[test]
    fn f16_decode_known() {
        assert_eq!(f16_to_f32(0), 0.0);
        assert_eq!(f16_to_f32(0x3c00), 1.0);
        assert_eq!(f16_to_f32(0x4000), 2.0);
        assert_eq!(f16_to_f32(0x8000), -0.0);
        assert_eq!(f16_to_f32(0xc000), -2.0);
        assert_eq!(f16_to_f32(0x0001), 5.9604644775390625e-8); // поднормаль
        let inf = f16_to_f32(0x7c00);
        assert_eq!(inf, f32::INFINITY);
    }

    pub fn half_bits(v: f32) -> u16 {
        // простое преобразование f32→f16 (для теста; точность ±ulp)
        let bits = v.to_bits();
        let s = ((bits >> 31) & 1) as u16;
        let e = ((bits >> 23) & 0xff) as i32;
        let m = bits & 0x7f_ffff;
        if e == 0xff + 0 {
            return (s << 15) | 0x7c00;
        }
        // нормаль: смещение 15-127 = -112
        let e16 = (e - 127 + 15) as i64;
        if e16 <= 0 {
            return s << 15;
        }
        if e16 >= 31 {
            return ((s << 15) | 0x7c00) as u16;
        }
        let m16 = (m >> 13) as u16;
        (s << 15) | ((e16 as u16) << 10) | m16
    }
}
