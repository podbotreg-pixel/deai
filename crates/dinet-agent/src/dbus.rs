//! Рукописный D-Bus-клиент (M2, шаг «AT-SPI: Linux GUI»): соединение по
//! UnixSocket (SASL EXTERNAL), маршалинг/разбор сообщений — без внешних
//! зависимостей, в духе проекта (как WS-клиент CDP в browser.rs).
//!
//! Реализовано ровно то, что нужно AT-SPI: method call → method return /
//! error; типы y b u i d s o g v a{..} (..); маленький эндриан ('l').
//! Сигналы приходят постоянно (шина a11y шумит) — recv пропускает их,
//! отвечая только на свой serial. Таймаут чтения — 5 с (не виснем).

use std::io::{Read, Write};
use std::os::unix::net::UnixStream;
use std::time::Duration;

/// Значение тела сообщения (типы, нужные AT-SPI).
#[derive(Debug, Clone, PartialEq)]
pub enum Val {
    Byte(u8),
    Bool(bool),
    U32(u32),
    I32(i32),
    F64(f64),
    Str(String),
    Obj(String),
    /// Сигнатура g (байт-длина): типы полей заголовка и подписи вариантов.
    Sig(String),
    /// Массив: сигнатура элемента (например "(so)", "s", "{ss}") + значения.
    Arr(String, Vec<Val>),
    /// Структура (…) или пара словаря {…}: значения по порядку.
    Struct(Vec<Val>),
    /// Вариант v: значение одного из базовых типов.
    Var(Box<Val>),
}

impl Val {
    pub fn as_str(&self) -> Option<&str> {
        match self {
            Val::Str(s) | Val::Obj(s) | Val::Sig(s) => Some(s),
            _ => None,
        }
    }
    pub fn as_u32(&self) -> Option<u32> {
        match self {
            Val::U32(v) => Some(*v),
            Val::Byte(v) => Some(*v as u32),
            Val::I32(v) if *v >= 0 => Some(*v as u32),
            _ => None,
        }
    }
    pub fn as_i32(&self) -> Option<i32> {
        match self {
            Val::I32(v) => Some(*v),
            Val::U32(v) if *v <= i32::MAX as u32 => Some(*v as i32),
            _ => None,
        }
    }
}

/// Выравнивание простого типа по сигнатурному символу (спека D-Bus).
fn align_of(c: char) -> usize {
    match c {
        'y' | 'g' | 'v' => 1, // v: сигнатура внутри сама выровняет значение
        'b' | 'u' | 'i' => 4,
        'x' | 't' | 'd' => 8,
        's' | 'o' => 4,
        'a' => 4, // выравнивание длины
        '(' | '{' => 8,
        _ => 1,
    }
}

/// Буфер записи с абсолютным выравниванием.
struct Enc {
    buf: Vec<u8>,
}

impl Enc {
    fn new() -> Self {
        Enc { buf: Vec::new() }
    }
    fn pad(&mut self, a: usize) {
        while self.buf.len() % a != 0 {
            self.buf.push(0);
        }
    }
    fn byte(&mut self, v: u8) {
        self.buf.push(v);
    }
    fn u32(&mut self, v: u32) {
        self.pad(4);
        self.buf.extend_from_slice(&v.to_le_bytes());
    }
    fn f64(&mut self, v: f64) {
        self.pad(8);
        self.buf.extend_from_slice(&v.to_le_bytes());
    }
    fn string(&mut self, s: &str, wide: bool) {
        // s/o: u32-длина + байты + NUL; g: u8-длина + байты + NUL
        if wide {
            self.u32(s.len() as u32);
        } else {
            self.pad(1);
            self.buf.push(s.len() as u8);
        }
        self.buf.extend_from_slice(s.as_bytes());
        self.buf.push(0);
    }
    /// Значение по сигнатурному символу (строки/объекты — с выравниванием 4).
    fn value(&mut self, sig: &str, v: &Val) {
        let mut it = sig.chars();
        let c = it.next().unwrap_or('y');
        match (c, v) {
            ('y', Val::Byte(b)) => {
                self.pad(1);
                self.buf.push(*b);
            }
            ('b', Val::Bool(b)) => {
                self.u32(if *b { 1 } else { 0 });
            }
            ('u', Val::U32(n)) => self.u32(*n),
            ('i', Val::I32(n)) => self.u32(*n as u32),
            ('d', Val::F64(x)) => self.f64(*x),
            ('s', Val::Str(s)) => {
                self.pad(4);
                self.string(s, true);
            }
            ('o', Val::Obj(p)) => {
                self.pad(4);
                self.string(p, true);
            }
            ('g', Val::Str(s)) | ('g', Val::Sig(s)) => self.string(s, false),
            ('v', Val::Var(inner)) => {
                let isig = sig_of(inner);
                self.string(&isig, false);
                self.value(&isig, inner);
            }
            ('a', Val::Arr(elem_sig, items)) => {
                self.pad(4);
                let len_pos = self.buf.len();
                self.buf.extend_from_slice(&0u32.to_le_bytes());
                // элементы начинаются на выравнивании элемента; длина
                // массива считает байты ИМЕННО от выровненного начала
                if !items.is_empty() {
                    let a = align_of(elem_sig.chars().next().unwrap_or('y'));
                    self.pad(a);
                }
                let content_start = self.buf.len();
                for item in items {
                    self.value(elem_sig, item);
                }
                let len = (self.buf.len() - content_start) as u32;
                self.buf[len_pos..len_pos + 4].copy_from_slice(&len.to_le_bytes());
            }
            ('(', Val::Struct(vals)) | ('{', Val::Struct(vals)) => {
                self.pad(8);
                let inner = &sig[1..sig.len() - 1];
                let mut rest = inner;
                for v in vals {
                    let (s, tail) = split_sig(rest);
                    self.value(s, v);
                    rest = tail;
                }
            }
            _ => panic!("маршалинг: сигнатура {sig:?} не бьётся со значением {v:?}"),
        }
    }
}

/// Сигнатура значения (для вариантов внутри).
fn sig_of(v: &Val) -> String {
    match v {
        Val::Byte(_) => "y".into(),
        Val::Bool(_) => "b".into(),
        Val::U32(_) => "u".into(),
        Val::I32(_) => "i".into(),
        Val::F64(_) => "d".into(),
        Val::Str(_) => "s".into(),
        Val::Obj(_) => "o".into(),
        Val::Sig(_) => "g".into(),
        Val::Var(inner) => sig_of(inner),
        Val::Arr(elem, _) => format!("a{elem}"),
        Val::Struct(vals) => {
            let inner: String = vals.iter().map(sig_of).collect();
            format!("({inner})")
        }
    }
}

/// Отрезать первую полную единицу сигнатуры: "(so)" целиком, "a{ss}"
/// целиком (префиксы a/v тянут следующий юнит), "s" — один символ.
fn split_sig(sig: &str) -> (&str, &str) {
    let b = sig.as_bytes();
    if b.is_empty() {
        return ("", "");
    }
    match b[0] {
        b'(' | b'{' => {
            let mut depth = 0usize;
            for (i, &c) in b.iter().enumerate() {
                match c {
                    b'(' | b'{' => depth += 1,
                    b')' | b'}' => {
                        depth -= 1;
                        if depth == 0 {
                            return (&sig[..=i], &sig[i + 1..]);
                        }
                    }
                    _ => {}
                }
            }
            (sig, "")
        }
        // a тянет следующий полный юнит (массив составного типа);
        // v — самостоятельный тип: значение несёт свою сигнатуру
        b'a' => {
            let (next, rest) = split_sig(&sig[1..]);
            (&sig[..1 + next.len()], rest)
        }
        _ => (&sig[..1], &sig[1..]),
    }
}

/// Кадр сообщения: заголовок + поля + тело.
pub struct Frame {
    pub kind: u8,
    pub serial: u32,
    pub reply_serial: Option<u32>,
    pub error_name: Option<String>,
    pub member: Option<String>,
    pub signature: Option<String>,
    pub body: Vec<u8>,
}

/// Собрать METHOD_CALL. Тело — по сигнатуре `body_sig` из значений.
pub fn method_call(
    dest: &str,
    path: &str,
    iface: &str,
    member: &str,
    body_sig: &str,
    body: &[Val],
    serial: u32,
) -> Vec<u8> {
    let mut fields: Vec<(u8, Val)> = vec![
        (1, Val::Obj(path.to_string())),
        (6, Val::Str(dest.to_string())),
        (3, Val::Str(member.to_string())),
    ];
    if !iface.is_empty() {
        fields.push((2, Val::Str(iface.to_string())));
    }
    if !body_sig.is_empty() {
        // тип поля SIGNATURE — сигнатура (g): демон проверяет типы полей
        fields.push((8, Val::Sig(body_sig.to_string())));
    }
    // специка D-Bus: поля в заголовке — по возрастанию кода
    fields.sort_by_key(|(code, _)| *code);
    let mut e = Enc::new();
    e.byte(b'l'); // little-endian
    e.byte(1); // METHOD_CALL
    e.byte(0); // флаги
    e.byte(1); // протокол
    // длины/serial встанут на 4-байтовом выравнивании сами
    e.u32(0); // body_len — заплатим
    e.u32(serial);
    // поля a(yv): длина массива по своему выравниванию (4), содержимое —
    // по выравниванию элемента (8) — считает сама запись массива
    let fv: Vec<Val> = fields
        .into_iter()
        .map(|(code, v)| Val::Struct(vec![Val::Byte(code), Val::Var(Box::new(v))]))
        .collect();
    e.value("a(yv)", &Val::Arr("(yv)".to_string(), fv));
    e.pad(8); // тело — по 8
    let header_len = e.buf.len();
    let mut rest = body_sig;
    let mut idx = 0usize;
    while !rest.is_empty() {
        let (unit, tail) = split_sig(rest);
        let v = body.get(idx).unwrap_or_else(|| panic!("маршалинг: нет значения для {unit}"));
        e.value(unit, v);
        idx += 1;
        rest = tail;
    }
    let body_len = (e.buf.len() - header_len) as u32;
    e.buf[4..8].copy_from_slice(&body_len.to_le_bytes());
    e.buf
}

/// Прочитать ровно один кадр из потока.
fn read_frame(stream: &mut UnixStream) -> Result<Frame, String> {
    let mut head = [0u8; 16];
    stream
        .read_exact(&mut head)
        .map_err(|e| format!("dbus: обрыв шины ({e})"))?;
    if head[0] != b'l' {
        return Err("dbus: неожиданный порядок байтов".into());
    }
    let kind = head[1];
    let body_len = u32::from_le_bytes([head[4], head[5], head[6], head[7]]) as usize;
    let serial = u32::from_le_bytes([head[8], head[9], head[10], head[11]]);
    let fields_len = u32::from_le_bytes([head[12], head[13], head[14], head[15]]) as usize;
    let mut rest = vec![0u8; fields_len];
    stream
        .read_exact(&mut rest)
        .map_err(|e| format!("dbus: обрыв полей ({e})"))?;
    // выравнивание тела: (16 + fields_len) до 8
    let skip = (8 - ((16 + fields_len) % 8)) % 8;
    let mut pad = vec![0u8; skip];
    stream
        .read_exact(&mut pad)
        .map_err(|e| format!("dbus: обрыв паддинга ({e})"))?;
    let mut body = vec![0u8; body_len];
    stream
        .read_exact(&mut body)
        .map_err(|e| format!("dbus: обрыв тела ({e})"))?;
    // разбор полей: содержимое массива = подряд структуры (yv)
    let mut fields: Vec<(u8, Val)> = Vec::new();
    {
        let mut q = 0usize;
        while q < fields_len {
            let (v, np) = parse_value(&rest, q, "(yv)")?;
            if let Val::Struct(kv) = v {
                let code = kv.first().and_then(|x| x.as_u32()).unwrap_or(0) as u8;
                if let Some(Val::Var(inner)) = kv.get(1) {
                    fields.push((code, (**inner).clone()));
                }
            }
            q = np;
        }
    }
    let mut reply_serial = None;
    let mut error_name = None;
    let mut member = None;
    let mut signature = None;
    for (code, v) in fields {
        match code {
            3 => member = v.as_str().map(|s| s.to_string()),
            4 => error_name = v.as_str().map(|s| s.to_string()),
            5 => reply_serial = v.as_u32(),
            8 => signature = v.as_str().map(|s| s.to_string()),
            _ => {}
        }
    }
    Ok(Frame { kind, serial, reply_serial, error_name, member, signature, body })
}

/// Разбор тела по сигнатуре.
pub fn parse_body(body: &[u8], sig: &str) -> Result<Vec<Val>, String> {
    let mut vals = Vec::new();
    let mut rest = sig;
    let mut pos = 0usize;
    while !rest.is_empty() {
        if pos >= body.len() {
            return Err(format!("dbus: тело кончилось на юните {rest:?} (pos {pos})"));
        }
        let (unit, tail) = split_sig(rest);
        let (v, p) = parse_value(body, pos, unit)?;
        vals.push(v);
        pos = p;
        rest = tail;
    }
    Ok(vals)
}

fn parse_value(body: &[u8], pos: usize, sig: &str) -> Result<(Val, usize), String> {
    let c = sig.chars().next().ok_or("dbus: пустая сигнатура")?;
    let al = align_of(c);
    let mut p = pos;
    if al > 1 {
        p += (al - p % al) % al;
        if p > body.len() {
            return Err(format!("dbus: тело короче выравнивания (pos {pos}, sig {sig})"));
        }
    }
    let rd_u32 = |p: usize| -> Result<u32, String> {
        if p + 4 > body.len() {
            return Err(format!("dbus: не хватает u32 (pos {p})"));
        }
        Ok(u32::from_le_bytes([body[p], body[p + 1], body[p + 2], body[p + 3]]))
    };
    match c {
        'y' => Ok((Val::Byte(body[p]), p + 1)),
        'b' => {
            let v = rd_u32(p)?;
            Ok((Val::Bool(v != 0), p + 4))
        }
        'u' => {
            let v = rd_u32(p)?;
            Ok((Val::U32(v), p + 4))
        }
        'i' => {
            let v = rd_u32(p)? as i32;
            Ok((Val::I32(v), p + 4))
        }
        'd' => {
            if p + 8 > body.len() {
                return Err(format!("dbus: не хватает f64 (pos {p})"));
            }
            let mut b8 = [0u8; 8];
            b8.copy_from_slice(&body[p..p + 8]);
            Ok((Val::F64(f64::from_le_bytes(b8)), p + 8))
        }
        's' | 'o' => {
            let len = rd_u32(p)? as usize;
            let s = p + 4;
            if s + len + 1 > body.len() {
                return Err(format!("dbus: не хватает строки (pos {p}, len {len})"));
            }
            let text = std::str::from_utf8(&body[s..s + len])
                .map_err(|e| format!("dbus: не utf8 ({e})"))?
                .to_string();
            let val = if c == 's' { Val::Str(text) } else { Val::Obj(text) };
            Ok((val, s + len + 1))
        }
        'g' => {
            let len = body[p] as usize;
            let s = p + 1;
            if s + len + 1 > body.len() {
                return Err(format!("dbus: не хватает сигнатуры (pos {p}, len {len})"));
            }
            let text = std::str::from_utf8(&body[s..s + len])
                .map_err(|e| format!("dbus: не utf8 ({e})"))?
                .to_string();
            Ok((Val::Sig(text), s + len + 1))
        }
        'v' => {
            let (sv, sp) = parse_value(body, p, "g")?;
            let sig = sv.as_str().ok_or("dbus: вариант без сигнатуры")?.to_string();
            let (v, np) = parse_value(body, sp, &sig)?;
            Ok((Val::Var(Box::new(v)), np))
        }
        'a' => {
            // длина массива, затем элементы по сигнатуре элемента
            let len = rd_u32(p)? as usize;
            let elem_sig = &sig[1..];
            let (first_unit, _) = split_sig(elem_sig);
            let mut items = Vec::new();
            let mut q = p + 4;
            if !first_unit.is_empty() {
                q += (align_of(first_unit.chars().next().unwrap()) - q % align_of(first_unit.chars().next().unwrap())) % align_of(first_unit.chars().next().unwrap());
            }
            let end = q + len;
            while q < end {
                let (unit, _) = split_sig(elem_sig);
                let (v, np) = parse_value(body, q, unit)?;
                items.push(v);
                q = np;
            }
            Ok((Val::Arr(elem_sig.to_string(), items), end))
        }
        '(' => {
            let inner = &sig[1..sig.len() - 1];
            p += (8 - p % 8) % 8;
            let mut vals = Vec::new();
            let mut rest = inner;
            while !rest.is_empty() {
                let (unit, tail) = split_sig(rest);
                let (v, np) = parse_value(body, p, unit)?;
                vals.push(v);
                p = np;
                rest = tail;
            }
            Ok((Val::Struct(vals), p))
        }
        '{' => {
            let inner = &sig[1..sig.len() - 1];
            p += (8 - p % 8) % 8;
            let (key_sig, after) = split_sig(inner);
            let (val_sig, _) = split_sig(after);
            let (k, np) = parse_value(body, p, key_sig)?;
            let (v, np2) = parse_value(body, np, val_sig)?;
            Ok((Val::Struct(vec![k, v]), np2))
        }
        other => Err(format!("dbus: неизвестный тип {other:?}")),
    }
}

/// Соединение с шиной: SASL EXTERNAL + method call/reply.
pub struct Conn {
    stream: UnixStream,
    next_serial: u32,
}

impl Conn {
    /// Открыть по адресу unix:path=… и пройти SASL EXTERNAL.
    pub fn open(addr: &str) -> Result<Conn, String> {
        let path = addr
            .strip_prefix("unix:path=")
            .ok_or_else(|| format!("dbus: адрес не unix:path= ({addr:?})"))?
            .split(',')
            .next()
            .unwrap_or_default()
            .to_string();
        let stream = UnixStream::connect(&path)
            .map_err(|e| format!("dbus: не подключился к {path} ({e})"))?;
        stream.set_read_timeout(Some(Duration::from_secs(5))).ok();
        stream.set_write_timeout(Some(Duration::from_secs(5))).ok();
        let mut c = Conn { stream, next_serial: 1 };
        c.handshake()?;
        Ok(c)
    }

    fn handshake(&mut self) -> Result<(), String> {
        // идентификатор = ASCII-десятичный uid, закодированный hex
        let uid = read_uid()?;
        let hex_uid: String = uid.bytes().map(|b| format!("{b:02x}")).collect();
        let welcome = format!("\0AUTH EXTERNAL {hex_uid}\r\n");
        self.stream
            .write_all(welcome.as_bytes())
            .map_err(|e| format!("dbus: SASL запись ({e})"))?;
        let mut buf = [0u8; 256];
        let n = self
            .stream
            .read(&mut buf)
            .map_err(|e| format!("dbus: SASL чтение ({e})"))?;
        let reply = String::from_utf8_lossy(&buf[..n]);
        if !reply.starts_with("OK ") {
            let trimmed = reply.trim().to_string();
            return Err(format!("dbus: SASL отверг ({trimmed})"));
        }
        self.stream
            .write_all(b"BEGIN\r\n")
            .map_err(|e| format!("dbus: SASL begin ({e})"))?;
        // Hello обязателен на шине до всяких других сообщений; ответ
        // (уникальное имя) не нужен вызывающему, ошибка честная при разрыве
        let _ = self.call(
            "org.freedesktop.DBus",
            "/org/freedesktop/DBus",
            "org.freedesktop.DBus",
            "Hello",
            "",
            &[],
        );
        Ok(())
    }

    /// Вызов метода; сигналы пропускаются, ошибки шины — Err с именем.
    pub fn call(
        &mut self,
        dest: &str,
        path: &str,
        iface: &str,
        member: &str,
        body_sig: &str,
        body: &[Val],
    ) -> Result<Vec<Val>, String> {
        let serial = self.next_serial;
        self.next_serial += 1;
        let frame = method_call(dest, path, iface, member, body_sig, body, serial);
        self.stream
            .write_all(&frame)
            .map_err(|e| format!("dbus: запись вызова ({e})"))?;
        loop {
            let reply = read_frame(&mut self.stream)?;
            if reply.reply_serial != Some(serial) {
                continue; // чужой/сигнал — пропустить
            }
            if reply.kind == 2 {
                let sig = reply.signature.clone().unwrap_or_default();
                return parse_body(&reply.body, &sig);
            }
            if reply.kind == 3 {
                let name = reply.error_name.clone().unwrap_or_default();
                let msg = parse_body(&reply.body, "s")
                    .ok()
                    .and_then(|v| v.into_iter().next())
                    .and_then(|v| v.as_str().map(|s| s.to_string()))
                    .unwrap_or_default();
                return Err(format!("dbus: {name}: {msg}"));
            }
        }
    }
}

/// UID процесса (для SASL EXTERNAL) — из /proc/self/status, без libc.
fn read_uid() -> Result<String, String> {
    let status = std::fs::read_to_string("/proc/self/status")
        .map_err(|e| format!("dbus: нет /proc/self/status ({e})"))?;
    for line in status.lines() {
        if let Some(rest) = line.strip_prefix("Uid:") {
            let uid = rest.split_whitespace().next().unwrap_or("");
            if !uid.is_empty() {
                return Ok(uid.to_string());
            }
        }
    }
    Err("dbus: UID не найден в /proc/self/status".into())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn roundtrip_basic_types() {
        // s, o, u, i, b, v(g-вариант), a(so), a{ss}, (iiii)
        let body_sig = "souibva(so)a{ss}(iiii)";
        let vals = vec![
            Val::Str("привет".into()),
            Val::Obj("/org/a11y/x".into()),
            Val::U32(42),
            Val::I32(-7),
            Val::Bool(true),
            Val::Var(Box::new(Val::Str("внутри".into()))),
            Val::Arr(
                "(so)".into(),
                vec![Val::Struct(vec![
                    Val::Str(":1.2".into()),
                    Val::Obj("/p1".into()),
                ])],
            ),
            Val::Arr(
                "{ss}".into(),
                vec![Val::Struct(vec![
                    Val::Str("ключ".into()),
                    Val::Str("значение".into()),
                ])],
            ),
            Val::Struct(vec![Val::I32(1), Val::I32(2), Val::I32(3), Val::I32(4)]),
        ];
        // маршалинг в несколько заходов: собрать через Enc как тело вызова
        let mut e = Enc::new();
        let mut rest = body_sig;
        for v in vals.iter() {
            let (unit, tail) = split_sig(rest);
            e.value(unit, v);
            rest = tail;
        }
        let back = parse_body(&e.buf, body_sig).expect("разбор");
        assert_eq!(back, vals, "круг маршалинг→разбор бит-в-бит");
    }

    #[test]
    fn alignment_rules_hold() {
        // строка после u32: выравнивание 4; массив структур выравнивает
        // элемент на 8
        let vals = vec![
            Val::U32(1),
            Val::Str("длинная строка ровно".into()),
            Val::Arr(
                "(so)".into(),
                vec![
                    Val::Struct(vec![Val::Str("b".into()), Val::Obj("/a".into())]),
                    Val::Struct(vec![Val::Str("d".into()), Val::Obj("/c".into())]),
                ],
            ),
        ];
        let mut e = Enc::new();
        let mut rest = "usa(so)";
        for v in vals.iter() {
            let (unit, tail) = split_sig(rest);
            e.value(unit, v);
            rest = tail;
        }
        let back = parse_body(&e.buf, "usa(so)").expect("разбор");
        assert_eq!(back, vals);
        // тело выровнено: u32 на 0..4, строка на 4
        assert_eq!(back[1].as_str(), Some("длинная строка ровно"));
    }

    #[test]
    fn call_frame_header_wellformed() {
        let frame = method_call(
            ":1.5",
            "/org/a11y/atspi/accessible/root",
            "org.a11y.atspi.Accessible",
            "GetChildren",
            "",
            &[],
            7,
        );
        assert_eq!(frame[0], b'l');
        assert_eq!(frame[1], 1); // METHOD_CALL
        assert_eq!(frame[3], 1); // протокол
        let body_len = u32::from_le_bytes([frame[4], frame[5], frame[6], frame[7]]);
        assert_eq!(body_len, 0);
        let serial = u32::from_le_bytes([frame[8], frame[9], frame[10], frame[11]]);
        assert_eq!(serial, 7);
        // кадр должен разбираться read_frame-логикой: поля содержат PATH/DEST/MEMBER
        let mut with_body = frame.clone();
        with_body.extend_from_slice(&[]); // тело пусто
        // разбор полей: (16 + fields_len) — по кадру
        let fields_len =
            u32::from_le_bytes([frame[12], frame[13], frame[14], frame[15]]) as usize;
        let skip = (8 - ((16 + fields_len) % 8)) % 8;
        let fields = &frame[16..16 + fields_len];
        let mut items: Vec<Val> = Vec::new();
        let mut q = 0usize;
        while q < fields.len() {
            let (v, np) = parse_value(fields, q, "(yv)").expect("поля");
            items.push(v);
            q = np;
        }
        let mut dest = None;
        let mut member = None;
        for it in &items {
            if let Val::Struct(kv) = it {
                let code = kv[0].as_u32().unwrap_or(0);
                if let Val::Var(inner) = &kv[1] {
                    match code {
                        3 => member = inner.as_str().map(|s| s.to_string()),
                        6 => dest = inner.as_str().map(|s| s.to_string()),
                        _ => {}
                    }
                }
            }
        }
        assert_eq!(dest.as_deref(), Some(":1.5"));
        assert_eq!(member.as_deref(), Some("GetChildren"));
        // тело начинается по границе 8 после полей
        assert_eq!((16 + fields_len + skip) % 8, 0);
    }

    #[test]
    fn split_sig_slices_complete_units() {
        let (a, b) = split_sig("(so)i");
        assert_eq!(a, "(so)");
        assert_eq!(b, "i");
        let (a, b) = split_sig("a{ss}s");
        assert_eq!(a, "a{ss}");
        assert_eq!(b, "s");
        let (a, b) = split_sig("ss");
        assert_eq!(a, "s");
        assert_eq!(b, "s");
        let (a, b) = split_sig("(i(iii))u");
        assert_eq!(a, "(i(iii))");
        assert_eq!(b, "u");
    }

    #[test]
    fn uid_parsed_from_proc() {
        let uid = read_uid().expect("uid");
        assert!(uid.parse::<u32>().is_ok(), "uid численный: {uid}");
    }
}
