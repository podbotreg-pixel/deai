//! Браузер-CDP клиент агента (M2, шаг «браузер-CDP/UIA-серверы»):
//! computer-use поверх chromium DevTools-протокола. WebSocket-клиент,
//! SHA-1/base64-рукопожатие и кодирование кадров написаны вручную —
//! без внешних зависимостей, в духе проекта.
//!
//! Политика: подключение к уже запущенному chromium (bin browser-cua
//! поднимает его сам); вкладка одна. URL разрешены только http/https/data
//! (file:// умеет читать локальные диски — обход джейла); URL прогоняется
//! через анти-фишинг ДО соединения. Браузерные инструменты требуют
//! уровня NETWORK и пишутся в аудит как все.

use crate::json::{self, Json};
use std::io::{Read, Write};
use std::net::TcpStream;
use std::time::Duration;

/// Политика URL: только http/https/data. file:// может читать локальные
/// файлы (обход джейла), about:/chrome: — служебные страницы браузера.
pub fn url_allowed(url: &str) -> Result<(), String> {
    let u = url.trim();
    if u.is_empty() {
        return Err("нужен url".into());
    }
    if u.starts_with("http://") || u.starts_with("https://") || u.starts_with("data:") {
        Ok(())
    } else {
        Err(format!(
            "обход джейла: схема URL не разрешена ({u:?}; разрешены http/https/data)"
        ))
    }
}

// ---------- SHA-1 (для проверки Sec-WebSocket-Accept) ----------

pub fn sha1(data: &[u8]) -> [u8; 20] {
    let mut h: [u32; 5] = [0x67452301, 0xEFCDAB89, 0x98BADCFE, 0x10325476, 0xC3D2E1F0];
    let ml = (data.len() as u64).wrapping_mul(8);
    let mut msg = data.to_vec();
    msg.push(0x80);
    while msg.len() % 64 != 56 {
        msg.push(0);
    }
    msg.extend_from_slice(&ml.to_be_bytes());
    for chunk in msg.chunks(64) {
        let mut w = [0u32; 80];
        for (i, wc) in chunk.chunks(4).enumerate() {
            w[i] = u32::from_be_bytes([wc[0], wc[1], wc[2], wc[3]]);
        }
        for i in 16..80 {
            w[i] = (w[i - 3] ^ w[i - 8] ^ w[i - 14] ^ w[i - 16]).rotate_left(1);
        }
        let (mut a, mut b, mut c, mut d, mut e) = (h[0], h[1], h[2], h[3], h[4]);
        for (i, &wi) in w.iter().enumerate() {
            let (f, k) = match i {
                0..=19 => ((b & c) | ((!b) & d), 0x5A827999u32),
                20..=39 => (b ^ c ^ d, 0x6ED9EBA1),
                40..=59 => ((b & c) | (b & d) | (c & d), 0x8F1BBCDC),
                _ => (b ^ c ^ d, 0xCA62C1D6),
            };
            let t = a
                .rotate_left(5)
                .wrapping_add(f)
                .wrapping_add(e)
                .wrapping_add(k)
                .wrapping_add(wi);
            e = d;
            d = c;
            c = b.rotate_left(30);
            b = a;
            a = t;
        }
        h[0] = h[0].wrapping_add(a);
        h[1] = h[1].wrapping_add(b);
        h[2] = h[2].wrapping_add(c);
        h[3] = h[3].wrapping_add(d);
        h[4] = h[4].wrapping_add(e);
    }
    let mut out = [0u8; 20];
    for (i, x) in h.iter().enumerate() {
        out[i * 4..i * 4 + 4].copy_from_slice(&x.to_be_bytes());
    }
    out
}

const B64: &[u8; 64] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";

pub fn base64(data: &[u8]) -> String {
    let mut o = String::new();
    for chunk in data.chunks(3) {
        let b = [
            chunk[0],
            chunk.get(1).copied().unwrap_or(0),
            chunk.get(2).copied().unwrap_or(0),
        ];
        let n = ((b[0] as u32) << 16) | ((b[1] as u32) << 8) | (b[2] as u32);
        o.push(B64[(n >> 18) as usize & 63] as char);
        o.push(B64[(n >> 12) as usize & 63] as char);
        o.push(if chunk.len() > 1 { B64[(n >> 6) as usize & 63] as char } else { '=' });
        o.push(if chunk.len() > 2 { B64[n as usize & 63] as char } else { '=' });
    }
    o
}

/// Sec-WebSocket-Accept из Sec-WebSocket-Key (RFC 6455 §1.3).
pub fn accept_of(key: &str) -> String {
    base64(&sha1(format!("{key}258EAFA5-E914-47DA-95CA-C5AB0DC85B11").as_bytes()))
}

// ---------- ГСЧ для масок/ключей (SplitMix64 от времени+pid) ----------

pub struct Rng(u64);

impl Rng {
    pub fn new() -> Rng {
        let nanos = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_nanos() as u64)
            .unwrap_or(0);
        Rng((nanos ^ ((std::process::id() as u64) << 32)) | 1)
    }
    pub fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E3779B97F4A7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
        z ^ (z >> 31)
    }
}

// ---------- Кадры WebSocket ----------

/// Отправить кадр (клиент обязан маскировать payload, RFC 6455 §5.3).
fn write_frame(w: &mut impl Write, rng: &mut Rng, opcode: u8, payload: &[u8]) -> Result<(), String> {
    let mut f = Vec::with_capacity(payload.len() + 14);
    f.push(0x80 | opcode);
    let l = payload.len();
    if l < 126 {
        f.push(0x80 | l as u8);
    } else if l <= 65535 {
        f.push(0x80 | 126);
        f.extend_from_slice(&(l as u16).to_be_bytes());
    } else {
        f.push(0x80 | 127);
        f.extend_from_slice(&(l as u64).to_be_bytes());
    }
    let m = [
        rng.next() as u8,
        (rng.next() >> 8) as u8,
        (rng.next() >> 16) as u8,
        (rng.next() >> 24) as u8,
    ];
    f.extend_from_slice(&m);
    for (i, b) in payload.iter().enumerate() {
        f.push(b ^ m[i % 4]);
    }
    w.write_all(&f).and_then(|_| w.flush()).map_err(io_err)
}

fn io_err(e: std::io::Error) -> String {
    match e.kind() {
        std::io::ErrorKind::WouldBlock | std::io::ErrorKind::TimedOut => "ws: таймаут".into(),
        _ => format!("ws: {e}"),
    }
}

/// Прочитать ОДНО сообщение (склейка фрагментов; ping → pong; close → Err).
/// Читатель и писатель — один объект (как TcpStream: Read + Write).
fn read_message<S: Read + Write>(s: &mut S, rng: &mut Rng) -> Result<Option<Vec<u8>>, String> {
    let mut payload: Vec<u8> = Vec::new();
    loop {
        let mut h = [0u8; 2];
        s.read_exact(&mut h).map_err(io_err)?;
        let fin = h[0] & 0x80 != 0;
        let op = h[0] & 0x0F;
        let masked = h[1] & 0x80 != 0;
        let mut l = (h[1] & 0x7F) as usize;
        if l == 126 {
            let mut b = [0u8; 2];
            s.read_exact(&mut b).map_err(io_err)?;
            l = u16::from_be_bytes(b) as usize;
        } else if l == 127 {
            let mut b = [0u8; 8];
            s.read_exact(&mut b).map_err(io_err)?;
            l = u64::from_be_bytes(b) as usize;
        }
        if l > 16 * 1024 * 1024 {
            return Err("ws: кадр слишком велик".into());
        }
        let mask = if masked {
            let mut m = [0u8; 4];
            s.read_exact(&mut m).map_err(io_err)?;
            Some(m)
        } else {
            None
        };
        let mut p = vec![0u8; l];
        s.read_exact(&mut p).map_err(io_err)?;
        if let Some(m) = mask {
            for (i, b) in p.iter_mut().enumerate() {
                *b ^= m[i % 4];
            }
        }
        match op {
            0x1 | 0x2 | 0x0 => {
                payload.extend_from_slice(&p);
                if fin {
                    return Ok(Some(payload));
                }
            }
            0x9 => {
                // ping → pong (тот же payload, маска обязательна)
                write_frame(s, rng, 0xA, &p)?;
            }
            0xA => {} // pong игнорируем
            0x8 => return Err("ws: соединение закрыто сервером".into()),
            _ => {}
        }
    }
}

// ---------- Рукопожатие и HTTP к DevTools ----------

fn http_req(host: &str, method_path: &str) -> Result<String, String> {
    let mut s = TcpStream::connect(host).map_err(|e| format!("{host}: {e}"))?;
    s.set_read_timeout(Some(Duration::from_secs(5))).ok();
    let req = format!(
        "{method_path} HTTP/1.1\r\nHost: {host}\r\nConnection: close\r\nUser-Agent: dinet-agent\r\n\r\n"
    );
    s.write_all(req.as_bytes()).map_err(io_err)?;
    let mut buf = Vec::new();
    let mut tmp = [0u8; 4096];
    loop {
        match s.read(&mut tmp) {
            Ok(0) => break,
            Ok(n) => buf.extend_from_slice(&tmp[..n]),
            Err(ref e)
                if e.kind() == std::io::ErrorKind::WouldBlock
                    || e.kind() == std::io::ErrorKind::TimedOut =>
            {
                break
            }
            Err(e) => return Err(format!("http: {e}")),
        }
        if buf.len() > 1 << 20 {
            break;
        }
    }
    let text = String::from_utf8_lossy(&buf).to_string();
    let status = text.lines().next().unwrap_or_default().to_string();
    if !status.contains(" 200") && !status.contains(" 201") {
        return Err(format!("HTTP не 200 ({status})"));
    }
    let body = text.split("\r\n\r\n").nth(1).unwrap_or_default().to_string();
    Ok(body)
}

fn ws_connect(host: &str, path: &str) -> Result<TcpStream, String> {
    let mut s = TcpStream::connect(host).map_err(|e| format!("ws {host}: {e}"))?;
    s.set_read_timeout(Some(Duration::from_secs(10))).ok();
    let mut kb = [0u8; 16];
    let mut r = Rng::new();
    for b in kb.iter_mut() {
        *b = r.next() as u8;
    }
    let key = base64(&kb);
    let req = format!(
        "GET {path} HTTP/1.1\r\nHost: {host}\r\nUpgrade: websocket\r\nConnection: Upgrade\r\nSec-WebSocket-Key: {key}\r\nSec-WebSocket-Version: 13\r\n\r\n"
    );
    s.write_all(req.as_bytes()).map_err(io_err)?;
    // читаем заголовки до \r\n\r\n (побайтно — до конца секции)
    let mut head = Vec::new();
    let mut b = [0u8; 1];
    loop {
        let n = s.read(&mut b).map_err(io_err)?;
        if n == 0 {
            return Err("ws: соединение закрыто в рукопожатии".into());
        }
        head.push(b[0]);
        if head.ends_with(b"\r\n\r\n") {
            break;
        }
        if head.len() > 16 * 1024 {
            return Err("ws: заголовок слишком велик".into());
        }
    }
    let text = String::from_utf8_lossy(&head).to_string();
    let first = text.lines().next().unwrap_or_default().to_string();
    if !first.contains(" 101 ") {
        return Err(format!("ws: рукопожатие не 101 ({first})"));
    }
    let got = text
        .lines()
        .find_map(|l| {
            let (k, v) = l.split_once(':')?;
            if k.eq_ignore_ascii_case("sec-websocket-accept") {
                Some(v.trim().to_string())
            } else {
                None
            }
        })
        .ok_or("ws: нет Sec-WebSocket-Accept")?;
    if got != accept_of(&key) {
        return Err("ws: Sec-WebSocket-Accept не сошёлся".into());
    }
    Ok(s)
}

// ---------- CDP-клиент ----------

/// Подключение к одной вкладке запущенного chromium.
pub struct Browser {
    stream: TcpStream,
    next_id: u64,
    rng: Rng,
}

impl Browser {
    /// cdp: HTTP-адрес DevTools («127.0.0.1:9222» или «http://…»).
    pub fn connect(cdp: &str) -> Result<Browser, String> {
        let host = cdp
            .trim()
            .trim_start_matches("http://")
            .trim_start_matches("https://")
            .trim_end_matches('/')
            .to_string();
        // список таргетов: берём первый type=="page"
        let list_raw = http_req(&host, "GET /json")?;
        let list = json::parse(&list_raw)?;
        let mut ws: Option<String> = None;
        if let Json::Arr(items) = &list {
            for it in items {
                if it.get("type").and_then(|t| t.as_str()) == Some("page") {
                    ws = it
                        .get("webSocketDebuggerUrl")
                        .and_then(|t| t.as_str())
                        .map(|s| s.to_string());
                    if ws.is_some() {
                        break;
                    }
                }
            }
        }
        // нет вкладки — создать (новый chromium требует PUT, старый понимает GET)
        if ws.is_none() {
            let made = http_req(&host, "PUT /json/new?about:blank")
                .or_else(|_| http_req(&host, "GET /json/new?about:blank"))?;
            let j = json::parse(&made)?;
            ws = j
                .get("webSocketDebuggerUrl")
                .and_then(|t| t.as_str())
                .map(|s| s.to_string());
        }
        let Some(ws) = ws else {
            return Err(format!("{host}: не найден page-таргет"));
        };
        let rest = ws.trim_start_matches("ws://");
        let (h, path) = rest
            .split_once('/')
            .ok_or_else(|| format!("ws-адрес не разобран: {ws}"))?;
        let stream = ws_connect(h, &format!("/{path}"))?;
        Ok(Browser { stream, next_id: 1, rng: Rng::new() })
    }

    /// CDP-вызов: отправить запрос, читать события до ответа с нашим id.
    fn call(&mut self, method: &str, params: Json) -> Result<Json, String> {
        let id = self.next_id;
        self.next_id += 1;
        let msg = Json::Obj(vec![
            ("id".into(), Json::Int(id as i64)),
            ("method".into(), Json::Str(method.into())),
            ("params".into(), params),
        ])
        .to_string();
        write_frame(&mut self.stream, &mut self.rng, 0x1, msg.as_bytes())?;
        let deadline = std::time::Instant::now() + Duration::from_secs(15);
        loop {
            if std::time::Instant::now() > deadline {
                return Err(format!("CDP {method}: таймаут ответа"));
            }
            let Some(bytes) = read_message(&mut self.stream, &mut self.rng)? else {
                continue;
            };
            let j = json::parse(&String::from_utf8_lossy(&bytes))?;
            if j.get("id") == Some(&Json::Int(id as i64)) {
                if let Some(e) = j.get("error") {
                    return Err(format!("CDP {method}: {}", e.to_string()));
                }
                return Ok(j.get("result").cloned().unwrap_or(Json::Obj(Vec::new())));
            }
            // событие (без id) — пропустить
        }
    }

    /// Runtime.evaluate с returnByValue — значение как строка.
    fn eval(&mut self, expr: &str) -> Result<String, String> {
        let r = self.call(
            "Runtime.evaluate",
            Json::Obj(vec![
                ("expression".into(), Json::Str(expr.into())),
                ("returnByValue".into(), Json::Bool(true)),
            ]),
        )?;
        if r.get("exceptionDetails").is_some() {
            return Err("CDP: JS-исключение".into());
        }
        let v = r
            .get("result")
            .and_then(|x| x.get("value"))
            .cloned()
            .unwrap_or(Json::Str(String::new()));
        Ok(match v {
            Json::Str(s) => s,
            other => other.to_string(),
        })
    }

    /// Открыть URL и дождаться document.readyState == complete (до 5с).
    pub fn navigate(&mut self, url: &str) -> Result<(), String> {
        self.call(
            "Page.navigate",
            Json::Obj(vec![("url".into(), Json::Str(url.into()))]),
        )?;
        for _ in 0..100 {
            if self.eval("document.readyState").as_deref() == Ok("complete") {
                return Ok(());
            }
            std::thread::sleep(Duration::from_millis(50));
        }
        Err("страница не загрузилась за 5с".into())
    }

    /// Текст страницы (innerText).
    pub fn page_text(&mut self) -> Result<String, String> {
        self.eval("document.body ? document.body.innerText : ''")
    }

    /// Клик по CSS-селектору: центр элемента → Input.dispatchMouseEvent.
    pub fn click(&mut self, selector: &str) -> Result<(), String> {
        let sel = Json::Str(selector.to_string()).to_string();
        let expr = format!(
            "(function(){{var el=document.querySelector({sel});if(!el) return '';var r=el.getBoundingClientRect();return JSON.stringify([r.x+r.width/2,r.y+r.height/2]);}})()"
        );
        let v = self.eval(&expr)?;
        if v.is_empty() {
            return Err(format!("элемент {selector:?} не найден"));
        }
        let j = json::parse(&v)?;
        let Json::Arr(a) = &j else {
            return Err("координаты не массивом".into());
        };
        if a.len() != 2 {
            return Err("координаты не пара".into());
        }
        let x = f64_of(&a[0]).ok_or("x не число")?;
        let y = f64_of(&a[1]).ok_or("y не число")?;
        self.mouse_event("mouseMoved", x, y, 0)?;
        self.mouse_event("mousePressed", x, y, 1)?;
        self.mouse_event("mouseReleased", x, y, 1)?;
        Ok(())
    }

    fn mouse_event(&mut self, typ: &str, x: f64, y: f64, clicks: i64) -> Result<(), String> {
        self.call(
            "Input.dispatchMouseEvent",
            Json::Obj(vec![
                ("type".into(), Json::Str(typ.into())),
                ("x".into(), Json::Float(x)),
                ("y".into(), Json::Float(y)),
                ("button".into(), Json::Str("left".into())),
                ("clickCount".into(), Json::Int(clicks)),
            ]),
        )
        .map(|_| ())
    }

    /// Ввести текст: клик по полю (фокус) + Input.insertText.
    pub fn type_text(&mut self, selector: &str, text: &str) -> Result<(), String> {
        self.click(selector)?;
        self.call(
            "Input.insertText",
            Json::Obj(vec![("text".into(), Json::Str(text.into()))]),
        )
        .map(|_| ())
    }
}

fn f64_of(j: &Json) -> Option<f64> {
    match j {
        Json::Float(f) => Some(*f),
        Json::Int(n) => Some(*n as f64),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Cursor;

    fn hex(b: &[u8]) -> String {
        b.iter().map(|x| format!("{x:02x}")).collect()
    }

    #[test]
    fn sha1_vectors() {
        assert_eq!(hex(&sha1(b"abc")), "a9993e364706816aba3e25717850c26c9cd0d89d");
        assert_eq!(hex(&sha1(b"")), "da39a3ee5e6b4b0d3255bfef95601890afd80709");
        assert_eq!(
            hex(&sha1(b"The quick brown fox jumps over the lazy dog")),
            "2fd4e1c67a2d28fced849ee1bb76e7391b93eb12"
        );
    }

    #[test]
    fn base64_vectors() {
        assert_eq!(base64(b""), "");
        assert_eq!(base64(b"f"), "Zg==");
        assert_eq!(base64(b"fo"), "Zm8=");
        assert_eq!(base64(b"foo"), "Zm9v");
        assert_eq!(base64(b"foobar"), "Zm9vYmFy");
    }

    /// RFC 6455 §1.3: известный вектор рукопожатия.
    #[test]
    fn websocket_accept_rfc_vector() {
        assert_eq!(accept_of("dGhlIHNhbXBsZSBub25jZQ=="), "s3pPLMBiTxaQ9kYGzzhZRbK+xOo=");
    }

    #[test]
    fn url_policy() {
        assert!(url_allowed("http://x.ru").is_ok());
        assert!(url_allowed("https://x.ru").is_ok());
        assert!(url_allowed("data:text/html,привет").is_ok());
        assert!(url_allowed("").is_err());
        for bad in ["file:///etc/passwd", "about:blank", "chrome://settings", "ftp://x"] {
            let e = url_allowed(bad).unwrap_err();
            assert!(e.contains("обход джейла"), "{bad}: {e}");
        }
    }

    /// Труба в памяти: Read+Write в одном объекте (как TcpStream).
    struct Pipe {
        inp: Cursor<Vec<u8>>,
        out: Vec<u8>,
    }
    impl Read for Pipe {
        fn read(&mut self, buf: &mut [u8]) -> std::io::Result<usize> {
            self.inp.read(buf)
        }
    }
    impl Write for Pipe {
        fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
            self.out.extend_from_slice(buf);
            Ok(buf.len())
        }
        fn flush(&mut self) -> std::io::Result<()> {
            Ok(())
        }
    }

    #[test]
    fn frame_roundtrip() {
        let mut rng = Rng::new();
        // короткий кадр (маскированный) → обратный разбор
        let mut p = Pipe { inp: Cursor::new(Vec::new()), out: Vec::new() };
        write_frame(&mut p, &mut rng, 0x1, "привет".as_bytes()).unwrap();
        p.inp = Cursor::new(p.out.clone());
        let msg = read_message(&mut p, &mut rng).unwrap().unwrap();
        assert_eq!(msg, "привет".as_bytes());
        // длинный payload — 16-битная длина
        let long: Vec<u8> = (0..300u32).map(|i| (i % 251) as u8).collect();
        let mut p = Pipe { inp: Cursor::new(Vec::new()), out: Vec::new() };
        write_frame(&mut p, &mut rng, 0x1, &long).unwrap();
        p.inp = Cursor::new(p.out.clone());
        let msg = read_message(&mut p, &mut rng).unwrap().unwrap();
        assert_eq!(msg, long);
        // 64-битная длина (маленькое значение под маркером 127); payload замаскирован
        let mask = [9u8, 9, 9, 9];
        let masked_payload: Vec<u8> =
            b"hello".iter().enumerate().map(|(i, b)| b ^ mask[i % 4]).collect();
        let mut raw = vec![0x81u8, 0x80 | 127];
        raw.extend_from_slice(&5u64.to_be_bytes());
        raw.extend_from_slice(&mask);
        raw.extend_from_slice(&masked_payload);
        let mut p = Pipe { inp: Cursor::new(raw), out: Vec::new() };
        let msg = read_message(&mut p, &mut rng).unwrap().unwrap();
        assert_eq!(msg, b"hello");
        // ping → pong (opcode 0xA) в исходящем потоке; далее — текст-кадр
        let mut raw = vec![0x89u8, 4, 1, 2, 3, 4]; // немаскированный ping
        raw.extend_from_slice(&[0x81u8, 2, b'o', b'k']);
        let mut p = Pipe { inp: Cursor::new(raw), out: Vec::new() };
        let msg = read_message(&mut p, &mut rng).unwrap().unwrap();
        assert_eq!(msg, b"ok");
        assert_eq!(p.out[0] & 0x0F, 0xA, "pong opcode");
        assert_eq!(p.out[1] & 0x7F, 4, "pong повторяет payload ping'а (4 байта), маска выставлена");
        assert_eq!(p.out.len(), 10, "2 заголовка + 4 маски + 4 payload");
    }

    /// Фрагментированное сообщение склеивается.
    #[test]
    fn fragments_assembled() {
        let raw = vec![0x01u8, 3, b'a', b'b', b'c', 0x80, 3, b'd', b'e', b'f'];
        let mut p = Pipe { inp: Cursor::new(raw), out: Vec::new() };
        let msg = read_message(&mut p, &mut Rng::new()).unwrap().unwrap();
        assert_eq!(msg, b"abcdef");
    }

    /// CDP-сообщение строится и разбирается (id/method/params).
    #[test]
    fn cdp_message_json() {
        let m = Json::Obj(vec![
            ("id".into(), Json::Int(7)),
            ("method".into(), Json::Str("Page.navigate".into())),
            (
                "params".into(),
                Json::Obj(vec![("url".into(), Json::Str("data:text/html,привет".into()))]),
            ),
        ])
        .to_string();
        let j = json::parse(&m).unwrap();
        assert_eq!(j.get("id"), Some(&Json::Int(7)));
        assert_eq!(j.get("method").and_then(|x| x.as_str()), Some("Page.navigate"));
        assert_eq!(
            j.get("params").and_then(|p| p.get("url")).and_then(|u| u.as_str()),
            Some("data:text/html,привет")
        );
    }
}
