//! Минимальный JSON (std-only): парсер + сериализатор для MCP-поверхности.
//! Подмножество RFC 8259, достаточное для JSON-RPC 2.0 и схем инструментов:
//! объекты, массивы, строки (с \uXXXX и сурогатными парами), числа
//! (i64/f64), true/false/null. Глубина вложения ограничена (защита от
//! стек-переполнения на вражеском входе).

#[derive(Debug, Clone, PartialEq)]
pub enum Json {
    Null,
    Bool(bool),
    /// Целое (в wire-формате без дробной части/экспоненты).
    Int(i64),
    /// Всё остальное число (дробь/экспонента); сериализация честная.
    Float(f64),
    Str(String),
    Arr(Vec<Json>),
    Obj(Vec<(String, Json)>),
}

const MAX_DEPTH: usize = 64;

impl Json {
    /// Сериализация в компактный wire-формат (одна строка — NDJSON).
    pub fn to_string(&self) -> String {
        let mut out = String::new();
        self.write(&mut out);
        out
    }

    fn write(&self, out: &mut String) {
        match self {
            Json::Null => out.push_str("null"),
            Json::Bool(b) => out.push_str(if *b { "true" } else { "false" }),
            Json::Int(n) => out.push_str(&n.to_string()),
            Json::Float(f) => {
                // finity обязательна: NaN/inf в JSON не существуют
                if f.is_finite() {
                    out.push_str(&f.to_string());
                } else {
                    out.push_str("null");
                }
            }
            Json::Str(s) => write_escaped(s, out),
            Json::Arr(items) => {
                out.push('[');
                for (i, it) in items.iter().enumerate() {
                    if i > 0 {
                        out.push(',');
                    }
                    it.write(out);
                }
                out.push(']');
            }
            Json::Obj(fields) => {
                out.push('{');
                for (i, (k, v)) in fields.iter().enumerate() {
                    if i > 0 {
                        out.push(',');
                    }
                    write_escaped(k, out);
                    out.push(':');
                    v.write(out);
                }
                out.push('}');
            }
        }
    }

    /// Поле объекта (первое с таким именем).
    pub fn get(&self, key: &str) -> Option<&Json> {
        match self {
            Json::Obj(fields) => fields.iter().find(|(k, _)| k == key).map(|(_, v)| v),
            _ => None,
        }
    }

    /// Строковое поле ("стринг-коэрция": Str | иначе None).
    pub fn as_str(&self) -> Option<&str> {
        match self {
            Json::Str(s) => Some(s),
            _ => None,
        }
    }
}

/// Экранирование строки: обязательные \" \\ и control-символы (<0x20).
fn write_escaped(s: &str, out: &mut String) {
    out.push('"');
    for c in s.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            '\u{08}' => out.push_str("\\b"),
            '\u{0C}' => out.push_str("\\f"),
            c if (c as u32) < 0x20 => {
                out.push_str(&format!("\\u{:04x}", c as u32));
            }
            c => out.push(c),
        }
    }
    out.push('"');
}

/// Разбор NDJSON-строки в Json. Ошибка — человекочитаемая позиция.
pub fn parse(input: &str) -> Result<Json, String> {
    let bytes: Vec<char> = input.chars().collect();
    let mut p = Parser { b: &bytes, i: 0 };
    p.ws();
    let v = p.value(0)?;
    p.ws();
    if p.i != p.b.len() {
        return Err(format!("лишние символы после значения на {}", p.i));
    }
    Ok(v)
}

struct Parser<'a> {
    b: &'a [char],
    i: usize,
}

impl<'a> Parser<'a> {
    fn peek(&self) -> Option<char> {
        self.b.get(self.i).copied()
    }

    fn ws(&mut self) {
        while matches!(self.peek(), Some(' ' | '\t' | '\r' | '\n')) {
            self.i += 1;
        }
    }

    fn eat(&mut self, c: char) -> Result<(), String> {
        if self.peek() == Some(c) {
            self.i += 1;
            Ok(())
        } else {
            Err(format!("ожидался '{c}' на позиции {}", self.i))
        }
    }

    fn value(&mut self, depth: usize) -> Result<Json, String> {
        if depth > MAX_DEPTH {
            return Err("слишком глубокая вложенность".into());
        }
        match self.peek() {
            Some('{') => self.object(depth),
            Some('[') => self.array(depth),
            Some('"') => Ok(Json::Str(self.string()?)),
            Some('t') => self.lit("true", Json::Bool(true)),
            Some('f') => self.lit("false", Json::Bool(false)),
            Some('n') => self.lit("null", Json::Null),
            Some(c) if c == '-' || c.is_ascii_digit() => self.number(),
            Some(c) => Err(format!("неожиданный символ '{c}' на {}", self.i)),
            None => Err("неожиданный конец входа".into()),
        }
    }

    fn lit(&mut self, word: &str, v: Json) -> Result<Json, String> {
        for c in word.chars() {
            self.eat(c)?;
        }
        Ok(v)
    }

    fn object(&mut self, depth: usize) -> Result<Json, String> {
        self.eat('{')?;
        let mut fields = Vec::new();
        self.ws();
        if self.peek() == Some('}') {
            self.i += 1;
            return Ok(Json::Obj(fields));
        }
        loop {
            self.ws();
            let key = self.string()?;
            self.ws();
            self.eat(':')?;
            self.ws();
            let val = self.value(depth + 1)?;
            fields.push((key, val));
            self.ws();
            match self.peek() {
                Some(',') => self.i += 1,
                Some('}') => {
                    self.i += 1;
                    return Ok(Json::Obj(fields));
                }
                _ => return Err(format!("ожидался ',' или '}}' на {}", self.i)),
            }
        }
    }

    fn array(&mut self, depth: usize) -> Result<Json, String> {
        self.eat('[')?;
        let mut items = Vec::new();
        self.ws();
        if self.peek() == Some(']') {
            self.i += 1;
            return Ok(Json::Arr(items));
        }
        loop {
            self.ws();
            items.push(self.value(depth + 1)?);
            self.ws();
            match self.peek() {
                Some(',') => self.i += 1,
                Some(']') => {
                    self.i += 1;
                    return Ok(Json::Arr(items));
                }
                _ => return Err(format!("ожидался ',' или ']' на {}", self.i)),
            }
        }
    }

    fn string(&mut self) -> Result<String, String> {
        self.eat('"')?;
        let mut s = String::new();
        loop {
            match self.peek() {
                Some('"') => {
                    self.i += 1;
                    return Ok(s);
                }
                Some('\\') => {
                    self.i += 1;
                    match self.peek() {
                        Some('"') => {
                            s.push('"');
                            self.i += 1;
                        }
                        Some('\\') => {
                            s.push('\\');
                            self.i += 1;
                        }
                        Some('/') => {
                            s.push('/');
                            self.i += 1;
                        }
                        Some('n') => {
                            s.push('\n');
                            self.i += 1;
                        }
                        Some('r') => {
                            s.push('\r');
                            self.i += 1;
                        }
                        Some('t') => {
                            s.push('\t');
                            self.i += 1;
                        }
                        Some('b') => {
                            s.push('\u{08}');
                            self.i += 1;
                        }
                        Some('f') => {
                            s.push('\u{0C}');
                            self.i += 1;
                        }
                        Some('u') => {
                            self.i += 1;
                            let hi = self.hex4()?;
                            let cp = if (0xD800..0xDC00).contains(&hi) {
                                // сурогатная пара: \uXXXX\uYYYY → кодпойнт
                                if self.peek() == Some('\\') {
                                    self.i += 1;
                                    self.eat('u')?;
                                    let lo = self.hex4()?;
                                    if !(0xDC00..0xE000).contains(&lo) {
                                        return Err("сурогатная пара: младший не DC00-EFFF".into());
                                    }
                                    0x10000 + ((hi - 0xD800) << 10) + (lo - 0xDC00)
                                } else {
                                    return Err("одинокий старший сурогат".into());
                                }
                            } else if (0xDC00..0xE000).contains(&hi) {
                                return Err("одинокий младший сурогат".into());
                            } else {
                                hi
                            };
                            match char::from_u32(cp) {
                                Some(c) => s.push(c),
                                None => return Err(format!("невалидный кодпойнт {cp}")),
                            }
                        }
                        _ => return Err(format!("плохой escape на {}", self.i)),
                    }
                }
                Some(c) => {
                    s.push(c);
                    self.i += 1;
                }
                None => return Err("строка не закрыта".into()),
            }
        }
    }

    fn hex4(&mut self) -> Result<u32, String> {
        let mut v = 0u32;
        for _ in 0..4 {
            let c = self.peek().ok_or("escape \\u не закрыт")?;
            let d = c.to_digit(16).ok_or(format!("'{c}' не hex-цифра"))?;
            v = v * 16 + d;
            self.i += 1;
        }
        Ok(v)
    }

    fn number(&mut self) -> Result<Json, String> {
        let start = self.i;
        if self.peek() == Some('-') {
            self.i += 1;
        }
        let mut digits = 0usize;
        while matches!(self.peek(), Some(c) if c.is_ascii_digit()) {
            self.i += 1;
            digits += 1;
        }
        if digits == 0 {
            return Err(format!("число без цифр на {}", start));
        }
        let mut is_float = false;
        if self.peek() == Some('.') {
            is_float = true;
            self.i += 1;
            let mut frac = 0usize;
            while matches!(self.peek(), Some(c) if c.is_ascii_digit()) {
                self.i += 1;
                frac += 1;
            }
            if frac == 0 {
                return Err(format!("дробь без цифр на {}", self.i));
            }
        }
        if matches!(self.peek(), Some('e' | 'E')) {
            is_float = true;
            self.i += 1;
            if matches!(self.peek(), Some('+' | '-')) {
                self.i += 1;
            }
            let mut exp = 0usize;
            while matches!(self.peek(), Some(c) if c.is_ascii_digit()) {
                self.i += 1;
                exp += 1;
            }
            if exp == 0 {
                return Err(format!("экспонента без цифр на {}", self.i));
            }
        }
        let text: String = self.b[start..self.i].iter().collect();
        if !is_float {
            if let Ok(n) = text.parse::<i64>() {
                return Ok(Json::Int(n));
            }
            // переполнение i64 → f64 (JSON не различает)
        }
        text.parse::<f64>()
            .map(Json::Float)
            .map_err(|_| format!("невалидное число '{text}'"))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn roundtrip_scalars_and_escapes() {
        for src in [
            "null", "true", "false", "0", "-17", "3.5", "1e-3", "\"а\\nб\\t\\\"\\\\\\u0432\"",
        ] {
            let v = parse(src).expect(src);
            assert_eq!(parse(&v.to_string()).expect("reparse"), v, "{src}");
        }
    }

    #[test]
    fn surrogate_pair_roundtrip() {
        // U+1F600 (смайл) в \u-парах
        let v = parse("\"\\ud83d\\ude00\"").expect("pair");
        assert_eq!(v, Json::Str("😀".into()));
        assert_eq!(parse(&v.to_string()).expect("reparse"), v);
    }

    #[test]
    fn nested_object_and_array() {
        let v = parse(r#"{"a":[1,{"b":null}],"c":"ы"}"#).expect("parse");
        assert_eq!(v.get("c").and_then(|j| j.as_str()), Some("ы"));
        assert_eq!(v.get("a").unwrap().to_string(), r#"[1,{"b":null}]"#);
    }

    #[test]
    fn bad_inputs_rejected() {
        for src in ["{", "{\"a\"}", "[1,]", "\"unterminated", "01x", "1.}", "\"\\u00\"", "nan"] {
            assert!(parse(src).is_err(), "{src} должен браковаться");
        }
    }

    #[test]
    fn deep_nesting_bounded() {
        // 65 уровней с пустым ядром парсятся (стек жив), 66 — брак
        let ok = "[".repeat(MAX_DEPTH) + &"]".repeat(MAX_DEPTH);
        assert!(parse(&ok).is_ok(), "глубина {MAX_DEPTH} — в норме");
        let bad = "[".repeat(MAX_DEPTH + 2) + &"]".repeat(MAX_DEPTH + 2);
        assert!(parse(&bad).is_err(), "глубина {} ограничена", MAX_DEPTH + 2);
    }
}
