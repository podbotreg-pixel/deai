//! dinet-client — КЛИЕНТ, ПОЛЬЗУЮЩИЙСЯ ЧАТОМ, = ПОЛНОПРАВНЫЙ УЧАСТНИК РОЯ.
//!
//! Принцип DI-NET: обучение распределено по клиентам. Пользователь общается
//! с чатом → его машина дообучается НА ЕГО ДИАЛОГАХ (speak→learn) →
//! подписанная DP-дельта уходит в эру роя (`dinet-node::run_era_partition`),
//! и чек-пойнт растёт от вкладов ВСЕХ пользователей.
//!
//! v0 (toy-мир): диалоговый микроязык из 8 слов; слово с индексом i = токен i
//! (словарь `LEX`). Пользователь-учитель всегда отвечает ПРАВИЛЬНЫМ словом
//! по правилу мира (t3+1)%8 — это «клиенты говорят на живом языке роя».
//! Точки прикрепления llama.cpp (v1):
//! - `pair_to_sample` ← натуральный токенизатор реальных диалогов;
//! - `batch_override` в `dinet-node::Node` ← данные из `ChatSession::pairs()`.
//!
//! std-only: никаких внешних зависимостей, как весь workspace.

use dinet_fl::toy_llm::{Sample, CTX, V};

/// Микроязык чата: слово i = токен i (v1: натуральные токены llama.cpp).
pub const LEX: [&str; V] = ["ай", "би", "ви", "ги", "ди", "ей", "жи", "зи"];

/// Текст (пробелы-разделённые слова микроязыка) → токены.
/// Неизвестное слово → None (в v1 — натуральный токенизатор,词条 не выкидываем).
pub fn lex_to_tokens(text: &str) -> Option<Vec<u8>> {
    let mut t = Vec::new();
    for w in text.split_whitespace() {
        match LEX.iter().position(|l| *l == w) {
            Some(i) => t.push(i as u8),
            None => return None,
        }
    }
    if t.is_empty() { None } else { Some(t) }
}

/// Пара диалога → обучающий sample:
/// ctx = первые 4 токена prompt, chosen = СЛОВО-ответ учителя (клиента),
/// rejected — заведомо «плохой» вариант (как в honesty-батче: chosen+3).
pub fn pair_to_sample(prompt: &str, answer: &str) -> Option<Sample> {
    let t = lex_to_tokens(prompt)?;
    if t.len() < CTX {
        return None;
    }
    let a = lex_to_tokens(answer)?;
    if a.len() != 1 {
        return None;
    }
    let chosen = a[0];
    Some(Sample {
        tokens: [t[0], t[1], t[2], t[3]],
        chosen,
        rejected: Some((chosen + 3) % V as u8),
    })
}

/// Батч клиента из списка диалогов; не-микроязыковые ходы пропускаются.
pub fn pairs_to_batch<L: AsRef<str>, R: AsRef<str>>(pairs: &[(L, R)]) -> Vec<Sample> {
    pairs
        .iter()
        .filter_map(|(p, a)| pair_to_sample(p.as_ref(), a.as_ref()))
        .collect()
}

fn phrase(t: &[u8; CTX]) -> String {
    t.iter().map(|&i| LEX[i as usize]).collect::<Vec<_>>().join(" ")
}

/// Скриптовый «пользователь»: k реплик (диалог-вопрос + правильное слово-
/// ответ). Детерминирован между процессами: токены из SHA-256(seed, i, j).
/// В v1 здесь — реальные ходы `ChatSession::turn()` пользователя.
pub fn user_script(seed: u64, k: u32) -> Vec<(String, String)> {
    (0..k)
        .map(|i| {
            let mut b = Vec::with_capacity(17);
            b.extend_from_slice(&seed.to_le_bytes());
            b.extend_from_slice(&(i as u32).to_le_bytes());
            let mut toks = [0u8; CTX];
            for (j, t) in toks.iter_mut().enumerate() {
                let mut bj = b.clone();
                bj.push(j as u8);
                *t = dinet_crypto::sha256(&bj)[0] % V as u8;
            }
            let chosen = (toks[CTX - 1] + 1) % V as u8;
            (phrase(&toks), LEX[chosen as usize].to_string())
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn lex_maps_word_to_token_id() {
        assert_eq!(lex_to_tokens("ай ви ди би"), Some(vec![0, 2, 4, 1]));
        assert_eq!(lex_to_tokens("зи"), Some(vec![7]));
        assert_eq!(lex_to_tokens("привет"), None, "вне словаря роя");
        assert_eq!(lex_to_tokens("ай привет"), None);
        assert_eq!(lex_to_tokens(""), None);
    }

    #[test]
    fn pair_walks_world_rule() {
        // t3 = 1 → правильный ответ = 2 («ви»)
        let s = pair_to_sample("ай ви ди би", "ви").expect("микроязык валиден");
        assert_eq!(s.tokens, [0, 2, 4, 1]);
        assert_eq!(s.chosen, 2);
        assert_eq!(s.rejected, Some((2 + 3) % 8));
        // неполный контекст / многословный ответ — не пара
        assert!(pair_to_sample("ай би", "ви").is_none());
        assert!(pair_to_sample("ай ви ди би", "ви ай").is_none());
    }

    #[test]
    fn user_script_is_deterministic_and_valid() {
        let a = user_script(9, 5);
        let b = user_script(9, 5);
        assert_eq!(a, b, "скриптовый юзер детерминирован");
        assert_eq!(a.len(), 5);
        for (p, ans) in &a {
            let s = pair_to_sample(p, ans).expect("скриптовые пары валидны");
            // юзер-учитель отвечает по правилу мира: chosen = (t3 + 1) % 8
            assert_eq!(s.chosen, (s.tokens[3] + 1) % 8, "{p} → {ans}");
        }
        let c = user_script(10, 5);
        assert_ne!(a, c, "разные сеансы — разные диалоги");
    }

    #[test]
    fn pairs_to_batch_filters_non_lex() {
        let d: Vec<(String, String)> = vec![
            ("ай ви ди би".into(), "ги".into()),
            ("как дела?".into(), "хорошо".into()), // вне микроязыка
            ("зи жи ди би".into(), "зи".into()),
        ];
        let b = pairs_to_batch(&d);
        assert_eq!(b.len(), 2);
        assert_eq!(b[0].chosen, 3);
        assert_eq!(b[1].chosen, 7);
    }
}
