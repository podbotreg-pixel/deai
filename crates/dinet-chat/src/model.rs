//! ChatModel — точка прикрепления llama.cpp.
//!
//! v0-реализация `ToyChatModel`:
//! - без `state` — детерминированный шаблонный ответчик (проверка пайплайна);
//! - с `state` (обученная реплика роя): micro-языковой ход → argmax
//!   предсказания реплики (чат отвечает ТОЙ ЖЕ сетью, которую рой дообучил).
//! v1: `LlamaChatModel` (GGUF Q4) заменит оба режима за тем же трейтом.

use dinet_fl::toy_llm::{lex_to_tokens, LEX, CTX, ToyModel};

/// Модели-ответчики чата. `context` — релевантные факты памяти (RAG top-k).
pub trait ChatModel {
    /// Ответ ассистента на ход пользователя с учётом контекста памяти.
    fn respond(&mut self, user: &str, context: &[String]) -> String;
    /// Короткое имя модели (для статистики/журнала сессии).
    fn name(&self) -> &'static str;
}

/// Шаблонный ответчик v0 / рой-обученный ответчик реплики.
#[derive(Debug, Default, Clone)]
pub struct ToyChatModel {
    /// Обученная роем реплика (base + LoRA); None → шаблонный режим.
    pub state: Option<ToyModel>,
}

impl ToyChatModel {
    /// Ответчик, читающий предсказания реплики чек-пойнтов роя.
    pub fn trained(model: ToyModel) -> ToyChatModel {
        ToyChatModel { state: Some(model) }
    }
}

impl ChatModel for ToyChatModel {
    fn respond(&mut self, user: &str, context: &[String]) -> String {
        // Рой-реплика: ход на микроязыке → argmax предсказания сети
        if let Some(m) = &self.state {
            if let Some(t) = lex_to_tokens(user).filter(|t| t.len() >= CTX) {
                let toks = [t[0], t[1], t[2], t[3]];
                let next = m.predict(&toks);
                return LEX[next as usize].to_string();
            }
        }
        // Шаблонный режим (пайплайн RAG/истории)
        if context.is_empty() {
            return format!(
                "Пока память пуста: запомню «{}» и в следующий раз отвечу с опорой на неё.",
                short(user, 64)
            );
        }
        let facts: Vec<&String> = context.iter().take(2).collect();
        let mut s = if context.len() == 1 {
            "Из того, что я помню, есть только:".to_string()
        } else {
            "Из того, что я помню, это связано:".to_string()
        };
        for f in facts {
            s.push_str(&format!("\n  — {}", short(f, 90)));
        }
        s
    }

    fn name(&self) -> &'static str {
        "toy-v0"
    }
}

fn short(s: &str, n: usize) -> String {
    let cut: String = s.chars().take(n).collect();
    if s.chars().count() > n {
        format!("{cut}…")
    } else {
        cut
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use dinet_fl::toy_llm::toy_world::{honesty_batch, truth_next};
    use dinet_fl::toy_llm::{local_fit, LoRAAdapter, ToyBase};

    #[test]
    fn toy_model_uses_context_and_names_itself() {
        let mut m = ToyChatModel::default();
        assert_eq!(m.name(), "toy-v0");
        let r1 = m.respond("привет", &[]);
        assert!(r1.contains("память пуста"));
        let r2 = m.respond("а что есть?", &["факт один".to_string(), "факт два".to_string()]);
        assert!(r2.contains("факт один") && r2.contains("факт два"));
        let r3 = m.respond("а что есть?", &["только этот".to_string()]);
        assert!(r3.contains("только этот"));
    }

    #[test]
    fn trained_replica_answers_dialect_word() {
        // рой-реплика: ход микроязыком → ответ обязан быть словом словаря
        let mut m = ToyChatModel::trained(ToyModel {
            base: ToyBase::from_seed(7),
            adapter: LoRAAdapter::lora_init(42),
        });
        let a = m.respond("ай ви ди би", &[]);
        assert!(LEX.contains(&a.as_str()), "ответ реплики — слово словаря, был: {a}");
        // короткий ход (< CTX) → шаблонный режим с RAG-фактом
        let b = m.respond("ай би", &["факт один".to_string()]);
        assert!(b.contains("факт"), "fallback шаблона сломан: {b}");
    }

    #[test]
    fn trained_replica_follows_world_rule() {
        // дообученная рой-реплика предсказывает правило мира лучше случайного
        let base = ToyBase::from_seed(7);
        let mut rng = dinet_fl::Lcg::new(1);
        let fitted = local_fit(
            &base,
            &LoRAAdapter::lora_init(42),
            &honesty_batch(96, 1),
            120, 0.6, 2.0, 3.0, 0.0, &mut rng,
        );
        let mut v = LoRAAdapter::lora_init(42).as_vec();
        for (x, d) in v.iter_mut().zip(&fitted.delta) {
            *x += d;
        }
        let m = ToyChatModel::trained(ToyModel {
            base,
            adapter: LoRAAdapter::from_vec(&v),
        }).state.expect("trained");
        let eval = honesty_batch(64, 555);
        let good = eval
            .iter()
            .filter(|s| m.predict(&s.tokens) == truth_next(&s.tokens))
            .count();
        assert!(
            good as f32 / eval.len() as f32 > 0.5,
            "реплика должна выучить правило мира: {good}/{}",
            eval.len()
        );
    }
}
