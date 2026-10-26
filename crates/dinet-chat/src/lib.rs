//! dinet-chat — локальная «жизнь» DI-NET v0: чат поверх локальной памяти.
//!
//! Пайплайн хода (всё локально, всё бесплатно, всё на диске после выхода):
//! 1. Сообщение пользователя → контекст = top-k фактов из памяти (RAG).
//! 2. `ChatModel.respond(user, context)` → ответ. llama.cpp подключается
//!    сюда (GGUF Q4-квант, стриминг позднее).
//! 3. История (`chat:turn:*`), память (`rag:doc:*`) и буфер обучения
//!    (`learn:pair:*`) пишутся в ОДИН append-only KV-файл `dinet-memory`.
//! 4. Speak→learn: каждый ход образует пару (prompt, answer) в буфере —
//!    рой дообучает модель этого клиента после общения; v1 придаст там
//!    LocalTrainer/llama.cpp (интерфейс был стабилен в dinet-fl).

pub mod model;
pub mod session;

pub use model::{ChatModel, ToyChatModel};
pub use session::{ChatSession, LearnPair, Role, Turn};
