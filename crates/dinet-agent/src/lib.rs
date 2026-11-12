//! Агент DI-NET (M2): MCP-сервер поверх песочницы + capability-токены.
//! Разбивка: json (RFC 8259 подмножество), caps (READ/WRITE/DESTRUCTIVE/
//! NETWORK, ed25519-подпись), jail (FS-песочница), journal (undo),
//! tools (инструменты), mcp (JSON-RPC 2.0 stdio NDJSON).

pub mod audit;
pub mod browser;
pub mod caps;
pub mod cua;
pub mod jail;
pub mod journal;
pub mod json;
pub mod mcp;
pub mod phishing;
pub mod phishing_model;
pub mod skills;
pub mod tools;

pub use audit::{Audit, AuditRec};
pub use caps::{CapLevel, CapSet, CapToken};
pub use jail::Jail;
pub use journal::{Inverse, Journal};
pub use mcp::Server;
pub use phishing::Verdict;
pub use phishing_model::Model;
pub use skills::{Skill, SkillStep, SkillStore};
pub use tools::{AllowRule, ToolWorld};
