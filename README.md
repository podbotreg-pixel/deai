# DI-NET

> **Бесплатная для всех · Децентрализованная · Нельзя отключить.**
> Федеративная нейросеть: чат + агент + дообучение на клиентских машинах.

## Суть

Каждое устройство пользователя — узел сети:

- **Чат** — локальная LLM (GGUF/llama.cpp, MoE ≤2B активных), персональный LoRA-адаптер,
  локальная память (RAG). Диалоги не покидают устройство никогда.
- **Бесплатно** — нет аккаунтов, оплаты, телеметрии. Ядро AGPL-3.0, SDK Apache-2.0.
- **Дообучение после общения** — из одобренных/исправленных ответов локально считается
  DP-SGD обновление (только LoRA), сжимается (≤20 МБ), подписывается и уходит в сеть
  анонимно. **Все исторические знания остаются у пользователя.**
- **Агент** — PAPAV-цикл, MCP-инструменты (файлы, CLI, браузер, GUI-автоматизация,
  соцсети — только официальные API и с явным подтверждением), undo-журнал, анти-фишинг.
- **Децентрализация** — libp2p (QUIC + WS-443), gossipsub, DHT, репутация,
  ротируемые роли (aggregator/notary/evaluator). Не заблокировать ничего — блокировка
  DNS/доменов/UDP ничего не ломает (см. `docs/RESILIENCE.md`).
- **Реестр весов** — append-only DAG чек-пойнтов с кворумом нотариусов и внешними
  якорями; модель распространяется как content-addressed данные (BitTorrent-схеме).

## Структура репозитория

```
docs/                 архитектура, протокол, устойчивость, дорожная карта
crates/
  dinet-crypto/       SHA-256, DID, Merkle, Signer-трейт (ed25519 — точка прикрепления)
  dinet-proto/        wire-протокол v0 (детерминированный CBOR-lite), коды сообщений
  dinet-core/         конфигурация, планировщик ресурсов (лимиты CPU/RAM/батарея/сеть)
  dinet-fl/           quant+error feedback, DP-клиппинг, trimmed mean / Krum, FedGraph-score
  dinet-node/         узел: DiLoCo-расписание эр, robust-скрининг, fork-choice, симуляция роя
  dinet-net/          пeer-to-peer слой (libp2p-интеграция — точка прикрепления)
  dinet-node/         рой: эпохи DiLoCo, secure-agg, robust-скрининг, fork-choice,
                      транспорт wire-кадров (in-memory mesh + файловый спул; libp2p —
                      следующий backend через трейт `Transport`)
  dinet-memory/       локальная память: append-only KV-лог (std-only, замена SQLite)
                      + hash-эмбеддинг RAG (замена sqlite-vec); libp2p/llama.cpp — v1
prototype/            исполняемый Питон-прототип: кодирование сообщений + демо Secure Agg
PROMPT.md             исходный мега-промт проектирования
```

## Сборка

```bash
cargo build          # только std, никаких внешних зависимостей
cargo test           # юнит-тесты всех крейтов
python3 prototype/proto.py   # демо: подпись→CBOR→secure-aggregation
cargo run -p dinet-node      # демо роя: 20 узлов, 10% отравителей, robust-агрегация, чек-пойнты
cargo run -p dinet-memory --bin memdemo -- ./dinet-mem
                             # REPL памяти: /remember | /ask | /forget | /stats
```

## Ключевые документы

- `docs/ARCHITECTURE.md` — вся система, от топологии до бюджета ресурсов.
- `docs/RESILIENCE.md` — исчерпывающая матрица «что сломали → что произойдёт».
- `docs/PROTOCOL.md` — wire-спецификация v0.
- `docs/ROADMAP.md` — фазы M0–M4 с Definition of Done.

## Статус

`M0` в пути (см. ROADMAP). Прототип протокола и крейты-фундамент — рабочие.
