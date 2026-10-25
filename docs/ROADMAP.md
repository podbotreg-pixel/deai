# ROADMAP — DI-NET

> Принциптал: каждая фаза имеет работающий артефакт, а не «план на бумаге».
> Free / decentralized / unstoppable — проверяется в каждой фазе.

## M0 — Локальная «жизнь» (4 недели)

- [x] Прототип протокола `prototype/proto.py` (сообщения + CBOR-lite + secure_agg demo)
- [ ] Ядро Rust собирается, конфигурация, планировщик ресурсов (лимиты CPU/RAM/батареи)
- [ ] llama.cpp binding, GGUF Q4, стриминг токенов
- [ ] Чат-UI (Tauri), история, экспорт, миграция
- [ ] **Speak→learn:** буфер пар → LoRA-DPO цикл (в фоне, safety-проповью DP-обрезка)
- [ ] Локальная память: SQLite + sqlite-vec RAG

**DoD ( проверить руками ):** 16-GB laptop без GPU, чат ≥15 ток/с; обучение партии 20 пар
≤20 мин с приоритетом ниже окна браузера; RAM-лимит держится (≤2 ГБ); кнопка «стоп»
останавливает любую джобу ≤2 с.

## M1 — Рой (6 недель)

- [ ] libp2p: QUIC+WS, gossipsub, Kademlia DHT, mDNS, PEX, 5 источников bootstrap
- [ ] Secure Aggregation (pairwise masking) — без отправки индивидуальных апдейтов
- [ ] Robust-agg: trimmed mean + Krum + FedGraph-score
- [ ] Ротируемые роли: aggregator каждый era (окно 1 ч), lazy-audit 10%
- [ ] Реестр чек-пойнтов: append-only лог + кворум нотариусов + внешние якоря
- [ ] Репликация весов через torrent-схему (дельты от parent)

**DoD:** 100 контейнеров (200 Mbps/5% loss эмуляция): метрики сходятся после N эпох;
10% malicious-узлов не могут сдвинуть eval на > заданный δ; выключение любого
агрегатора/нотарriуса/релея замедляет, но не останавливает.

## M2 — Агент (8 недель)

- [ ] MCP-серверы по платформам: FS-песочница, CLI-allowlist, браузер-CDP, UIA/AX/AT-SPI
- [ ] PAPAV-цикл + undo-журнал + Recovery (ReTrace-принцип)
- [ ] Capability-токены, уровни READ/WRITE/DESTRUCTIVE/NETWORK
- [ ] Анти-фишинг-детекторы (AgentPhishing-тренированные) на входе
- [ ] Evolving Skills: накопление, воспроизведение, share через SKILL_SHARE

**DoD:** ≥20 открытых CUA-задач (CUA-Lite-базовые), succ≥50%; фишинг-тест не выводит секреты;
все действия в журнале.

## M3 — Мобильные (8 недель)

- [ ] Android (Compose UI, JDK ↔ Rust core ↔ llama.cpp NDK)
- [ ] WorkManager-расписание (charging + un-metered Wi-Fi only)
- [ ] EFFEKT-прокси: телефон обучает 0.5B-прокси, вклад — через DSL/дистилляцию

**DoD:** телефон - вклад при батарее>70% и Wi-Fi, RAM ≤1.2 ГБ, thermal-останавливают.

## M4 — Продвинутые каналы (непрерывно)

- [ ] DSL-канал (обмен carrier-logits)
- [ ] CamoMark-подобный watermark
- [ ] Optimal Hydra: разные клиенты — разные подсети
- [ ] OPTIONAL: Tor-транспорт, MoE-экспертная федерация

**DoD:** сеть сходится при потере 30% узлов за окно; доля-модели восстанавливается из magnet.
