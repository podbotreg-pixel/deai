# ROADMAP — DI-NET

> Принциптал: каждая фаза имеет работающий артефакт, а не «план на бумаге».
> Free / decentralized / unstoppable — проверяется в каждой фазе.

## M0 — Локальная «жизнь» (4 недели)

- [x] Прототип протокола `prototype/proto.py` (сообщения + CBOR-lite + secure_agg demo)
- [x] Ядро Rust собирается, конфигурация, планировщик ресурсов (лимиты CPU/RAM/батареи)
      (`dinet-crypto`, `dinet-proto`, `dinet-core`, `dinet-fl`, `dinet-net`, `dinet-node`; std-only)
- [ ] llama.cpp binding, GGUF Q4, стриминг токенов (точка прикрепления: LocalTrainer-интерфейс в dinet-fl)
- [x] История чата v0 (CLI REPL `dinet-chat::chatdemo`): сессия над локальной памятью,
      RAG-контекст хода, буфер пар speak→learn в том же KV-файле; трейт `ChatModel` —
      точка прикрепления llama.cpp
- [ ] Чат-UI (Tauri) поверх `ChatSession`, экспорт, миграция (механика сессии уже готова)
- [x] **Speak→learn (механика):** буфер пар → LoRA + hinge-предпочтение (saturating DPO-lite),
      DP-клиппинг, local_fit → secure_agg — на toy-модели, интерфейс LocalTrainer устойчив
      к замене на llama.cpp
- [x] Локальная память v0: append-only KV-лог на std (замена SQLite — ноль зависимостей,
      WAL-семантика: обрыв хвоста обрезается, CRC-записи, tombstones, компактация)
      + hash-embedding RAG (замена sqlite-vec; точка прикрепления: `rag::embed()` → llama.cpp)
      (`dinet-memory`: `KvStore` + `Retriever`; REPL-демо `memdemo`)

**DoD ( проверить руками ):** 16-GB laptop без GPU, чат ≥15 ток/с; обучение партии 20 пар
≤20 мин с приоритетом ниже окна браузера; RAM-лимит держится (≤2 ГБ); кнопка «стоп»
останавливает любую джобу ≤2 с.

## M1 — Рой (6 недель)

- [ ] libp2p: QUIC+WS, gossipsub, Kademlia DHT, mDNS, PEX, 5 источников bootstrap
      (точка прикрепления: трейт `transport::Transport` — тот же wire-набор 0x11/0x22/0x30/0x31)
- [x] Transport v0 в симуляции: обмен РЕАЛЬНЫМИ подписанными wire-кадрами dinet-proto
      (in-memory mesh + файловый спул между процессами; подделка payload → отбраковка по подписи)
- [x] Рой в разных ОС-процессах (`spoolswarm`): процесс владеет частью узлов,
      между процессами — только файлы (кадры + барьеры фаз A/V/N/S);
      результат бит-точно совпадает с однопроцессным роем (тест эквивалентности)
- [x] Secure Aggregation (pairwise masking) — без отправки индивидуальных апдейтов
      (`dinet-fl::secure_agg`: инвариант Σмасок==Σθ проверен тестами; v0-masks на SHA-256, v1 — X25519)
- [x] Robust-agg: median norm-band + Krum-консенсус + trimmed mean
      (`dinet-node::swarm::robust_screen`; 10% poison-узлов отсекаются на reveal-эрах до чек-пойнта)
- [ ] Ротируемые роли: aggregator каждый era (окно 1 ч), lazy-audit 10%
      (v0: leaderless-сим — каждый узел проверяет всех Himself; ролям делать нечего до libp2p)
- [x] Fork-choice чек-пойнта: publish только при качестве ≥2/3 узлов (eval-vote), иначе last-good
- [ ] Реестр чек-пойнтов: append-only лог + кворум нотариусов + внешние якоря
- [ ] Репликация весов через torrent-схему (дельты от parent)

**DoD:** 100 контейнеров (200 Mbps/5% loss эмуляция): метрики сходятся после N эпох;
10% malicious-узлов не могут сдвинуть eval на > заданный δ; выключение любого
агрегатора/нотарriуса/релея замедляет, но не останавливает.

**Статус v0 (2026-10-25):** DiLoCo-расписание (маскированные эры накопления + редкие
REVEAL-эры с robust-агрегацией и чек-пойнтом); симуляция 20 узлов / 10% отравителей:
точность 0.08→1.00 по чек-пойнтам, отравители отбракованы на каждой reveal-эре,
выход 40% узлов не останавливает сеть (`cargo run -p dinet-node`).

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
