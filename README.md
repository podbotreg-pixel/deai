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
  dinet-crypto/       SHA-256, DID, Merkle, CRC-32, ed25519-dalek подписи, X25519-маски
  dinet-proto/        wire-протокол v0 (детерминированный CBOR-lite), коды сообщений
  dinet-core/         конфигурация, планировщик ресурсов (лимиты CPU/RAM/батарея/сеть)
  dinet-fl/           федеративная математика: quant+error feedback, DP-клиппинг,
                      trimmed mean / Krum, FedGraph-score (std-only)
  dinet-net/          сетевой слой-скелет: bootstrap-источники, роли, ставка репутации
  dinet-registry/     реестр чек-пойнтов: append-only лог с нотарными печатями,
                      кусочная репликация дельт (std-only)
  dinet-node/         узел/рой: DiLoCo-эры, secure-agg, robust-скрининг, fork-choice,
                      симуляция роя (in-memory mesh + файловый спул)
  dinet-memory/       локальная память: append-only KV-лог (std-only, замена SQLite)
                      + hash-эмбеддинг RAG (замена sqlite-vec)
  dinet-chat/         чат-сессия: история, RAG-контекст хода, буфер speak→learn;
                      `ChatModel` — точка прикрепления llama.cpp
  dinet-client/       клиент: чат → speak→learn → локальный fit → подписанный вклад
                      в рой; webchat (браузерная оболочка) и clientswarm
  dinet-p2p/          РЕАЛЬНАЯ сеть: libp2p (TCP+Noise+Yamux+gossipsub) под кадрами
                      dinet-proto; GossipBridge реализует трейт Transport
  dinet-train/        train-cpp: дообучение настоящей llama.cpp — LoRA-дельта головы,
                      экспорт в adapter-GGUF
  dinet-llama/        llama.cpp (GGUF Q4): `LlamaChatModel` — тот же трейт `ChatModel`
  dinet-agent/        M2-АГЕНТ (std-only): MCP-сервер (stdio NDJSON, спека 2025-06-18)
                      поверх FS-песочницы (jail) и CLI-allowlist; capability-токены
                      READ/WRITE/DESTRUCTIVE/NETWORK; undo-журнал; анти-фишинг-детектор
                      входа; полный аудит-журнал; Evolving Skills (SKILL_SHARE);
                      CUA-Lite бенчмарк. bin: agentd, cua-bench
prototype/            исполняемый Питон-прототип: кодирование сообщений + демо Secure Agg
PROMPT.md             исходный мега-промт проектирования
```

## Сборка

```bash
cargo build          # std + ed25519-dalek (подписи wire-кадров — настоящие)
cargo test           # юнит-тесты всех крейтов
python3 prototype/proto.py   # демо: подпись→CBOR→secure-aggregation
cargo run -p dinet-node      # демо роя: 20 узлов, 10% отравителей, robust-агрегация, чек-пойнты
# рой в РАЗНЫХ процессах (обмен только файлами, без сокетов):
rm -rf /tmp/swarm && \
cargo run -p dinet-node --bin spoolswarm -- /tmp/swarm --pid 0 --procs 2 --nodes 4 --eras 4 &
cargo run -p dinet-node --bin spoolswarm -- /tmp/swarm --pid 1 --procs 2 --nodes 4 --eras 4
                             # оба процесса сходятся к одним чек-пойнтам
# РАСПРЕДЕЛЁННОЕ ОБУЧЕНИЕ: клиенты чата = учители роя, каждый учится на СВОИХ диалогах:
rm -rf /tmp/opencode/clients && \
cargo run -p dinet-client --bin clientswarm -- /tmp/opencode/clients --pid 0 --procs 2 --clients 4 --eras 8 &
cargo run -p dinet-client --bin clientswarm -- /tmp/opencode/clients --pid 1 --procs 2 --clients 4 --eras 8
                             # юзер говорит → локальный fit → вклад в эру → чек-пойнт 0.08→0.85
# ГРАФИЧЕСКАЯ ОБОЛОЧКА: чат в браузере + вклад клиента в рой с живого ввода:
cargo run -p dinet-client --bin webchat -- ./dinet-web-mem --port 8080
                             # открой http://localhost:8080 (std-only, страница вшита в бинарник;
                             # Tauri 2 позже завернёт ЭТУ ЖЕ страницу в окно без переписывания)
                             # кнопка «Учить» = пара учителя → каждые 4 пары локальная эра роя
                             # → REVEAL-эра принимает чек-пойнт → ЧАТ ОТВЕЧАЕТ ОБУЧЕННЫМ РЕПЛИКОЙ
                             # (реплика переживает рестарт: ./dinet-web-mem.roi)
# GUI + ^НАСТОЯЩАЯ LLM (llama.cpp): свободные ходы — GGUF, микроязык — рой;
# ответ СТРИМИТСЯ в браузер токен-за-токеном (HTTP/1.0 до закрытия, fetch getReader):
cargo run -p dinet-client --features llama --bin webchat -- \
    ./dinet-web-mem --port 8080 --model /tmp/opencode/gemma270m.gguf
                             # русский свободно отвечает llama.cpp («Ты кто?»),
                             # слова «ай би ви…» отвечает реплика роя; обучение
                             # на парах учителя и эры работают так же (stats: model=…)
# РЕАЛЬНАЯ P2P-СЕТЬ (libp2p): TCP+Noise+Yamux+gossipsub на localhost, узлы с ed25519-ключами:
cargo run -p dinet-p2p --bin p2pmesh -- 6
                             # 6 узлов, полный меш, каждое сообщение проходит ed25519-проверку
                             # (кадры dinet-proto поверх gossipsub; подделка невозможна)
# РОЙ ПОВЕРХ РЕАЛЬНОЙ СЕТИ: полные фазы эр (publish→vote→tally) через GossipBridge (трейт Transport):
cargo run -p dinet-p2p --bin p2pswarm -- 4 2
                             # 4 узла × 2 эры: реплики БИТ-В-БИТ равны in-memory симуляции —
                             # сетевой транспорт не меняет математику роя (mean acc 0.542)
# РЕАЛЬНАЯ НЕЙРОСЕТЬ llama.cpp (GGUF): настоящая генерация на CPU:
cargo run -p dinet-llama --bin llamachat -- /tmp/opencode/gemma270m.gguf "Привет! 2+2=?"
                             # gemma-3-270m Q8 (реальный llama.cpp, — модель качается с HuggingFace);
                             # LlamaChatModel реализует ТОТ ЖЕ трейт ChatModel, что и
                             # микроязыковая сеть роя (webchat/Tauri подложит её флагом)
cargo run -p dinet-memory --bin memdemo -- ./dinet-mem
                             # REPL памяти: /remember | /ask | /forget | /stats
cargo run -p dinet-chat --bin chatdemo -- ./dinet-chat-mem
                             # REPL-чат: история, RAG-контекст, пары speak→learn
# M2-АГЕНТ: MCP-сервер поверх FS-песочницы (agentd — NDJSON/stdio, спека 2025-06-18):
cargo run -p dinet-agent --bin agentd -- /tmp/opencode/agent-jail --caps DESTRUCTIVE --allow echo
                             # процесс = готовый MCP-сервер для любого клиента; stdout = протокол,
                             # логи в stderr; 12 инструментов: fs_read/fs_list/fs_write/fs_delete/
                             # undo/cli_run/audit_tail + skill_save/skill_run/skill_share/
                             # skill_import/skill_list; права = max(--caps, подписанный --token)
# живая сессия (хэндшейк → запись → удаление → undo вернул файл):
printf '%s\n' \
 '{"jsonrpc":"2.0","id":1,"method":"initialize","params":{}}' \
 '{"jsonrpc":"2.0","method":"notifications/initialized"}' \
 '{"jsonrpc":"2.0","id":2,"method":"tools/call","params":{"name":"fs_write","arguments":{"path":"док/привет.txt","content":"M2 агент жив"}}}' \
 '{"jsonrpc":"2.0","id":3,"method":"tools/call","params":{"name":"fs_delete","arguments":{"path":"док/привет.txt"}}}' \
 '{"jsonrpc":"2.0","id":4,"method":"tools/call","params":{"name":"undo","arguments":{}}}' \
 | cargo run -p dinet-agent --bin agentd -- /tmp/opencode/agent-jail --caps DESTRUCTIVE
                             # секрет-пути (.env, id_rsa…) и эксфильтрация (webhook.site, | curl)
                             # бракуются АНТИ-ФИШИНГом ДО исполнения даже READ-агенту;
                             # ВСЕ вызовы (включая отказы и чтения) — в agent-audit.roi;
                             # навыки подписаны ключом агента (agent-key.seed, 0600) и
                             # переносятся конвертом SKILL_SHARE v1 (подделка отбрасывается)
# обучение модели анти-фишинга AgentPhishing v1 (логрегресс на детерминированном
# корпусе); agentd подхватывает файл agent-phishing.model из джейла автоматически —
# модель = «второе мнение» (поднять до Suspicious), блокируют по-прежнему правила:
cargo run -p dinet-agent --bin phishing-train -- /tmp/opencode/agent-jail/agent-phishing.model
# CUA-Lite бенчмарк: открытые задачи агенту поверх ToolWorld (v0-политика — точка
# прикрепления LLM; сам набор задач и чекеров — то, что измеряется):
cargo run -p dinet-agent --bin cua-bench
```

## Ключевые документы

- `docs/ARCHITECTURE.md` — вся система, от топологии до бюджета ресурсов.
- `docs/RESILIENCE.md` — исчерпывающая матрица «что сломали → что произойдёт».
- `docs/PROTOCOL.md` — wire-спецификация v0.
- `docs/ROADMAP.md` — фазы M0–M4 с Definition of Done.

## Статус

- **M0 закрыт** — память (KV-лог + RAG), чат-сессия, speak→learn (DPO-lite), локальный fit.
- **M1 закрыт** — рой: DiLoCo-эры, secure-agg, robust-агрегация, fork-choice; РЕАЛЬНАЯ
  сеть libp2p (p2pswarm сходится бит-в-бит с симуляцией); реестр чек-пойнтов
  с нотарными печатями и кусочной репликацией.
- **M2-инфосфера закрыта** — клиенты учат рой со своих диалогов (clientswarm),
  многоузловые эры через мост, LoRA-диссеминация через рой; webchat: браузерный
  чат с сетевыми эрами (`--net`) и настоящей llama.cpp (`--features llama`).
- **M2-Агент в пути** — MCP-сервер (12 инструментов), capability-токены, undo-журнал,
  анти-фишинг (rules v1 + тренированная модель AgentPhishing v1: логрегресс ловит
  перефразировки — «второе мнение», поднимает до Suspicious), полный аудит,
  Evolving Skills (SKILL_SHARE v1), CUA-Lite харнесс — рабочие и живо-проверяемые;
  далее браузер-CDP/UIA-серверы, DoD-бенчмарк с LLM-политикой.
- Все крейсы зелёные (тесты), warnings 0, AGPL-3.0 ядро. Подробности — `docs/ROADMAP.md`.
