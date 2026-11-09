# PROTOCOL — DI-NET Wire Protocol v0

Канонический референс: реализация в `crates/dinet-proto/` (источник правды),
этот документ — описание. Кодировщик: **CBOR-lite** (детерминированный: поли-префиксная
схема, без контекстов).

## 0. принципы

- Каждый пакет: `version(1B) | type(1B) | sender DID-key (32B) | sig(64B) | payload`.
  Подпись — ed25519 от `type||payload||sender`; проверка на приёме обязательна.
- Все числа little-endian. Строки — UTF-8, length-prefixed.
- Контент-адресация: `sha32` (32 байта SHA-256) во всей сети.

## 1. Заголовок (Header)

| поле | тип | описание |
|---|---|---|
| version | u8 = 0 | версия протокола |
| type | u8 | тип сообщения (см. §2) |
| sender | 32B | ed25519 pubkey (did:key) |
| sig | 64B | подпись |
| payload | bytes | типозависимо |

## 2. Сообщения (коды типов)

| код | имя | payload |
|---|---|---|
| 0x01 | `HELLO` | u16 proto_ver; u8[4] caps (roles); u32 epoch_time; 2B port; PeerInfo[] |
| 0x02 | `HELLO_ACK` | PeerInfo[] (наши адреса) |
| 0x10 | `TRAIN_OFFER` | era_id(32B), adapter_hash(32B), dp_eps(u8 x10), size_bytes(u32), steps(u32) |
| 0x11 | `UPD_SHARE` (SecureAgg-share) | era_id; agg_id(32B); c_share(i64[N], покомпонентно, зашифрованно pairwise-маской); nonce(16B) |
| 0x12 | `POW-TR` (proof-of-training) | min_train_swarm-seconds, подпись времени, hash-LM-snapshot (merkle) |
| 0x20 | `AGG_COMMIT` | aggregator_id, era_id, agg_result_hash(32B), num_contrib(u16), кворум-подписи |
| 0x21 | `ROBUST_REPORT` | Krum/Trimmed-нормы, FedGraph-score (f32), флаги выбраковки |
| 0x22 | `ROBUST_INPUT` | era_id; delta (f32[N] LE, DP'd LoRA-дельта); norm (f32, L2 до клипа); steps (u32, claim PoW) |
| 0x23 | `LAZY_AUDIT` | auditor_id, checkpoint_id, ppl_before/after(f32), entropy(f32), verdict |
| 0x30 | `CHECKPOINT_ANNOUNCE` | id(32B), parent_id(32B), era_id, merge_recipe_hash(32B), eval-метрики (CBOR), watermark_blob, notary-кворум |
| 0x31 | `EVAL_VOTE` | evaluator_id, checkpoint_id, ppl(f32), pass@k(u8), sig |
| 0x40 | `SKILL_SHARE` | skill_hash(32B), abi_version, opaque skill-описание (без приватных данных), repsig |
| 0x50 | `EVAL_VOTE-REQ` | checkpoint_id, датасет-chunk-ссылки |
| 0x60 | `REP_GOSSIP` | pair (peer_a, peer_b, score i8 окна 30д) — housekeeping |

## 3. Раунды и «эры»

- `era_id = SHA256(parent_checkpoint_id || floor(unix_time / 3600))` — часовой счётчик.
- Внутри эры: window ≥50 узлов → aggregator-кворум → checkpoint.
- Если window не набран: era переносится (апдейты копятся, трафик равен 0).

## 4. Secure Aggregation (Bonawitz-2017, краткое)

1. Пары соседей (из DHT-«клику») обмениваются u64 seeds Ab и Ba → секреты.
2. Клиент i понимает маску PRG-поток (ChaCha по секретам), вычитает вклад пар-соседей:
   `share_i = θ_i + Σ_j f(s_ij) − Σ_k f(s_ki)`, где θ_i — обрезанный+квантованный апдейт.
3. Off-line узлы не закрывают раунд: их «левые» вклады закрываются shamir-prep (v2).
4. Aggregator видит сумму без индивидуальных вкладов (нулевая сумма масок).

## 5. Дельтизация весов

- Checkpoint = DAG по parent. узел подтягивает только «различные» чанки
  своего текущего анцестра → обычно 1–2% чанков, редко больше 50 МБ.
- Каждая дельта: content-addressed, в BitTorrent-friendly схеме (пир может
  стать автоматически «сиидером» последних 2 чек-пойнтов, лимит дисковый выбирает
  пользователь: 4/8/16 ГБ кеша).

## 6. Ограничение размера

- `UPD_SHARE` payload ≤ 24 МБ (per-element qint8/df-префикс + varint-сжатие).
- `SKILL_SHARE` ≤ 256 КБ. Что не влезло — за content-addressed ссылку.

## 7. Репутация

- Локальные вопросики метрик: «апдейт дошёл? разнообразие? норма адекватна?»
- score −100..+100 с экспоненциальным старением за 30 дней; новый peer за одну эру существенно не сдвинется.
- Репутация не переносимая (нет трансакций), только локальная вероятность выбора
  будущего aggregator-а. Сибилла-ферма с новыми ключами — стартует с 0 → не влияет.
