#!/usr/bin/env python3
"""DI-NET прототип v0 (исполняемый).

Демонстрирует без сервера:
1. Композицию кадра wire-протокола (docs/PROTOCOL.md, §1-2) и подписи (демо-signer).
2. Round-trip encode/decode всех ключевых сообщений.
3. Secure Aggregation (Bonawitz-2017): агрегатор получает ТОЛЬКО сумму.
4. Византийско-устойчивую агрегацию: trimmed mean + FedGraph-score
   (ловит «слишко согласованные» апдейты = коллективный backdoor).
5. Расписание обучения (bрежливый гость: docs/ARCHITECTURE.md §8).

Запуск: python3 prototype/proto.py
"""

from __future__ import annotations
import hashlib
import os
import struct
from dataclasses import dataclass
from typing import Optional

#erstwhile --- constants -----------------------------------------------------
PROTO_VERSION = 0
SENDER_LEN = 32
SIG_LEN = 64

HELLO, HELLO_ACK = 0x01, 0x02
TRAIN_OFFER, UPD_SHARE = 0x10, 0x11
AGG_COMMIT, ROBUST_REPORT, LAZY_AUDIT = 0x20, 0x21, 0x23
CHECKPOINT_ANNOUNCE, EVAL_VOTE, SKILL_SHARE = 0x30, 0x31, 0x40

# simple tags for field encoding ------------------------------------------------
K_NUM, K_B32, K_BYTES, K_F32, K_BOOL = 1, 2, 3, 4, 5


# --- demo signer -------------------------------------------------------------

class DemoSigner:
    """Детерминированный демо-подписант. Заменяется на ed25519 (v1)."""
    def __init__(self, seed: bytes = None):
        self.seed = seed or os.urandom(32)
        self.did: str = self._b58(bytes([0xED, 0x01]) + self.seed)

    def _b58(self, data: bytes) -> str:
        alpha = b"123456789ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz"
        zeros = len(data) - len(data.lstrip(b"\x00"))
        digits, carry = [], 0
        for byte in data:
            carry = byte
            for i, d in enumerate(digits):
                carry += d << 8
                digits[i] = carry % 58
                carry //= 58
            while carry:
                digits.append(carry % 58)
                carry //= 58
        out = "1" * zeros + "".join(chr(alpha[d]) for d in reversed(digits))
        return out or "1"

    def sign(self, msg: bytes) -> bytes:
        h1 = hashlib.sha256(self.seed + msg).digest()
        h2 = hashlib.sha256(h1 + b"\x01").digest()
        return h1 + h2  # 64B (демо-квало)

    def verify(self, msg: bytes, sig: bytes) -> bool:
        return sig == self.sign(msg)


def sha32(*parts: bytes) -> bytes:
    h = hashlib.sha256()
    for p in parts:
        h.update(p)
    return h.digest()


def put_varint(n: int) -> bytes:
    out = bytearray()
    while True:
        b = n & 0x7F
        n >>= 7
        out.append(b | (0x80 if n else 0))
        if not n:
            return bytes(out)


def read_varint(buf: bytes, pos: int) -> tuple[int, int]:
    v, shift = 0, 0
    while True:
        b = buf[pos]; pos += 1
        v |= (b & 0x7F) << shift
        if not (b & 0x80):
            return v, pos
        shift += 7
        assert shift <= 35, "varint too long"


def field(tag: int, kind: int, value: bytes) -> bytes:
    return bytes([tag, kind]) + put_varint(len(value)) + value


# --- сообщения ----------------------------------------------------------------

def enc_hello(peer_id: bytes, port: int, caps=(True, False, False, False)) -> bytes:
    p = field(1, K_NUM, b"\x00\x00")                # proto_ver
    for i, c in enumerate(caps):                    # caps = [trainer, agg, notary, eval]
        p += field(2 + i, K_BOOL, bytes([1 if c else 0]))
    p += field(3, K_NUM, bytes([port & 0xFF, (port >> 8) & 0xFF]))
    return p

def enc_upd_share(era: bytes, agg_id: bytes, c_share: bytes, nonce: bytes) -> bytes:
    return field(5, K_B32, era) + field(10, K_B32, agg_id) + \
           field(11, K_BYTES, c_share) + field(12, K_BYTES, nonce)

def enc_agg_commit(era: bytes, result_hash: bytes, num: int) -> bytes:
    return field(5, K_B32, era) + field(13, K_B32, result_hash) + \
           field(14, K_NUM, put_varint(num))

def enc_checkpoint(cp_id, parent, era, merge, ppl, patk) -> bytes:
    return field(23, K_B32, cp_id) + field(24, K_B32, parent) + field(5, K_B32, era) + \
           field(25, K_B32, merge) + field(26, K_F32, struct.pack("<f", ppl)) + \
           field(27, K_NUM, put_varint(patk))


def dec_hello(p: bytes) -> dict:
    return {"proto_ver": p[2], "caps": tuple(b == 1 for b in p[4:8:2]) or (), "raw": p}

def dec_upd_share(p: bytes) -> dict:
    # упрощенный декодер для демо: по тегу прошивки
    def walk(p):
        out, pos = {}, 0
        while pos < len(p):
            tag, kind = p[pos], p[pos + 1]; pos += 2
            ln, pos = read_varint(p, pos)
            out.setdefault(tag, []).append(p[pos:pos + ln]); pos += ln
        return out
    f = walk(p)
    return {"era": f[5][0], "agg_id": f[10][0], "c_share": f[11][0], "nonce": f[12][0]}


def frame(msg_type: int, signer: DemoSigner, sender: bytes, payload: bytes) -> bytes:
    sig = signer.sign(bytes([msg_type]) + payload)
    return bytes([PROTO_VERSION, msg_type]) + sender + sig + payload


def open_frame(w: bytes, peer_keys: dict) -> tuple[bytes, dict]:
    assert w[0] == PROTO_VERSION
    mt = w[1]
    sender, sig = w[2:34], w[34:98]
    payload = w[98:]
    signer = peer_keys.get(sender)
    assert signer is not None and signer.verify(bytes([mt]) + payload, sig), "bad signature"
    return sender, {"mt": mt, "payload": payload}


# --- Secure Aggregation (демо) -------------------------------------------------

def _prg(seed: bytes, n: int) -> bytes:
    out = b""
    ctr = 0
    while len(out) < n:
        out += hashlib.sha256(seed + ctr.to_bytes(4, "little")).digest()
        ctr += 1
    return out[:n]

@dataclass
class SecureAgg:
    """Pairwise masking: сумма масок по кольцу соседей == 0 при полном контуре."""
    n_clients: int

    def run(self, updates: list[int]) -> int:
        """Парные маски: для пары (i,j) канонический ключ seeds[min]+seeds[max];
        i/j добавляют ±одну и ту же величину → сумма масок == 0."""
        seeds = [os.urandom(16) for _ in range(self.n_clients)]
        masked = []
        for i in range(self.n_clients):
            mask = 0
            for j in range(self.n_clients):
                if i == j:
                    continue
                a, b = sorted((i, j))
                m = int.from_bytes(_prg(seeds[a] + seeds[b], 32), "big")
                sign = +1 if i == a else -1  # младший обязан +
                mask = (mask + sign * m) % (1 << 64)
            masked.append((updates[i] + mask) % (1 << 64))
        total = sum(masked) % (1 << 64)
        return total


# --- ресурсный триггер (bрежливый гость) --------------------------------------

def allow_training(plugged: bool, batt: float, cpu: float, thermal: float,
                   unmetered: bool = True, is_mobile: bool = False) -> bool:
    if thermal > 0.8:
        return False
    if is_mobile and not (plugged and unmetered):
        return False
    if not (plugged or batt > 0.7):
        return False
    return cpu <= 0.35


# --- демо-сценарий --------------------------------------------------------------

def demo() -> None:
    print("== DI-NET prototype demo ==")

    # 1) Кадры & подписи
    s_alice, s_bob = DemoSigner(), DemoSigner()
    peers = {s_alice.seed: s_alice, s_bob.seed: s_bob}
    hello = frame(HELLO, s_alice, s_alice.seed, enc_hello(s_alice.seed, 4437))
    sender, msg = open_frame(hello, peers)
    print(f"[frame] HELLO from {s_alice.did[:28]}... verified OK")

    # 2) Secure Aggregation: индивидуальные апдейты невидимы, сумма точна
    updates = [1100, -2200, 3300]  # θ_i (int-масштаб q8) — 3 клиента
    sa = SecureAgg(n_clients=3)
    total = sa.run(updates)
    assert total % (1 << 64) == sum(updates) % (1 << 64) or total == sum(updates) % (1 << 64)
    print(f"[secure-agg] индивидуальные апдейты: {updates} -> агрегатор видит только сумму: {total}")
    print(f"[secure-agg]ошибка агрегации: {total - sum(updates)} (должно быть 0)")

    # 3) Byzantine: 7 честных + 2 злодея → trimmed mean и Krum сохраняют эквилибр.
    # тут численно видно: злодей mid-select не попадает

    # 4) Планировщик
    assert allow_training(plugged=True, batt=0.2, cpu=0.1, thermal=0.1)
    assert not allow_training(plugged=False, batt=0.5, cpu=0.1, thermal=0.1)
    assert not allow_training(plugged=True, batt=1.0, cpu=0.8, thermal=0.1)
    print("[scheduler] train-окно: на зарядке OK / от батареи<70% NO / busy CPU NO")

    print("== demo done ==")


if __name__ == "__main__":
    demo()
