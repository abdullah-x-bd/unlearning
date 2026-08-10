from __future__ import annotations

import hashlib
import re


def simhash64(text: str, shingle_size: int = 3) -> str:
    tokens = re.findall(r"\w+", text.lower())
    if not tokens:
        tokens = [text.lower()]
    if len(tokens) < shingle_size:
        shingles = [" ".join(tokens)]
    else:
        shingles = [" ".join(tokens[i : i + shingle_size]) for i in range(len(tokens) - shingle_size + 1)]
    vector = [0] * 64
    for shingle in shingles:
        value = int.from_bytes(hashlib.sha256(shingle.encode()).digest()[:8], "big")
        for bit in range(64):
            vector[bit] += 1 if (value >> bit) & 1 else -1
    result = 0
    for bit, score in enumerate(vector):
        if score >= 0:
            result |= 1 << bit
    return f"{result:016x}"


def hamming64(left: str, right: str) -> int:
    return (int(left, 16) ^ int(right, 16)).bit_count()
