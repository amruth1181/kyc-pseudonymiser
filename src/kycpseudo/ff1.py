"""FF1 format-preserving encryption (NIST SP 800-38G), AES-based.

Used for identifiers (document and account numbers): the output keeps the
length and alphabet of the input, and the key holder can decrypt it.
FF3/FF3-1 are deliberately not offered: NIST's 2025 draft revision withdrew them.

Verified against the NIST sample vectors in tests/test_ff1.py.
"""
from __future__ import annotations

import math

from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes

ALPHABET_36 = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"


class FF1:
    def __init__(self, key: bytes, radix: int = 10, alphabet: str | None = None):
        if len(key) not in (16, 24, 32):
            raise ValueError("AES key must be 16, 24 or 32 bytes")
        self.radix = radix
        self.alphabet = alphabet or ALPHABET_36[:radix]
        if len(self.alphabet) != radix:
            raise ValueError("alphabet length must equal radix")
        self._ecb = Cipher(algorithms.AES(key), modes.ECB())

    # --- primitives -------------------------------------------------------
    def _ciph(self, block: bytes) -> bytes:
        enc = self._ecb.encryptor()
        return enc.update(block) + enc.finalize()

    def _prf(self, data: bytes) -> bytes:
        """CBC-MAC with a zero IV: the last ciphertext block."""
        y = bytes(16)
        for i in range(0, len(data), 16):
            y = self._ciph(bytes(a ^ b for a, b in zip(y, data[i:i + 16])))
        return y

    def _num(self, s: str) -> int:
        n = 0
        for ch in s:
            n = n * self.radix + self.alphabet.index(ch)
        return n

    def _str(self, x: int, m: int) -> str:
        out = []
        for _ in range(m):
            x, r = divmod(x, self.radix)
            out.append(self.alphabet[r])
        return "".join(reversed(out))

    # --- FF1 ----------------------------------------------------------------
    def _crypt(self, x: str, tweak: bytes, decrypt: bool) -> str:
        n, t, radix = len(x), len(tweak), self.radix
        if radix ** n < 1_000_000:
            raise ValueError("domain too small for FF1 (radix**len must be >= 1,000,000)")
        u, v = n // 2, n - n // 2
        a, b = x[:u], x[u:]
        bb = math.ceil(math.ceil(v * math.log2(radix)) / 8)
        d = 4 * math.ceil(bb / 4) + 4
        p = bytes([1, 2, 1]) + radix.to_bytes(3, "big") + bytes([10, u % 256]) + n.to_bytes(4, "big") + t.to_bytes(4, "big")
        rounds = range(9, -1, -1) if decrypt else range(10)
        for i in rounds:
            src = a if decrypt else b
            q = tweak + bytes((-t - bb - 1) % 16) + bytes([i]) + self._num(src).to_bytes(bb, "big")
            r = self._prf(p + q)
            s = r
            j = 1
            while len(s) < d:
                s += self._ciph(bytes(x ^ y for x, y in zip(r, j.to_bytes(16, "big"))))
                j += 1
            y = int.from_bytes(s[:d], "big")
            m = u if i % 2 == 0 else v
            if decrypt:
                c = (self._num(b) - y) % (radix ** m)
                b, a = a, self._str(c, m)
            else:
                c = (self._num(a) + y) % (radix ** m)
                a, b = b, self._str(c, m)
        return a + b

    def encrypt(self, x: str, tweak: bytes = b"") -> str:
        return self._crypt(x, tweak, decrypt=False)

    def decrypt(self, x: str, tweak: bytes = b"") -> str:
        return self._crypt(x, tweak, decrypt=True)
