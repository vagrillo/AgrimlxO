# SPDX-License-Identifier: Apache-2.0
"""Regression tests for the incremental detokenizer (tokenizer.py).

The engine's toy tokenizer stub returns empty strings, so the detokenizer
needs its own tests with a byte-accurate fake tokenizer that mimics how
real BPE tokenizers (Qwen) split multi-byte UTF-8 characters across tokens.
"""

import numpy as np
import pytest

from agrimlxo.tokenizer import Detokenizer


class ByteTokenizer:
    """Token id -> raw bytes; decode joins the bytes and utf-8 decodes with
    replacement — exactly the behavior that splits 'è' (0xC3 0xA8) or '中'
    across two tokens."""

    def __init__(self):
        self.vocab = {
            0: b"c",
            1: b"iao",
            2: b"\xa8",           # trailing byte of 'è'
            3: b"\xc3",           # leading byte of 'è' -> invalid alone
            4: b" ",
            5: b"come",
            6: b" ",
            7: b"stai",
            8: b"?", 
            9: b"\xe4\xb8",       # first 2 bytes of '中'
            10: b"\xad",          # last byte of '中'
            11: b"Ben", 12: b"e", 13: b"!",
        }

    def decode(self, tokens, skip_special_tokens=True):
        raw = b"".join(self.vocab[int(t)] for t in tokens)
        return raw.decode("utf-8", errors="replace")


def test_simple_word_tokens():
    d = Detokenizer(ByteTokenizer())
    out = ""
    for t in [0, 1, 4, 5, 6, 7, 8]:  # c|iao| |come| |stai|?
        out += d.push(t)
    assert out == "ciao come stai?"
    assert d.text == "ciao come stai?"


def test_split_utf8_char_is_held_then_emitted():
    d = Detokenizer(ByteTokenizer())
    emitted = []
    for t in [0, 1, 4, 3, 2]:  # c|iao| |<lead>|<trail>  -> 'ciao è'
        emitted.append(d.push(t))
    # the leading byte alone must emit nothing (replacement char held)
    assert emitted[3] == ""
    assert "".join(emitted) == "ciao è"
    assert d.text == "ciao è"


def test_three_byte_utf8_split():
    d = Detokenizer(ByteTokenizer())
    out = ""
    for t in [9, 10]:  # '中' split 2+1 bytes
        out += d.push(t)
    assert out == "中"


def test_no_piece_lost_or_duplicated():
    """The emitted stream must equal the full decode of the whole sequence —
    the bug seen on the Mac dropped pieces ('Caoome aiut' instead of
    'Come aiut...') because the buffer was compared against the global
    text."""
    d = Detokenizer(ByteTokenizer())
    seq = [11, 12, 4, 5, 6, 7, 8, 4, 0, 1]  # Ben|e| |come| |stai|?| |c|iao
    out = "".join(d.push(t) for t in seq)
    full = ByteTokenizer().decode(seq)
    assert out == full == "Bene come stai? ciao"


def test_finalize_flushes_held_bytes():
    d = Detokenizer(ByteTokenizer())
    out = d.push(3)          # lone leading byte -> held
    assert out == ""
    rest = d.finalize()      # incomplete sequence: nothing safe to emit
    assert rest == ""
    assert d.text == ""

    d2 = Detokenizer(ByteTokenizer())
    d2.push(3)
    d2.push(2)               # completes 'è'
    assert d2.finalize() == ""  # already flushed


def test_push_after_hold_completes_previous_prefix():
    """After a hold, the next decode extends the SAME buffer: the delta must
    be computed against the previous buffer decode, not the global text."""
    d = Detokenizer(ByteTokenizer())
    e1 = d.push(5)     # 'come'
    assert e1 == "come"
    e2 = d.push(3)     # leading byte -> held
    assert e2 == ""
    e3 = d.push(2)     # 'è' completes -> emit exactly ' è'? no space: 'è'
    assert e3 == "è"
    assert d.text == "comeè"


def test_reset():
    d = Detokenizer(ByteTokenizer())
    d.push(0)
    d.reset()
    assert d.text == ""
    assert d.push(11) == "Ben"
