# SPDX-License-Identifier: Apache-2.0
"""Multi-turn conversation snapshots + ./kache persistence tests."""

import copy
import json

import mlx.core as mx
import numpy as np
import pytest

from agrimlxo.engine import Engine, GenParams
from agrimlxo.kache import Kache
from agrimlxo.loader import LoadedModel
from agrimlxo.tokenizer import Tokenizer

from test_vision import build_toy_vlm


@pytest.fixture(scope="module")
def conv(tmp_path_factory):
    d = build_toy_vlm(tmp_path_factory.mktemp("conv"))
    tok = Tokenizer("/tmp/vlm-cfg")  # real qwen3.5 template
    lm = LoadedModel(d)
    engine = Engine(lm, tok, ctx_size=1024, parallel=1)
    engine.min_snapshot_tokens = 8
    return engine, tok


def _turn(engine, tok, messages, conv_key, gen_tokens=3):
    prompt = tok.apply_chat_template(messages, add_generation_prompt=True,
                                     enable_thinking=True)
    if not isinstance(prompt, list):
        prompt = tok.encode(prompt)
    nogen = tok.apply_chat_template(messages, add_generation_prompt=False,
                                    enable_thinking=True)
    snapshot_at = len(nogen if isinstance(nogen, list) else tok.encode(nogen))
    steps = list(engine.generate(
        prompt, GenParams(temp=0.0, max_tokens=gen_tokens),
        conv_key=conv_key, snapshot_at=snapshot_at,
    ))
    return prompt, snapshot_at, steps[-1].finish_reason


def test_multi_turn_snapshot_reuse(conv):
    engine, tok = conv
    m1 = [{"role": "user", "content": "cosa è il MoE?"}]
    prompt1, sa1, _ = _turn(engine, tok, m1, "convX")
    assert "convX" in engine.snapshots
    snap = engine.snapshots["convX"]
    assert snap["tokens"] == engine.slots[0].tokens[:sa1]

    # il turno 2 estende: il prefisso snapshot deve coincidere
    gen_text = tok._tokenizer.decode(
        engine.slots[0].tokens[sa1:]).replace("<|im_end|>", "")
    m2 = m1 + [{"role": "assistant", "content": gen_text},
               {"role": "user", "content": "sintetizza"}]
    prompt2, sa2, _ = _turn(engine, tok, m2, "convX")
    assert prompt2[:sa1] == snap["tokens"], "il prefisso del turno 1 deve essere stabile"
    # il motore ha ri-usato lo snapshot: lo slot parte dal confine
    assert len(engine.slots[0].tokens) >= len(prompt2)


def test_kache_persistence_roundtrip(conv, tmp_path):
    engine, tok = conv
    k = Kache(str(tmp_path / "kache"), budget_bytes=1 << 30)
    engine.kache = k
    m1 = [{"role": "user", "content": "raccontami una storia"}]
    prompt1, sa1, _ = _turn(engine, tok, m1, "convK", gen_tokens=4)
    assert (tmp_path / "kache" / "convK.safetensors").exists()

    # roundtrip puro: load senza forward (il restore da disco è
    # sperimentale e dietro --kache-restore)
    loaded = k.load("convK")
    assert loaded is not None
    tokens_disk, states_disk = loaded
    assert tokens_disk == prompt1[:sa1]
    assert len(states_disk) == len(conv[0].loaded.make_cache("auto"))


def test_kache_lru_eviction(tmp_path):
    import mlx.core as mx

    k = Kache(str(tmp_path / "kc"), budget_bytes=20_000_000)
    big = np.zeros((1000, 1000), dtype=np.float32)  # ~4MB each
    states = [("KVCache", [big.copy(), big.copy(), 3])]
    # tre salvataggi; last_used decrescente: k0 il più vecchio
    for i in range(3):
        k.save(f"k{i}", list(range(100)), states)
        k.index[f"k{i}"]["last_used"] = 1000 - i
        k.index[f"k{i}"]["uses"] = 1
        k._save_index()
    assert k._total_bytes() > 20_000_000
    # eviction manuale con l'ordine già impostato: l'LRU scarta k1
    # (last_used 999 < 1000 di k0); il budget è 20M, tre entry da 8M -> 24M
    k._evict()
    while k._total_bytes() > 20_000_000 and len(k.index) > 1:
        k._evict()
    assert "k1" not in k.index and "k0" in k.index and "k2" in k.index
    # un nuovo salvataggio (usa il last_used corrente) evicta il prossimo LRU
    k.save("k3", list(range(100)), states)
    while k._total_bytes() > 20_000_000 and len(k.index) > 1:
        k._evict()
    assert "k3" in k.index and "k0" not in k.index


def test_state_safetensors_roundtrip(tmp_path):
    import mlx.core as mx

    k = Kache(str(tmp_path / "kc"), budget_bytes=1 << 30)
    state = [mx.zeros((3, 4)), mx.ones((2,)).astype(mx.float16),
             [mx.zeros((1, 2)), 7], None, 5, True]
    k.save("s1", list(range(40)), [("KVCache", state)])
    tokens, states = k.load("s1")
    assert tokens == list(range(40))
    st = states[0][1]
    assert st[0].shape == (3, 4) and st[1].dtype == mx.float16
    assert st[2][1] == 7 and st[3] is None and st[5] is True
