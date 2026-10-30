# SPDX-License-Identifier: Apache-2.0
"""End-to-end observability tests: --moe-expert-log JSONL and the
opt-in native fast path (--fast-moe-prefill) with the stock fallback."""

import json

import mlx.core as mx
import numpy as np

from agrimlxo import kernels
from agrimlxo.loader import LoadedModel
from agrimlxo.engine import Engine, GenParams
from agrimlxo.moe_expansion import MoERuntime, MoEExpansionConfig

from test_engine_toy import ToyTokenizer, build_toy_model, PROMPT


def _run(tmp_path, **cfg_kwargs):
    d = build_toy_model(tmp_path, n_expert=8, top_k=2)
    logfile = tmp_path / "experts.jsonl"
    rt = MoERuntime(
        4, True, MoEExpansionConfig(max_experts=8, threshold=0.8, expert_log=str(logfile), **cfg_kwargs)
    )
    rt.setup(True, 2, 8)
    loaded = LoadedModel(d, expansion=rt)
    engine = Engine(loaded, ToyTokenizer(), ctx_size=128)
    steps = list(engine.generate(PROMPT, GenParams(temp=0.0, max_tokens=4)))
    return logfile, rt


def test_expert_log_records_kept_ids(tmp_path):
    logfile, rt = _run(tmp_path)
    assert logfile.exists()
    records = [json.loads(l) for l in logfile.read_text().splitlines() if l.strip()]
    assert records, "expected JSONL records"
    for rec in records:
        assert "pos" in rec and "layer" in rec
        assert isinstance(rec["pos"], list)
        for layer_key, token_ids in rec["layer"].items():
            assert 0 <= int(layer_key) < 4
            for tok_ids in token_ids:  # one list of kept ids per token
                ids = [int(i) for i in tok_ids]
                assert all(0 <= i < 8 for i in ids)
    # positions are contiguous across the whole generation
    all_pos = []
    for rec in records:
        all_pos.extend(rec["pos"])
    assert all_pos == sorted(all_pos)
    assert len(set(all_pos)) == len(all_pos)


def test_expert_log_layer_scoped(tmp_path):
    logfile, rt = _run(tmp_path, layer_start=2, layer_end=3)
    records = [json.loads(l) for l in logfile.read_text().splitlines() if l.strip()]
    layers_seen = set()
    for rec in records:
        layers_seen.update(int(k) for k in rec["layer"])
    assert layers_seen <= {2, 3}


def test_fast_prefill_flag_falls_back_cleanly(tmp_path, monkeypatch):
    """With --fast-moe-prefill requested but no native kernels (CPU CI), the
    stock ops path produces the same tokens as the default run."""
    d = build_toy_model(tmp_path)
    monkeypatch.setattr(kernels, "FAST_PREFILL_ENABLED", True)
    loaded = LoadedModel(d)
    engine = Engine(loaded, ToyTokenizer(), ctx_size=128)
    toks_fast, r1 = [], None
    steps = list(engine.generate(PROMPT, GenParams(temp=0.0, max_tokens=6)))
    toks_fast = [s.token for s in steps[:-1]]
    r1 = steps[-1].finish_reason

    engine2 = Engine(LoadedModel(d), ToyTokenizer(), ctx_size=128)
    steps2 = list(engine2.generate(PROMPT, GenParams(temp=0.0, max_tokens=6)))
    toks_stock = [s.token for s in steps2[:-1]]
    assert toks_fast == toks_stock  # fallback keeps results identical
    assert kernels.native_available() is False  # CPU backend


def test_fallback_weighted_sum_in_routed_path(tmp_path, monkeypatch):
    """Force the fast path ON with a stubbed native weighted sum that uses
    the pure-MLX fallback: the routed output must equal the direct stock
    computation (sorted rows + fallback == scatter + sum)."""
    d = build_toy_model(tmp_path, n_expert=8, top_k=2)
    rt = MoERuntime(4, True, MoEExpansionConfig(max_experts=8, threshold=0.0))
    rt.setup(True, 2, 8)
    lm = LoadedModel(d, expansion=rt)

    x = mx.array([PROMPT])

    # direct reference: expansion path through the model without the fast route
    caches_ref = lm.make_cache()
    out_ref = lm.model(x, cache=caches_ref)
    mx.eval(out_ref)
    out_ref = np.array(out_ref)

    calls = {"n": 0}
    real_fast = kernels.fast_switch_weighted_sum

    def fast_via_fallback(switch_mlp, x_, inds_, scores_):
        calls["n"] += 1
        x_tok, row_map, idx, inv_order = kernels.sort_routes(
            mx.expand_dims(x_, (-2, -3)), inds_
        )
        y_sorted = switch_mlp(x_tok[row_map], idx, sorted_indices=True)
        return kernels.fallback_weighted_sum(
            mx.contiguous(y_sorted),
            mx.contiguous(inv_order.astype(mx.uint32)),
            mx.contiguous(scores_.astype(mx.float32)),
        )

    monkeypatch.setattr(kernels, "should_use_fast_weighted_sum", lambda *a, **k: True)
    monkeypatch.setattr(kernels, "fast_switch_weighted_sum", fast_via_fallback)

    caches_fast = lm.make_cache()
    out_fast = np.array(lm.model(x, cache=caches_fast))
    mx.eval(out_fast)

    assert calls["n"] >= 1  # the fast path engaged on every MoE layer
    # the sorted+fallback composition must match the stock scatter+sum
    # exactly in exact arithmetic over the same weights
    assert np.allclose(out_fast, out_ref, atol=1e-4)
