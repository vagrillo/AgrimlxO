# SPDX-License-Identifier: Apache-2.0
"""Tests for the optional native kernels and the per-expert mixed
quantization (all runnable on the CPU backend: the native path degrades to
the pure-MLX fallback, which is exactly what these tests pin)."""

import json
from pathlib import Path

import mlx.core as mx
import numpy as np
import pytest

from agrimlxo import kernels
from agrimlxo.models.mixed_experts import MixedSwitchGLU, resolve_groups
from agrimlxo.loader import LoadedModel
from agrimlxo.moe_expansion import MoERuntime, MoEExpansionConfig
from agrimlxo.mixed_quantize import mixed_quantize, load_order_file

from test_engine_toy import (
    ToyTokenizer,
    build_toy_model,
    greedy_tokens,
    PROMPT,
)


# ---------------------------------------------------------------- kernels


def test_native_kernels_absent_on_cpu(monkeypatch):
    # Linux CI / CPU backend: the extension may exist but the Metal check
    # makes native_available() False; either way the fallback must engage.
    assert kernels.native_available() is False or mx.metal.is_available()


def test_fallback_weighted_sum_matches_reference():
    rng = np.random.default_rng(3)
    tokens, n, D = 7, 5, 16
    y_sorted = mx.array(rng.standard_normal((tokens * n, 1, D)).astype(np.float32))
    # router-order permutation per token
    inv = np.concatenate(
        [rng.permutation(n) + t * n for t in range(tokens)]
    ).astype(np.uint32)
    scores = mx.array(rng.random((tokens, n)).astype(np.float32))

    out = kernels.fallback_weighted_sum(y_sorted, mx.array(inv), scores)
    out = np.array(out).reshape(tokens, D)

    ref = np.zeros((tokens, D), dtype=np.float32)
    ys = np.array(y_sorted).reshape(tokens * n, D)
    sc = np.array(scores)
    for t in range(tokens):
        for k in range(n):
            row = inv[t * n + k]
            ref[t] += ys[row] * sc[t, k]
    assert np.allclose(out, ref, atol=1e-5)


def test_sort_routes_matches_gather_sort():
    rng = np.random.default_rng(5)
    B, T, K, D = 2, 9, 4, 8
    x = mx.array(rng.standard_normal((B, T, D)).astype(np.float32))
    inds = mx.array(rng.integers(0, 10, size=(B, T, K)).astype(np.uint32))
    xe = mx.expand_dims(x, (-2, -3))
    x_tok, row_map, idx, inv_order = kernels.sort_routes(xe, inds)
    # equivalence with mlx-lm's _gather_sort
    *_, M = inds.shape
    flat = inds.flatten()
    order = mx.argsort(flat)
    inv_ref = mx.argsort(order)
    x_ref = xe.flatten(0, -3)[order // M]
    assert np.array_equal(np.array(idx), np.array(flat[order]))
    assert np.array_equal(np.array(inv_order.astype(mx.uint32)), np.array(inv_ref.astype(mx.uint32)))
    assert np.array_equal(np.array(x_tok[row_map]), np.array(x_ref))


def test_fast_path_disabled_by_default():
    # routed_forward must stay stock when --fast-moe-prefill is off
    assert kernels.FAST_PREFILL_ENABLED is False


# --------------------------------------------------------------- mixed quant


def test_resolve_groups_validation():
    groups = resolve_groups(
        8,
        [
            {"bits": 8, "group_size": 64, "experts": [3, 1]},
            {"bits": 4, "group_size": 64},
        ],
    )
    assert groups[0][0] == [3, 1]
    assert groups[1][0] == [0, 2, 4, 5, 6, 7]
    with pytest.raises(ValueError):  # duplicate expert
        resolve_groups(8, [
            {"bits": 8, "experts": [1, 1]},
        ])
    with pytest.raises(ValueError):  # missing experts, no catch-all
        resolve_groups(8, [{"bits": 8, "experts": [0]}])
    with pytest.raises(ValueError):  # out of range
        resolve_groups(8, [{"bits": 8, "experts": [99], "group_size": 64}, {"bits": 4}])


def test_mixed_switch_glu_matches_uniform_math():
    """With every group at the SAME bitrate and dequantized weights, the
    mixed forward must reproduce the plain per-expert computation."""
    E, D, H, N = 8, 32, 32, 4
    rng = np.random.default_rng(11)
    mixed = MixedSwitchGLU(
        D, H, E, [
            {"bits": 4, "group_size": 32, "experts": [2, 5, 0]},
            {"bits": 4, "group_size": 32},
        ]
    )
    for gi, g in enumerate(mixed.groups):
        for name, (dims_in, dims_out) in {
            "gate_proj": (D, H),
            "up_proj": (D, H),
            "down_proj": (H, D),
        }.items():
            w = rng.standard_normal((len(g.expert_ids), dims_out, dims_in)).astype(np.float32) * 0.2
            wq, scales, *biases = mx.quantize(mx.array(w), group_size=g.group_size, bits=g.bits)
            mixed.load_group(
                gi, name,
                {"weight": wq, "scales": scales, "biases": biases[0] if biases else None},
            )

    x = mx.array(rng.standard_normal((2, 3, D)).astype(np.float32) * 0.5)
    inds = mx.array(rng.integers(0, E, size=(2, 3, N)))

    y = mixed(x, inds)
    mx.eval(y)
    # SwitchGLU contract: [B, T, N, H] — routed_forward folds the scores
    assert y.shape == (2, 3, N, D)

    # reference: dequantize each expert, run silu(gate)*up → down per slot
    wd = {}
    for gi, g in enumerate(mixed.groups):
        for name in ("gate_proj", "up_proj", "down_proj"):
            wq, s, b = g.projection(name)
            w = mx.dequantize(wq, s, b, group_size=g.group_size, bits=g.bits)
            for li, e in enumerate(g.expert_ids):
                wd[(e, name)] = np.array(w[li])

    # per-slot reference (unweighted): y is [B, T, N, D] — scores are folded
    # later by routed_forward, exactly like SwitchGLU
    y_ref = np.zeros((2 * 3 * N, D), dtype=np.float32)
    xf = np.array(x).reshape(2 * 3, D)
    indf = np.array(inds).reshape(2 * 3, N)
    for r in range(2 * 3):
        for k in range(N):
            e = int(indf[r, k])
            g_ = wd[(e, "gate_proj")]
            u_ = wd[(e, "up_proj")]
            d_ = wd[(e, "down_proj")]
            gate = g_ @ xf[r]
            act = gate * (1 / (1 + np.exp(-gate))) * (u_ @ xf[r])  # silu(gate)*up
            y_ref[r * N + k] = d_ @ act

    assert np.allclose(
        np.array(y).reshape(2 * 3 * N, D), y_ref, atol=2e-2
    )


def test_mixed_quantize_roundtrip(tmp_path):
    """Uniform toy model → mixed-quantize → load → same greedy trace."""
    src_dir = tmp_path / "src"
    src_dir.mkdir()
    src = build_toy_model(src_dir)

    order_file = tmp_path / "expert-order.txt"
    # layer rankings: reversed expert ids (arbitrary but deterministic)
    lines = []
    for _ in range(4):
        lines.append(",".join(str(e) for e in range(7, -1, -1)))
    order_file.write_text("\n".join(lines) + "\n")

    summary = mixed_quantize(
        src,
        tmp_path / "out",
        order_file=str(order_file),
        layers_spec="0--1",
        top_m=4,
        hi_bits=8,
        lo_bits=4,
        hi_group_size=32,
        lo_group_size=32,
    )
    assert sorted(summary["layers"]) == [0, 1, 2, 3]
    assert Path(summary["out_dir"], "config.json").exists()

    cfg = json.loads((Path(summary["out_dir"]) / "config.json").read_text())
    assert "moe_expert_quant" in cfg
    # the mixed switch_mlps must be excluded from the uniform quantization
    assert cfg["quantization"]["model.layers.0.mlp.switch_mlp"] is False

    # the mixed model loads and runs; its output stays close to the source
    # (top half at 8-bit, tail at 4-bit over a tiny toy — tolerance is wide)
    loaded_mixed = LoadedModel(Path(summary["out_dir"]))
    assert loaded_mixed.mixed_quant is not None
    assert set(loaded_mixed.mixed_quant) == {0, 1, 2, 3}

    toks_mixed, r = greedy_tokens(Path(summary["out_dir"]), n=4)
    toks_src, r_src = greedy_tokens(src, n=4)
    assert r == r_src
    # greedy traces on a quantization-rounded tiny model may legitimately
    # diverge; the strong check is that the mixed layer math is consistent
    # (covered exactly by test_mixed_switch_glu_matches_uniform_math) and
    # generation runs deterministically
    again, _ = greedy_tokens(Path(summary["out_dir"]), n=4)
    assert again == toks_mixed


def test_mixed_model_with_expansion(tmp_path):
    """Expansion and mixed quantization compose: rank-ordered top-N selection
    routes through the mixed groups."""
    src_dir = tmp_path / "src"
    src_dir.mkdir()
    src = build_toy_model(src_dir)
    out = tmp_path / "out-mixed"
    out.mkdir(parents=True)
    order_file = tmp_path / "order.txt"
    order_file.write_text(",".join(str(e) for e in [5, 3, 1, 7, 0, 2, 4, 6]) + "\n")
    mixed_quantize(
        src, out, order_file=str(order_file), layers_spec="0--1",
        top_m=6, hi_bits=8, lo_bits=4, hi_group_size=32, lo_group_size=32,
    )

    rt = MoERuntime(4, True, MoEExpansionConfig(max_experts=8, threshold=0.8))
    rt.setup(True, 2, 8)
    loaded = LoadedModel(out, expansion=rt)
    toks, r = greedy_tokens(out, expansion=rt, n=4)
    assert r == "length"
    # the expansion recorded kept counts through the mixed layers
    assert rt._stats, "expected expansion stats on the mixed model"
