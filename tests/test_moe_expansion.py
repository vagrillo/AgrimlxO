# SPDX-License-Identifier: Apache-2.0
"""Golden-case tests for the MoE expansion post-pass.

The expected values are hand-derived from the normative semantics of
``build_moe_expansion_weights`` (vagrillo/llama.cpp, src/llama-moe-expansion.h)
— the MLX port in agrimlxo.moe_expansion must match them exactly. These mirror
the golden checks of tests/test-moe-expansion.cpp in the llama.cpp fork.

Rules under test:
  * selection: keep = min(step(F - j) + step(w > T * p_ref), 1),
    F = max(1, N/4), p_ref = weight of rank N/2 (1-indexed, i.e. index N/2-1)
  * decay: ranks >= K get 0.99 + t*(D - 0.99), t = (j-K)/(N-K-1) clamped to
    [0,1]; a single extra rank gets the midpoint (0.99 + D)/2
  * renormalize: kept weights divided by their sum (min guard 2^-14)
"""

import math

import mlx.core as mx
import numpy as np
import pytest

from agrimlxo.moe_expansion import (
    MoEExpansionConfig,
    MoEReduceConfig,
    MoERuntime,
    expansion_post_pass,
    reduce_post_pass,
    validate_expansion,
    default_profile_for,
)


def run_post(w, N, K, T=0.0, D=0.5, no_decay=False, renorm=True):
    w_mx = mx.array(np.array([w], dtype=np.float32))  # [1, N]
    out, cnt = expansion_post_pass(w_mx, N, K, T, D, no_decay, renorm)
    mx.eval(out, cnt)
    return np.array(out)[0], np.array(cnt)[0]


def close(a, b, tol=1e-6):
    return np.allclose(a, b, rtol=0, atol=tol)


# ---------------------------------------------------------------- T=0 fixed N


def test_t0_keeps_all_n_and_applies_decay():
    w = [0.5, 0.3, 0.1, 0.07, 0.02, 0.01]
    N, K, D = 6, 2, 0.5
    out, cnt = run_post(w, N, K, T=0.0, D=D, renorm=False)
    assert cnt == 6
    # decay: j=2 -> 0.99, j=3 -> 0.99 + 1/3*(0.5-0.99), j=4 -> ..., j=5 -> 0.5
    t = np.clip((np.arange(N) - K) / (N - K - 1), 0, 1)
    fac = 0.99 + t * (D - 0.99)
    fac[:K] = 1.0
    assert close(out, np.array(w) * fac)


def test_t0_renorm_sums_to_one():
    w = [0.5, 0.3, 0.1, 0.07, 0.02, 0.01]
    out, cnt = run_post(w, 6, 2, T=0.0, D=0.5, renorm=True)
    assert cnt == 6
    assert abs(out.sum() - 1.0) < 1e-6
    # native ranks keep the output scale: after decay w2 = [0.5, 0.3, 0.099,
    # 0.07*fac3, 0.02*fac4, 0.01*0.5]; renorm rescales the whole set.
    raw = np.array([0.5, 0.3, 0.099, 0.07 * 0.8266667, 0.02 * 0.6633333, 0.005])
    assert close(out, raw / raw.sum())


# ---------------------------------------------------------------- threshold T


def test_threshold_power_law_cut():
    w = [0.4, 0.2, 0.1, 0.05, 0.03, 0.02]
    # p_ref = w[N/2 - 1] = w[2] = 0.1; keep while w > 0.8 * 0.1 = 0.08
    out, cnt = run_post(w, 6, 2, T=0.8, D=0.5, renorm=False)
    assert cnt == 3
    # dropped ranks carry zero weight; kept rank 2 (index == K) is the first
    # extra rank and takes the 0.99 head-of-ramp factor
    assert close(out[3:], [0.0, 0.0, 0.0])
    assert close(out[:3], [0.4, 0.2, 0.1 * 0.99])


def test_threshold_floor_is_max_1_n_over_4():
    w = [0.5, 0.3, 0.1, 0.07]
    # N=4 -> F = 1; T=10: threshold = 10 * w[1] = 3.0 -> nothing above
    out, cnt = run_post(w, 4, 2, T=10.0, renorm=False)
    assert cnt == 1  # the floor
    assert close(out, np.array([0.5, 0.0, 0.0, 0.0]))


def test_flat_probabilities_keep_all_n():
    w = [0.25, 0.25, 0.25, 0.25]
    out, cnt = run_post(w, 4, 2, T=0.8, renorm=False)
    assert cnt == 4


def test_threshold_strictly_greater_than_t_times_p_ref():
    # rank exactly at the threshold is dropped (step semantics)
    w = [0.3, 0.2, 0.2, 0.1]
    # T = 1.0: p_ref = w[1] = 0.2; only ranks with w > 0.2 kept above the floor
    out, cnt = run_post(w, 4, 2, T=1.0, renorm=False)
    assert cnt == 1
    assert close(out, np.array([0.3, 0.0, 0.0, 0.0]))


def test_pruning_below_native_k():
    w = [0.4, 0.2, 0.1, 0.05, 0.03, 0.02]
    # T = 2: threshold = 2 * w[2] = 0.2 -> only rank 0 (0.4 > 0.2)
    out, cnt = run_post(w, 6, 2, T=2.0, renorm=False)
    assert cnt == 1
    assert close(out[0], 0.4)


# ---------------------------------------------------------------- decay edges


def test_single_extra_rank_gets_midpoint_factor():
    # N = K + 1: t = 0.5 -> fac = 0.99 + 0.5 * (D - 0.99) = (0.99 + D) / 2
    w = [0.6, 0.3, 0.1]
    out, cnt = run_post(w, 3, 2, T=0.0, D=0.5, renorm=False)
    assert cnt == 3
    assert close(out, [0.6, 0.3, 0.1 * 0.745])


def test_decay_end_d030_reaches_last_rank():
    w = [0.4, 0.25, 0.15, 0.12, 0.08]
    N, K, D = 5, 1, 0.30
    out, _ = run_post(w, N, K, T=0.0, D=D, renorm=False)
    t = np.clip((np.arange(N) - K) / (N - K - 1), 0, 1)
    fac = 0.99 + t * (D - 0.99)
    fac[:K] = 1.0
    assert close(out, np.array(w) * fac)
    assert close(out[-1], 0.08 * 0.30)  # last extra rank -> D
    assert close(out[K], 0.25 * 0.99)  # first extra rank -> 0.99


def test_no_decay_full_influence():
    w = [0.5, 0.3, 0.1, 0.07, 0.02, 0.01]
    out, cnt = run_post(w, 6, 2, T=0.0, D=0.5, no_decay=True, renorm=False)
    assert cnt == 6
    assert close(out, np.array(w))


def test_n_equals_k_is_exactly_stock_weights():
    # N == K with no threshold: no decay ramp, no cut — weights pass through
    w = [0.5, 0.3, 0.2]
    out, cnt = run_post(w, 3, 3, T=0.0, renorm=True)
    assert cnt == 3
    assert close(out, np.array(w) / sum(w))


# ---------------------------------------------------------------- misc


def test_per_token_independence():
    w = np.array(
        [
            [0.4, 0.2, 0.1, 0.05, 0.03, 0.02],  # power law -> 3 kept at T=0.8
            [0.25, 0.25, 0.25, 0.25, 0.0, 0.0],  # flat head -> 4 kept
        ],
        dtype=np.float32,
    )
    out_mx, cnt_mx = expansion_post_pass(
        mx.array(w), 6, 2, 0.8, 0.5, False, False
    )
    mx.eval(out_mx, cnt_mx)
    cnt = np.array(cnt_mx)
    assert list(cnt) == [3, 4]
    out = np.array(out_mx)
    # ranks >= K (=2) get the decay ramp 0.99, 0.99 + t*(0.5-0.99), ...
    fac = 0.99 + np.clip((np.arange(6) - 2) / 3.0, 0, 1) * (0.5 - 0.99)
    fac[:2] = 1.0
    exp0 = w[0] * fac
    exp0[3:] = 0.0  # dropped by the threshold
    exp1 = w[1] * fac
    exp1[4:] = 0.0  # already zero; dropped beyond rank 3
    assert close(out[0], exp0)
    assert close(out[1], exp1)


def test_renorm_off_keeps_raw_decayed_scale():
    w = [0.5, 0.3, 0.1]
    # single extra rank with D=0.2: fac = (0.99 + 0.2) / 2 = 0.595
    out, _ = run_post(w, 3, 2, T=0.0, D=0.2, renorm=False)
    assert close(out, np.array([0.5, 0.3, 0.1 * 0.595]))
    assert abs(out.sum() - 1.0) > 1e-3  # not renormalized


def test_renorm_never_zero_guard():
    # pathological all-zero input: the 2^-14 guard avoids division by zero,
    # the weights simply stay zero (no NaN, no crash)
    w = [0.0, 0.0, 0.0, 0.0]
    out, cnt = run_post(w, 4, 2, T=0.0, renorm=True)
    assert not np.isnan(out).any()
    assert close(out, np.zeros(4))


def test_kept_weights_sum_to_one_when_renorm():
    rng = np.random.default_rng(0)
    for trial in range(32):
        N = int(rng.integers(3, 9))
        K = int(rng.integers(1, N))
        # the post-pass requires non-increasing (rank-ordered) weights
        w = np.sort(rng.random(N).astype(np.float32))[::-1]
        T = float(rng.choice([0.0, 0.5, 0.8, 1.0, 2.0, 10.0]))
        D = float(rng.uniform(0.05, 0.95))
        out, cnt = run_post(list(w), N, K, T=T, D=D, renorm=True)
        assert abs(out.sum() - 1.0) < 1e-5, (N, K, T, D)
        # kept set is a contiguous prefix of the ranks
        cnt = int(cnt)
        assert close(out[cnt:], np.zeros(N - cnt))
        assert (out[:cnt] > 0).all()


# ---------------------------------------------------------------- reduce


def test_reduce_keeps_only_experts_above_factor_times_top1():
    w = [0.4, 0.2, 0.1, 0.05]
    out, cnt = reduce_post_pass(mx.array(np.array([w], dtype=np.float32)), 0.3, True)
    mx.eval(out, cnt)
    out = np.array(out)[0]
    # keep while w >= 0.3 * 0.4 = 0.12 -> ranks 0,1 (0.1 < 0.12 is dropped)
    assert int(np.array(cnt)[0]) == 2
    assert close(out, np.array([0.4, 0.2, 0.0, 0.0]) / 0.6)


# ---------------------------------------------------------------- runtime


def test_layer_range_resolution():
    rt = MoERuntime(40, True, MoEExpansionConfig(max_experts=20, threshold=0.8))
    rt.setup(True, 8, 256)
    assert rt.active and rt.n == 20
    assert (rt.layer_start, rt.layer_end) == (0, 39)  # defaults: all layers

    rt = MoERuntime(
        40, True,
        MoEExpansionConfig(max_experts=20, threshold=0.8, layer_start=0.5),
    )
    rt.setup(True, 8, 256)
    assert rt.layer_start == 20 and rt.layer_end == 39

    rt = MoERuntime(
        40, True,
        MoEExpansionConfig(max_experts=20, threshold=0.8, layer_start=25, layer_end=39),
    )
    rt.setup(True, 8, 256)
    assert rt.active_for(25, 8, 256) and not rt.active_for(24, 8, 256)


def test_n_equals_k_auto_uses_stock_path():
    rt = MoERuntime(40, True, MoEExpansionConfig(max_experts=8))
    rt.setup(True, 8, 256)
    assert not rt.active_for(10, 8, 256)  # == stock → untouched path

    rt = MoERuntime(40, True, MoEExpansionConfig(max_experts=8, renorm="always"))
    rt.setup(True, 8, 256)
    assert rt.active_for(10, 8, 256)  # forced renorm changes semantics


def test_experts_add_resolves_on_top_of_native_k():
    rt = MoERuntime(40, True, MoEExpansionConfig(add_experts=12))
    rt.setup(True, 8, 256)
    assert rt.n == 20


def test_validation_fail_fast():
    with pytest.raises(ValueError):  # N out of range
        MoERuntime(40, True, MoEExpansionConfig(max_experts=999)).setup(True, 8, 256)
    with pytest.raises(ValueError):  # non-MoE model
        MoERuntime(40, True, MoEExpansionConfig(max_experts=20)).setup(False, 0, 0)
    with pytest.raises(ValueError):  # threshold out of range
        MoERuntime(40, True, MoEExpansionConfig(max_experts=20, threshold=11.0)).setup(True, 8, 256)
    with pytest.raises(ValueError):  # decay end out of range
        MoERuntime(40, True, MoEExpansionConfig(max_experts=20, decay_end=0.995)).setup(True, 8, 256)
    with pytest.raises(ValueError):  # layer start after end
        MoERuntime(
            40, True,
            MoEExpansionConfig(max_experts=20, layer_start=30, layer_end=10),
        ).setup(True, 8, 256)
    with pytest.raises(ValueError):  # threshold without N
        MoERuntime(40, True, MoEExpansionConfig(threshold=0.8)).setup(True, 8, 256)
    with pytest.raises(ValueError):  # reduce + expansion mutually exclusive
        MoERuntime(
            40, True,
            MoEExpansionConfig(max_experts=20),
            MoEReduceConfig(factor=0.5),
        ).setup(True, 8, 256)


def test_banner_lines_match_llama_cpp_format():
    rt = MoERuntime(
        40, True,
        MoEExpansionConfig(max_experts=20, threshold=0.8, decay_end=0.5, layer_start=25, layer_end=39),
    )
    rt.setup(True, 8, 256)
    lines = rt.banner_lines()
    assert any(
        l.startswith("moe: routed experts per token: 20 (model default 8; ranks 9..20")
        and "0.99..0.5" in l
        for l in lines
    ), lines
    assert any(
        "p(rank 10)" in l and "0.8" in l and "experts 5..20" in l for l in lines
    ), lines
    assert any(
        "expansion active on layers 25..39 of 40" in l for l in lines
    ), lines


def test_default_profiles():
    p = default_profile_for(40, 256, 8)  # Qwen3.6-35B-A3B
    assert p["max_experts"] == 20 and p["threshold"] == 0.80
    assert p["layer_start"] == 25 and p["layer_end"] == 39
    g = default_profile_for(48, 128, 8)  # generic MoE fallback
    assert g["max_experts"] == 20 and "layer_start" not in g
    assert default_profile_for(12, 4, 4) is None  # too small to expand


def test_stats_window():
    import mlx.core as mx
    import numpy as np

    rt = MoERuntime(40, True, MoEExpansionConfig(max_experts=20, threshold=0.0))
    rt.setup(True, 8, 256)
    rt.stats_every = 4
    for _ in range(4):
        rt.record(25, mx.array([12.0, 20.0]))
        rt.note_tokens(2)
    line = rt.stats_line()
    assert line is not None and "L25:16.0" in line and "mean 16.0" in line
    assert rt.stats_line() is None  # window reset
