# SPDX-License-Identifier: Apache-2.0
"""Liquid MoE expansion tests: band rules, perplexity indices, the vectorized
ladder (golden hand-derived cases), the graph/numpy index twin cross-check,
the router-distribution capture round-trip and the toy-model end-to-end
behavior (stock equivalence outside the bands, deterministic expansion).

The golden ladder case: w = [0.5, 0.2, 0.15, 0.1, 0.05], K=2, band T0=0.8 /
T'=0.7 / topE=5. Checks: budget 2 -> tail 0.3 / p 0.2 = 1.5 -> clipped 1.0 >
0.8 pass; budget 3 -> tail 0.15 / p 0.15 = 1.0 > 0.7 pass; budget 4 ->
tail 0.05 / p 0.1 = 0.5 <= 0.7 stop -> kept = 4.
"""

import json

import mlx.core as mx
import numpy as np
import pytest

from agrimlxo.moe_expansion import MoERuntime
from agrimlxo.moe_liquid import (
    DEFAULT_BANDS,
    DEFAULT_INDEX,
    INDEX_NAMES,
    LiquidBand,
    LiquidRules,
    compare_runs,
    default_rules,
    index_series,
    index_series_np,
    layer_stats,
    liquid_ladder,
    liquid_post_pass,
    load_capture,
)
from test_engine_toy import PROMPT, ToyTokenizer, build_toy_model, greedy_tokens


# ---------------------------------------------------------------- rules


def test_default_rules_match_notes():
    rules = default_rules()
    assert rules.index == "tail_rel"
    assert [(b.start, b.end, b.perp_t, b.perp_next, b.top_e) for b in rules.bands] == [
        (2, 13, 0.80, 0.70, 10),
        (14, 27, 0.60, 0.60, 12),
        (28, 39, 0.50, 0.50, 20),
    ]
    assert rules.band_for(2) is rules.bands[0]
    assert rules.band_for(0) is None and rules.band_for(1) is None  # stock
    assert rules.band_for(39) is rules.bands[2]


def test_rules_json_roundtrip(tmp_path):
    rules = default_rules()
    p = tmp_path / "rules.json"
    rules.save(p)
    loaded = LiquidRules.load(p)
    assert loaded.to_dict() == rules.to_dict()

    # alternative key spelling ("layers": [s, e]) and explicit perp_next
    alt = LiquidRules.from_dict({
        "index": "tail",
        "bands": [{"layers": [4, 7], "perp_t": 0.5, "top_e": 12}],
    })
    assert alt.index == "tail"
    assert (alt.bands[0].start, alt.bands[0].end) == (4, 7)
    assert alt.bands[0].perp_next == 0.5  # defaults to perp_t


def test_rules_validation_fail_fast():
    with pytest.raises(ValueError):  # band out of the layer range
        LiquidRules(bands=[LiquidBand(35, 45, 0.5, 0.5, 12)]).validate(40, 8, 256)
    with pytest.raises(ValueError):  # overlapping bands
        LiquidRules(
            bands=[LiquidBand(2, 13, 0.8, 0.7, 10), LiquidBand(13, 27, 0.6, 0.6, 12)]
        ).validate(40, 8, 256)
    with pytest.raises(ValueError):  # topE below the native K
        LiquidRules(bands=[LiquidBand(0, 3, 0.8, 0.7, 4)]).validate(8, 8, 32)
    with pytest.raises(ValueError):  # topE above the expert count
        LiquidRules(bands=[LiquidBand(0, 3, 0.8, 0.7, 300)]).validate(8, 8, 256)
    with pytest.raises(ValueError):  # threshold outside [0, 1]
        LiquidRules(bands=[LiquidBand(0, 3, 1.5, 0.7, 12)]).validate(8, 8, 256)
    with pytest.raises(ValueError):  # unknown index
        LiquidRules.from_dict({"index": "nope", "bands": []})
    # a valid minimal ruleset passes
    LiquidRules(bands=[LiquidBand(0, 3, 0.8, 0.7, 12)]).validate(8, 8, 256)


# ---------------------------------------------------------------- indices


def _sorted_dist(head, tail):
    """Descending distribution summing to 1 (hand-built, rank-ordered)."""
    v = np.array(head + tail, dtype=np.float32)
    return v / v.sum()


def test_tail_rel_matches_the_notes_indecision_case():
    # certain: ranks 9..12 far below rank 8 -> index ~ 0
    certain = _sorted_dist([0.30, 0.20, 0.10, 0.08, 0.06, 0.05, 0.04, 0.04],
                           [0.00002] * 8)
    # undecided: ranks 9..12 comparable to rank 8 -> index high
    undecided = _sorted_dist([0.09, 0.08, 0.07, 0.06, 0.05, 0.05, 0.04, 0.04],
                             [0.035, 0.034, 0.033, 0.032, 0.01, 0.01, 0.009, 0.007])
    for d, want in ((certain, 0.0), (undecided, 1.0)):
        w = mx.array(d[None, :8])  # top-8 prefix; full dist has 16 experts
        (idx,) = index_series("tail_rel", w, None, 8, 9)
        mx.eval(idx)
        v = float(np.array(idx)[0])
        if want == 0.0:
            assert v < 0.05, v
        else:
            assert v > 0.9, v


def test_index_series_bounds_and_check_count():
    rng = np.random.default_rng(1)
    s = rng.random((5, 3, 32)).astype(np.float32)
    s /= s.sum(-1, keepdims=True)
    for name in INDEX_NAMES:
        idx = index_series_np(name, s, 8, 16)
        assert idx.shape == (5, 3, 8)  # checks at budgets 8..15
        assert np.isfinite(idx).all()
        assert (idx >= -1e-6).all() and (idx <= 1 + 1e-6).all()


def test_graph_and_numpy_index_twins_agree():
    rng = np.random.default_rng(2)
    s = rng.random((6, 4, 24)).astype(np.float32)
    s /= s.sum(-1, keepdims=True)
    for name in INDEX_NAMES:
        npv = index_series_np(name, s, 4, 12)
        w = mx.sort(mx.array(s), axis=-1)[..., ::-1][..., :12]
        lst = index_series(name, w, mx.array(s), 4, 12)
        mxv = np.stack([np.array(x) for x in lst], axis=-1)
        assert np.allclose(npv, mxv, atol=1e-5), name


# ---------------------------------------------------------------- ladder


def test_ladder_golden_case():
    w = mx.array(np.array([[0.5, 0.2, 0.15, 0.1, 0.05]], dtype=np.float32))
    idx = index_series("tail_rel", w, None, 2, 5)
    kept = liquid_ladder(idx, 0.8, 0.7, 2)
    mx.eval(kept)
    assert int(np.array(kept)[0]) == 4


def test_ladder_stops_at_first_failure_leading_run_only():
    # budgets 2,3,4 pass thresholds 0.8, 0.7, 0.7; budget 5 fails, then 6
    # would pass again — must NOT be counted (leading run semantics)
    idx = [mx.array([1.0]), mx.array([0.75]), mx.array([0.71]),
           mx.array([0.1]), mx.array([0.9])]
    kept = liquid_ladder(idx, 0.8, 0.7, 2)
    mx.eval(kept)
    assert int(np.array(kept)[0]) == 5  # 2 + 3 passes


def test_ladder_certain_token_stays_at_native_k():
    # every check below its threshold: kept == K, no expansion
    idx = [mx.array([0.1]), mx.array([0.05])]
    kept = liquid_ladder(idx, 0.8, 0.7, 8)
    mx.eval(kept)
    assert int(np.array(kept)[0]) == 8


def test_liquid_post_pass_weights_decay_and_renorm():
    w = mx.array(np.array([[0.5, 0.2, 0.15, 0.1, 0.05]], dtype=np.float32))
    band = LiquidBand(0, 0, 0.8, 0.7, 5)
    out, cnt = liquid_post_pass(
        w, None, band, "tail_rel", 2, decay_end=0.5, no_decay=False,
        renormalize=False,
    )
    mx.eval(out, cnt)
    # kept=4; decay ramp against topE=5: ranks 2,3 extra -> 0.99, 0.745
    assert int(np.array(cnt)[0]) == 4
    exp = np.array([0.5, 0.2, 0.15 * 0.99, 0.1 * 0.745, 0.0], dtype=np.float32)
    assert np.allclose(np.array(out)[0], exp, atol=1e-6)

    # renormalized variant sums to 1 over the kept prefix
    outr, cntr = liquid_post_pass(
        w, None, band, "tail_rel", 2, 0.5, False, True
    )
    mx.eval(outr, cntr)
    arr = np.array(outr)[0]
    assert abs(arr.sum() - 1.0) < 1e-5
    assert np.allclose(arr[4:], 0.0, atol=0)


# ---------------------------------------------------------------- runtime


def test_runtime_liquid_setup_and_active_for():
    rt = MoERuntime(40, True, liquid=default_rules())
    rt.setup(True, 8, 256)
    assert rt.liquid_active
    assert rt.liquid_active_for(2) and rt.liquid_active_for(27)
    assert not rt.liquid_active_for(0) and not rt.liquid_active_for(1)
    assert not rt.liquid_active_for(40)  # out of range
    assert rt.liquid_band_for(20).top_e == 12

    # an inert band (topE == K) never activates the liquid path
    rt2 = MoERuntime(8, True, liquid=LiquidRules(bands=[LiquidBand(0, 7, 0.9, 0.9, 8)]))
    rt2.setup(True, 8, 256)
    assert not rt2.liquid_active
    assert not rt2.liquid_active_for(0)


def test_runtime_liquid_is_exclusive_and_validated():
    from agrimlxo.moe_expansion import MoEExpansionConfig, MoEReduceConfig

    with pytest.raises(ValueError):  # liquid + static expansion
        MoERuntime(
            40, True, MoEExpansionConfig(max_experts=12), liquid=default_rules()
        ).setup(True, 8, 256)
    with pytest.raises(ValueError):  # liquid + reduce
        MoERuntime(
            40, True,
            reduce=MoEReduceConfig(factor=0.5),
            liquid=default_rules(),
        ).setup(True, 8, 256)
    with pytest.raises(ValueError):  # non-MoE model
        MoERuntime(40, True, liquid=default_rules()).setup(False, 0, 0)
    with pytest.raises(ValueError):  # bands outside the model geometry
        MoERuntime(10, True, liquid=default_rules()).setup(True, 8, 256)


def test_runtime_liquid_banner_and_apply_config_guard():
    from agrimlxo.moe_expansion import MoEExpansionConfig

    rt = MoERuntime(40, True, liquid=default_rules())
    rt.setup(True, 8, 256)
    lines = rt.banner_lines()
    assert any("liquid expansion" in l and "tail_rel" in l for l in lines), lines
    assert any("liquid band 2..13" in l and "0.8->0.7" in l and "topE 10" in l
               for l in lines), lines
    assert any("liquid band 28..39" in l and "topE 20" in l for l in lines), lines
    with pytest.raises(ValueError):  # the webui MOE panel cannot reconfigure
        rt.apply_config(MoEExpansionConfig(max_experts=12))


# ---------------------------------------------------------------- toy model


def test_toy_inert_bands_bit_identical_to_stock(tmp_path):
    d = build_toy_model(tmp_path, n_expert=8, top_k=2)
    stock, _ = greedy_tokens(d)
    # band covers layer 3 but topE == K -> inert -> untouched stock path
    rt = MoERuntime(
        4, True, liquid=LiquidRules(bands=[LiquidBand(3, 3, 0.9, 0.9, 2)])
    )
    rt.setup(True, 2, 8)
    same, _ = greedy_tokens(d, expansion=rt)
    assert same == stock


def test_toy_liquid_expands_deterministic_and_differs_from_stock(tmp_path):
    from agrimlxo.loader import LoadedModel

    d = build_toy_model(tmp_path, n_expert=8, top_k=2)
    stock, _ = greedy_tokens(d)
    # T=0: every check passes (tail_rel > 0 on a random router) -> topE kept
    rt = MoERuntime(
        4, True,
        liquid=LiquidRules(bands=[LiquidBand(0, 3, 0.0, 0.0, 6)]),
    )
    rt.setup(True, 2, 8)
    # tensor level: the liquid routing changes the model output vs stock
    lm_stock = LoadedModel(d)
    lm_liq = LoadedModel(d, expansion=rt)
    x = mx.array([PROMPT])
    out_stock = lm_stock.model(x, cache=lm_stock.make_cache())
    out_liq = lm_liq.model(x, cache=lm_liq.make_cache())
    mx.eval(out_stock, out_liq)
    assert not np.allclose(np.array(out_stock), np.array(out_liq))

    # engine level: deterministic generation with the liquid expansion on
    liq, r = greedy_tokens(d, expansion=rt)
    assert r == "length"
    again, _ = greedy_tokens(d, expansion=rt)
    assert again == liq  # deterministic
    # the stats window shows the band budget everywhere
    counts = [
        mx.concatenate([c.reshape(-1) for c in st.counts]) for st in rt._stats.values()
    ]
    all_counts = mx.concatenate(counts)
    mx.eval(all_counts)
    assert float(all_counts.mean()) == 6.0


# ---------------------------------------------------------------- capture


def test_capture_roundtrip(tmp_path):
    from agrimlxo.engine import Engine, GenParams
    from agrimlxo.loader import LoadedModel
    from agrimlxo.moe_liquid import RouterCapture, set_capture_sink

    d = build_toy_model(tmp_path, n_expert=8, top_k=2)
    loaded = LoadedModel(d)
    capture = RouterCapture(4, 8)
    set_capture_sink(capture)
    try:
        engine = Engine(loaded, ToyTokenizer(), ctx_size=128, capture=capture)
        capture.start_run({"model": "toy", "temp": 0.7, "n_prompt": len(PROMPT)})
        steps = list(engine.generate(list(PROMPT), GenParams(temp=0.0, max_tokens=4)))
        toks = [s.token for s in steps[:-1]]
        capture.run["tokens"] = toks
        capture.run["prompt_tokens"] = list(PROMPT)
        capture.run["text"] = "x" * len(toks)
        path = tmp_path / "cap" / "run000.npz"
        meta = capture.save_run(path)
        assert capture.run is None  # run closed
        assert meta["n_rows"] == len(PROMPT) + len(toks)
    finally:
        set_capture_sink(None)

    run = load_capture(path)
    assert run.scores.shape[1:] == (4, 8)  # [T, L, E]
    assert run.scores.shape[0] == meta["n_rows"]
    assert run.n_prompt == len(PROMPT)
    assert run.tokens == toks
    sums = run.scores.sum(axis=-1)
    assert np.allclose(sums, 1.0, atol=1e-2)  # post-softmax distributions
    assert run.decode_scores().shape[0] == len(toks)


def test_capture_ignores_traffic_outside_a_run(tmp_path):
    from agrimlxo.loader import LoadedModel
    from agrimlxo.moe_liquid import RouterCapture, set_capture_sink

    d = build_toy_model(tmp_path, n_expert=8, top_k=2)
    loaded = LoadedModel(d)
    capture = RouterCapture(4, 8)
    set_capture_sink(capture)
    try:
        caches = loaded.make_cache()
        out = loaded.model(mx.array([PROMPT]), cache=caches)  # no run open
        mx.eval(out)
        capture.drain_all()
        assert capture.pending() == 0
        assert all(not rows for rows in capture.rows)
    finally:
        set_capture_sink(None)


# ---------------------------------------------------------------- analysis


def _fake_run(tmp_path, name, seed, layers=4, experts=8):
    from agrimlxo.moe_liquid import CaptureRun

    rng = np.random.default_rng(seed)
    scores = rng.random((9, layers, experts)).astype(np.float32)
    scores /= scores.sum(-1, keepdims=True)
    return CaptureRun(
        path=name, scores=scores, n_prompt=5, tokens=[1, 2, 3], prompt_tokens=[1, 2],
        meta={"mode": "stock", "text": "abc", "index": ""},
    )


def test_layer_stats_and_compare():
    runs = [_fake_run(None, "a.npz", 3), _fake_run(None, "b.npz", 4)]
    for name in INDEX_NAMES:
        stats = layer_stats(index_series_np(name, runs[0].decode_scores(), 2, 6))
        assert stats["mean"].shape == (4,)
        assert (stats["p50"] <= stats["p90"] + 1e-9).all()
    base = _fake_run(None, "base.npz", 3)
    other = _fake_run(None, "liq.npz", 5)
    res = compare_runs(base, other, "tail_rel", 2, 6)
    assert res["tokens_base"] == 3 and res["tokens_other"] == 3
    assert 0.0 <= res["token_agreement"] <= 1.0
    assert res["delta_mean"].shape == (4,)


# ---------------------------------------------------------------- CLI args


def test_cli_liquid_flags_and_mutex():
    from agrimlxo.args import liquid_rules_from_args, parse_args

    a = parse_args(["serve", "--moe-liquid"])
    rules = liquid_rules_from_args(a)
    assert rules is not None and rules.index == DEFAULT_INDEX
    assert rules.to_dict()["bands"][0]["top_e"] == 10

    b = parse_args(["serve", "--moe-liquid-index", "entropy"])
    assert liquid_rules_from_args(b).index == "entropy"

    c = parse_args([])  # off by default
    assert liquid_rules_from_args(c) is None

    d = parse_args(["serve", "--moe-liquid", "--moe-experts", "20"])
    with pytest.raises(SystemExit):
        liquid_rules_from_args(d)

    e = parse_args(["serve", "--moe-liquid", "--moe-reduce", "0.5"])
    with pytest.raises(SystemExit):
        liquid_rules_from_args(e)

    with pytest.raises(SystemExit):  # unknown index
        liquid_rules_from_args(parse_args(["serve", "--moe-liquid-index", "zz"]))


def test_cli_capture_subcommand_parses(tmp_path):
    from agrimlxo.args import parse_args

    rules_file = tmp_path / "rules.json"
    rules_file.write_text(json.dumps({
        "index": "tail",
        "bands": [{"layers": [2, 13], "perp_t": 0.8, "perp_next": 0.7, "top_e": 10}],
    }))
    a = parse_args([
        "capture", "-m", "mlx-community/Qwen3.6-35B-A3B-8bit",
        "--out", str(tmp_path), "--temp", "0.7", "--runs", "3",
        "--moe-liquid-rules", str(rules_file),
    ])
    assert a.command == "capture"
    assert a.temp == 0.7 and a.runs == 3
    from agrimlxo.args import liquid_rules_from_args

    rules = liquid_rules_from_args(a)
    assert rules.index == "tail"
    assert (rules.bands[0].start, rules.bands[0].end) == (2, 13)
    assert rules.bands[0].top_e == 10


def test_moe_analyze_and_compare_subcommands_parse(tmp_path):
    from agrimlxo.args import parse_args

    a = parse_args(["moe-analyze", "a.npz", "b.npz", "--calibrate", "0.9"])
    assert a.command == "moe-analyze" and a.calibrate == 0.9
    assert a.top_k == 8 and a.max_experts == 20
    b = parse_args(["moe-compare", "base.npz", "liq.npz", "--show-text"])
    assert b.command == "moe-compare" and b.show_text
