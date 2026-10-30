# SPDX-License-Identifier: Apache-2.0
"""Speed layer tests: mx.compile of the FFN sub-calls must be output-neutral,
and warmup must not pollute the expansion stats window."""

import mlx.core as mx
import numpy as np

from agrimlxo.loader import LoadedModel
from agrimlxo.engine import Engine, GenParams
from agrimlxo.moe_expansion import MoERuntime, MoEExpansionConfig
from agrimlxo.speed import compile_ffn

from test_engine_toy import ToyTokenizer, build_toy_model, PROMPT


def _greedy(model_dir, expansion=None, n=6):
    loaded = LoadedModel(model_dir, expansion=expansion)
    engine = Engine(loaded, ToyTokenizer(), ctx_size=128)
    steps = list(engine.generate(PROMPT, GenParams(temp=0.0, max_tokens=n)))
    return [s.token for s in steps[:-1]], steps[-1].finish_reason


def test_compile_ffn_output_neutral(tmp_path):
    d = build_toy_model(tmp_path, n_expert=8, top_k=2)
    stock, r1 = _greedy(d)
    loaded = LoadedModel(d)
    n = compile_ffn(loaded.model)
    assert n > 0
    compiled, r2 = _greedy(d)
    assert r1 == r2 == "length"
    assert compiled == stock, "mx.compile of the FFN must not change outputs"
    # second pass: same instances already compiled
    assert compile_ffn(loaded.model) == 0


def test_warmup_keeps_stats_clean_and_generation_working(tmp_path):
    d = build_toy_model(tmp_path, n_expert=8, top_k=2)
    rt = MoERuntime(4, True, MoEExpansionConfig(max_experts=8, threshold=0.0))
    rt.setup(True, 2, 8)
    loaded = LoadedModel(d, expansion=rt)
    engine = Engine(loaded, ToyTokenizer(), ctx_size=128)

    engine.warmup()
    assert rt._stats == {}  # warmup did not pollute the stats window
    assert rt._tokens_since_stats == 0

    toks, r = [], None
    steps = list(engine.generate(PROMPT, GenParams(temp=0.0, max_tokens=4)))
    toks = [s.token for s in steps[:-1]]
    assert steps[-1].finish_reason == "length"
    # real generation after warmup recorded stats again
    assert rt._stats, "expected stats after post-warmup generation"
    # T=0 keeps exactly N experts per token
    counts = [
        mx.concatenate([c.reshape(-1) for c in st.counts]) for st in rt._stats.values()
    ]
    all_counts = mx.concatenate(counts)
    mx.eval(all_counts)
    assert float(all_counts.mean()) == 8.0
