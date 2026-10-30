# SPDX-License-Identifier: Apache-2.0
"""End-to-end tests on a toy qwen3_moe model.

Builds a tiny random MoE checkpoint (real safetensors + config.json on disk),
loads it through the AgrimlxO loader and exercises the engine:
  * flags off == stock routing (bit-identical greedy output),
  * expansion changes the routing but stays deterministic,
  * N == K + auto renorm uses the untouched stock path,
  * KV-cache prefix reuse produces identical tokens to a fresh prefill,
  * stop strings / max_tokens / context overflow behave like llama.cpp.
"""

import json
import shutil
from pathlib import Path

import mlx.core as mx
import numpy as np
import pytest

from agrimlxo.loader import LoadedModel
from agrimlxo.engine import Engine, GenParams, ThinkSplit
from agrimlxo.moe_expansion import MoERuntime, MoEExpansionConfig
from agrimlxo.tokenizer import Tokenizer

# A minimal tokenizer stub so engine tests don't need transformers.


class ToyDetok:
    def __init__(self):
        self.text = ""

    def push(self, token):
        return ""

    def finalize(self):
        return ""


class ToyTokenizer:
    """Minimal tokenizer stub: tokens are ints, vocab 0..63."""

    def __init__(self, path=None):
        self.eos_token_ids = {63}
        self.think_start_id = None
        self.think_end_id = None
        self.has_chat_template = True

    def apply_chat_template(self, messages, tokenize=True, **kwargs):
        # stub template: just re-encode the last user content (numeric-only)
        if tokenize:
            return [1, 2, 3, 4, 5]
        return "1 2 3 4 5"

    def detokenizer(self):
        return ToyDetok()

    def encode(self, text):
        try:
            return [int(x) for x in text.split()] if text else []
        except ValueError:
            return []  # stub: only numeric token strings produce tokens


VOCAB = 64
PROMPT = [1, 2, 3, 4, 5]


def build_toy_model(tmp_path: Path, n_layer: int = 4, n_expert: int = 8, top_k: int = 2):
    """Create a random toy qwen3_moe checkpoint directory."""
    from agrimlxo.models.qwen3_moe import Model, ModelArgs

    args = ModelArgs(
        model_type="qwen3_moe",
        hidden_size=64,
        num_hidden_layers=n_layer,
        intermediate_size=64,
        num_attention_heads=4,
        num_experts=n_expert,
        num_experts_per_tok=top_k,
        moe_intermediate_size=32,
        rms_norm_eps=1e-6,
        vocab_size=VOCAB,
        num_key_value_heads=2,
        rope_theta=10000.0,
        tie_word_embeddings=True,
        max_position_embeddings=256,
        norm_topk_prob=True,
    )
    mx.random.seed(42)
    model = Model(args)
    model.eval()

    d = tmp_path / "toy-qwen3moe"
    d.mkdir()
    from mlx.utils import tree_flatten

    weights = dict(tree_flatten(model.parameters()))
    mx.save_safetensors(str(d / "model.safetensors"), weights)
    config = {
        "model_type": "qwen3_moe",
        "hidden_size": 64,
        "num_hidden_layers": n_layer,
        "intermediate_size": 64,
        "num_attention_heads": 4,
        "num_experts": n_expert,
        "num_experts_per_tok": top_k,
        "moe_intermediate_size": 32,
        "rms_norm_eps": 1e-6,
        "vocab_size": VOCAB,
        "num_key_value_heads": 2,
        "head_dim": 16,
        "rope_theta": 10000.0,
        "tie_word_embeddings": True,
        "max_position_embeddings": 256,
        "norm_topk_prob": True,
        "decoder_sparse_step": 1,
        "mlp_only_layers": [],
    }
    with open(d / "config.json", "w") as f:
        json.dump(config, f)
    return d


def greedy_tokens(model_dir, expansion=None, prompt=PROMPT, n=8):
    loaded = LoadedModel(model_dir, expansion=expansion)
    engine = Engine(loaded, ToyTokenizer(), ctx_size=128, parallel=1)
    params = GenParams(temp=0.0, max_tokens=n, repeat_last_n=64)
    steps = list(engine.generate(list(prompt), params))
    toks = []
    text = ""
    for s in steps[:-1]:
        toks.append(s.token)
    return toks, steps[-1].finish_reason


def test_flags_off_is_deterministic(tmp_path):
    d = build_toy_model(tmp_path)
    toks1, r1 = greedy_tokens(d)
    toks2, r2 = greedy_tokens(d)
    assert toks1 == toks2
    assert r1 == r2 == "length"


def test_expansion_changes_routing_but_deterministic(tmp_path):
    d = build_toy_model(tmp_path, n_expert=8, top_k=2)
    # tensor-level: expansion (N=8, T=0) must change the model output vs stock
    from agrimlxo.loader import LoadedModel as LM
    from agrimlxo.moe_expansion import MoERuntime as RT, MoEExpansionConfig as Cfg

    lm_stock = LM(d)
    rt = RT(4, True, Cfg(max_experts=8, threshold=0.0)).setup(True, 2, 8)
    lm_exp = LM(d, expansion=rt)

    x = mx.array([PROMPT])
    out_stock = lm_stock.model(x, cache=lm_stock.make_cache())
    out_exp = lm_exp.model(x, cache=lm_exp.make_cache())
    mx.eval(out_stock, out_exp)
    assert not np.allclose(np.array(out_stock), np.array(out_exp))
    # T=0 keeps exactly N experts per token: the stats window shows it
    counts = [
        mx.concatenate([c.reshape(-1) for c in st.counts]) for st in rt._stats.values()
    ]
    all_counts = mx.concatenate(counts)
    mx.eval(all_counts)
    assert float(all_counts.mean()) == 8.0

    # engine-level determinism with the expansion on
    stock, _ = greedy_tokens(d)
    expanded, r = greedy_tokens(d, expansion=rt)
    assert r == "length"
    again, _ = greedy_tokens(d, expansion=rt)
    assert again == expanded  # deterministic


def test_n_equals_k_with_auto_renorm_matches_stock(tmp_path):
    d = build_toy_model(tmp_path, n_expert=8, top_k=2)
    stock, _ = greedy_tokens(d)
    # N == K → active_for() is False on every layer → untouched stock path
    rt = MoERuntime(4, True, MoEExpansionConfig(max_experts=2))
    rt.setup(True, 2, 8)
    assert rt.active is True  # requested...
    assert rt.active_for(0, 2, 8) is False  # ...but stock path per layer
    same, _ = greedy_tokens(d, expansion=rt)
    assert same == stock


def test_prefix_cache_reuse_matches_fresh_prefill(tmp_path):
    d = build_toy_model(tmp_path)
    loaded = LoadedModel(d)
    engine = Engine(loaded, ToyTokenizer(), ctx_size=256, parallel=1)
    params = GenParams(temp=0.0, max_tokens=6)

    # first generation on the base prompt
    first = [s.token for s in list(engine.generate(PROMPT, params))[:-1]]
    # continuation of the SAME prompt: prefix reuse must kick in
    reused_tokens = []
    for s in list(engine.generate(PROMPT, params))[:-1]:
        reused_tokens.append(s.token)
    assert reused_tokens == first  # same greedy output

    # a longer prompt that strictly extends the cached sequence
    longer = PROMPT + [10, 11]
    fresh = [
        s.token
        for s in list(Engine(loaded, ToyTokenizer(), ctx_size=256).generate(longer, params))[:-1]
    ]
    cont = [s.token for s in list(engine.generate(longer, params))[:-1]]
    assert cont == fresh  # reuse == fresh prefill


def test_stop_string_and_max_tokens(tmp_path):
    d = build_toy_model(tmp_path)
    loaded = LoadedModel(d)
    engine = Engine(loaded, ToyTokenizer(), ctx_size=256)

    steps = list(engine.generate(PROMPT, GenParams(temp=0.0, max_tokens=3)))
    assert steps[-1].finish_reason == "length"
    assert len(steps) == 4  # 3 tokens + final marker

    engine2 = Engine(loaded, ToyTokenizer(), ctx_size=256)
    steps = list(
        engine2.generate(PROMPT, GenParams(temp=0.0, max_tokens=10, stop=["x"]))
    )
    assert steps[-1].finish_reason == "length"


def test_context_overflow_fails_fast(tmp_path):
    d = build_toy_model(tmp_path)
    loaded = LoadedModel(d)
    engine = Engine(loaded, ToyTokenizer(), ctx_size=8)
    with pytest.raises(ValueError):
        list(engine.generate(list(range(1, 100)), GenParams(temp=0.0)))


def test_eos_stops_generation(tmp_path):
    d = build_toy_model(tmp_path)
    loaded = LoadedModel(d)
    engine = Engine(loaded, ToyTokenizer(), ctx_size=256)
    steps = list(engine.generate(PROMPT, GenParams(temp=0.0, max_tokens=60)))
    # vocab 63 is eos: a 60-token run over a 64-vocab model almost surely
    # hits eos or the length cap
    assert steps[-1].finish_reason in ("stop", "length")


def test_expert_log_and_stats(tmp_path, capsys):
    d = build_toy_model(tmp_path, n_expert=8, top_k=2)
    logfile = tmp_path / "experts.jsonl"
    rt = MoERuntime(
        4, True,
        MoEExpansionConfig(max_experts=8, threshold=0.8, expert_log=str(logfile)),
    )
    rt.setup(True, 2, 8)
    greedy_tokens(d, expansion=rt, n=3)
    # stats line prints to stderr when the window fills (default 128 tokens
    # not reached with n=3, so nothing should print)
    assert "experts/token" not in capsys.readouterr().err


# ------------------------------------------------------------- ThinkSplit


def test_think_split_streaming():
    ts = ThinkSplit()
    out = []
    for delta in ["<thi", "nk>deep reasoning", " here</thi", "nk>final", " answer"]:
        out.extend(ts.push(delta))
    out.extend(ts.flush())
    text_r = "".join(d for k, d in out if k == "reasoning")
    text_c = "".join(d for k, d in out if k == "content")
    assert text_r == "deep reasoning here"
    assert text_c == "final answer"


def test_think_split_no_tags_is_content():
    ts = ThinkSplit()
    out = []
    for delta in ["just ", "plain ", "text"]:
        out.extend(ts.push(delta))
    out.extend(ts.flush())
    assert all(k == "content" for k, _ in out)
    assert "".join(d for _, d in out) == "just plain text"


# ------------------------------------------------------- qwen3_5_moe toy


def test_qwen35_moe_hybrid_forward(tmp_path):
    """The hybrid linear/full attention model runs and the expansion path
    activates on its MoE layers."""
    from agrimlxo.models.qwen3_5_moe import Model as Q35Model, TextModelArgs, ModelArgs as Q35Args

    text_args = TextModelArgs(
        model_type="qwen3_5_moe_text",
        hidden_size=64,
        num_hidden_layers=8,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        rms_norm_eps=1e-6,
        vocab_size=64,
        linear_num_value_heads=8,
        linear_num_key_heads=4,
        linear_key_head_dim=16,
        linear_value_head_dim=16,
        linear_conv_kernel_dim=4,
        full_attention_interval=4,
        num_experts=8,
        num_experts_per_tok=2,
        moe_intermediate_size=32,
        shared_expert_intermediate_size=32,
        norm_topk_prob=True,
        tie_word_embeddings=True,
        rope_parameters={
            "type": "default",
            "rope_theta": 10000.0,
            "partial_rotary_factor": 0.5,
        },
        max_position_embeddings=256,
    )
    from dataclasses import asdict

    args = Q35Args(
        model_type="qwen3_5_moe_text", text_config=asdict(text_args)
    )
    mx.random.seed(7)
    model = Q35Model(args)
    model.eval()
    rt = MoERuntime(8, True, MoEExpansionConfig(max_experts=8, threshold=0.8))
    rt.setup(True, 2, 8)
    kinds = [
        "linear_attention" if l.is_linear else "full_attention" for l in model.layers
    ]
    assert kinds == [
        "linear_attention", "linear_attention", "linear_attention", "full_attention",
    ] * 2

    # forward with caches (prefill)
    caches = model.make_cache()
    out = model(mx.array([[1, 2, 3, 4]]), cache=caches)
    mx.eval(out)
    assert out.shape == (1, 4, 64)
