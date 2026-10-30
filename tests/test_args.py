# SPDX-License-Identifier: Apache-2.0
"""CLI/args tests: flag parsing, aliases, validation, default profile
injection (AgrillaMoE behavior)."""

import pytest

from agrimlxo.args import (
    expansion_configs_from_args,
    parse_args,
    user_requested_moe,
)
from agrimlxo.moe_expansion import default_profile_for
from agrimlxo.cli import apply_default_profile


class NS:
    """args namespace stand-in with the expansion fields."""

    def __init__(self, **kw):
        d = dict(
            no_moe_expansion=False,
            moe_experts=0,
            moe_experts_add=0,
            moe_expert_threshold=0.0,
            moe_expert_decay_end=0.5,
            moe_no_expert_decay=False,
            moe_expert_renorm="auto",
            moe_expert_layer_start=0.0,
            moe_expert_layer_end=-1.0,
            moe_stats_every=0,
            moe_expert_log="",
            moe_reduce=0.0,
            moe_reduce_layers="0--1",
        )
        d.update(kw)
        for k, v in d.items():
            setattr(self, k, v)


def test_flag_parsing_and_aliases():
    args = parse_args([
        "serve", "-m", "mlx-community/Qwen3.6-35B-A3B-8bit",
        "--moe-experts", "20", "--moe-expert-threshold", "0.8",
        "--moe-expert-layer-start", "25", "--moe-expert-layer-end", "39",
        "-c", "131072", "--parallel", "4",
    ])
    assert args.model == "mlx-community/Qwen3.6-35B-A3B-8bit"
    assert args.moe_experts == 20
    assert args.moe_expert_threshold == 0.8
    assert args.moe_expert_layer_start == 25
    assert args.ctx_size == 131072
    assert args.parallel == 4


def test_ds4_aliases_equivalent():
    a = parse_args(["serve", "--moe-experts", "20", "--q35-expert-threshold", "0.8",
                    "--q35-no-expert-decay"])
    b = parse_args(["serve", "--q35-experts", "20", "--moe-expert-threshold", "0.8",
                    "--moe-no-expert-decay"])
    assert a.moe_experts == b.moe_experts == 20
    assert a.moe_expert_threshold == b.moe_expert_threshold == 0.8
    assert a.moe_no_expert_decay and b.moe_no_expert_decay


def test_user_requested_moe_detection():
    assert user_requested_moe(["serve", "--moe-experts", "20"])
    assert user_requested_moe(["serve", "--moe-experts=20"])
    assert user_requested_moe(["serve", "--q35-experts", "16"])
    assert user_requested_moe(["serve", "--no-moe-expansion"])
    assert user_requested_moe(["serve", "--moe-reduce", "0.5"])
    assert not user_requested_moe(["serve", "--temp", "0.8", "--moe-stats-every", "64"])


def test_default_command_is_serve():
    args = parse_args(["--moe-experts", "12", "--host", "0.0.0.0"])
    assert args.command == "serve"
    assert args.moe_experts == 12
    assert args.host == "0.0.0.0"


def test_expansion_configs_off_when_disabled(monkeypatch):
    args = NS(moe_experts=20)
    monkeypatch.setenv("AGRIMLXO_NO_MOE_EXPANSION", "1")
    expansion, reduce = expansion_configs_from_args(args)
    assert expansion is None and reduce is None


def test_expansion_configs_built(monkeypatch):
    monkeypatch.delenv("AGRIMLXO_NO_MOE_EXPANSION", raising=False)
    args = NS(
        moe_experts=20, moe_expert_threshold=0.8,
        moe_expert_layer_start=25, moe_expert_layer_end=39,
    )
    expansion, reduce = expansion_configs_from_args(args)
    assert expansion.max_experts == 20
    assert expansion.threshold == 0.8
    assert reduce is None

    args2 = NS(moe_reduce=0.5, moe_reduce_layers="25-39")
    expansion2, reduce2 = expansion_configs_from_args(args2)
    assert expansion2 is None
    assert reduce2.factor == 0.5 and reduce2.layers == (25, 39)


def test_default_profile_injection(tmp_path, capsys, monkeypatch):
    monkeypatch.delenv("AGRIMLXO_NO_MOE_EXPANSION", raising=False)
    import json

    d = tmp_path / "m"
    d.mkdir()
    with open(d / "config.json", "w") as f:
        json.dump(
            {
                "model_type": "qwen3_5_moe",
                "text_config": {
                    "num_hidden_layers": 40,
                    "num_experts": 256,
                    "num_experts_per_tok": 8,
                    "norm_topk_prob": True,
                },
            },
            f,
        )

    argv = ["serve", "-m", str(d), "--temp", "0"]
    args = NS()
    apply_default_profile(args, d, argv)
    assert args.moe_experts == 20
    assert args.moe_expert_threshold == 0.8
    assert args.moe_expert_layer_start == 25
    assert args.moe_expert_layer_end == 39
    assert "default expansion profile injected" in capsys.readouterr().err

    # user flags prevent injection
    argv2 = ["serve", "-m", str(d), "--moe-experts", "16"]
    args2 = NS()
    apply_default_profile(args2, d, argv2)
    assert args2.moe_experts == 0

    # explicit disable prevents injection
    argv3 = ["serve", "-m", str(d), "--no-moe-expansion"]
    args3 = NS(no_moe_expansion=True)
    apply_default_profile(args3, d, argv3)
    assert args3.moe_experts == 0


def test_profile_signature_and_fallback():
    p = default_profile_for(40, 256, 8)
    assert p and p["max_experts"] == 20 and "layer_start" in p
    g = default_profile_for(48, 128, 8)
    assert g and g["max_experts"] == 20 and "layer_start" not in g
    assert default_profile_for(16, 4, 4) is None
