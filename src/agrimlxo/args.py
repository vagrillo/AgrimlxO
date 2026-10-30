# SPDX-License-Identifier: Apache-2.0
"""Command-line interface surface for AgrimlxO.

llama.cpp-server-style flags, AgrillaMoE MoE-expansion flags (including the
ds4 aliases) and the AgrillaMoE default-profile injection behavior:

  the default expansion profile is injected ONLY when the user passes no
  --moe-* / --q35-* flag; --no-moe-expansion (or AGRIMLXO_NO_MOE_EXPANSION=1)
  disables the expansion entirely.
"""

import argparse
import os
import sys
from typing import List, Optional

from .moe_expansion import MoEExpansionConfig, MoEReduceConfig

_MOE_FLAGS = [
    "--moe-experts",
    "--q35-experts",
    "--moe-experts-add",
    "--moe-expert-threshold",
    "--q35-expert-threshold",
    "--moe-expert-decay-end",
    "--moe-no-expert-decay",
    "--q35-no-expert-decay",
    "--moe-expert-renorm",
    "--moe-expert-layer-start",
    "--moe-expert-layer-end",
]


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="agrimlxo",
        description=(
            "AgrimlxO — dedicated MLX inference server for Qwen3.x-35B MoE with"
            " runtime MoE expert expansion (see docs/moe-expansion.md and"
            " https://zenodo.org/records/22255483)"
        ),
    )
    sub = p.add_subparsers(dest="command")

    # ---- serve (default command) ----
    serve = sub.add_parser("serve", help="run the inference server (default)")
    _add_serve_args(serve)

    # ---- download ----
    dl = sub.add_parser(
        "download", help="download a model (HF repo id, e.g. mlx-community/Qwen3.6-35B-A3B-4bit)"
    )
    dl.add_argument("model", help="HuggingFace repo id")
    dl.add_argument("--models-dir", default=None, help="extra local models dir (informational)")

    # ---- list ----
    ls = sub.add_parser("list", help="list locally available models")
    ls.add_argument("--models-dir", default=None, help="extra dir to scan for models")

    # ---- mixed-quantize ----
    mq = sub.add_parser(
        "mixed-quantize",
        help="build a per-expert mixed-quantization model (hot experts at high bits)",
    )
    mq.add_argument("model", help="source model directory (uniform quant or bf16)")
    mq.add_argument("--out", required=True, help="output model directory")
    mq.add_argument("--order", default=None,
                    help="expert-order.txt: one line per layer, comma-separated expert ids"
                         " ranked most-important first (from --moe-expert-log + moe_predict.py;"
                         " default: natural expert order)")
    mq.add_argument("--layers", default="0--1",
                    help="MoE layers to convert: '25-39', '0--1' or '3,7,11' (default: all)")
    mq.add_argument("--top-m", dest="top_m", type=int, default=32,
                    help="number of hot experts per layer at the high bitrate (default: 32)")
    mq.add_argument("--hi-bits", dest="hi_bits", type=int, default=8)
    mq.add_argument("--lo-bits", dest="lo_bits", type=int, default=4)
    mq.add_argument("--hi-group-size", dest="hi_group_size", type=int, default=64)
    mq.add_argument("--lo-group-size", dest="lo_group_size", type=int, default=64)
    return p


def _add_serve_args(p: argparse.ArgumentParser):
    # ---- basic server options (llama.cpp-server style) ----
    p.add_argument("-m", "--model", default=None, help="model: local directory or HF repo id (mlx-community format)")
    p.add_argument("--models-dir", default=None, help="extra directory to scan for local models")
    p.add_argument("--host", default="127.0.0.1", help="IP address to listen on (default: 127.0.0.1)")
    p.add_argument("--port", type=int, default=8080, help="port to listen on (default: 8080)")
    p.add_argument("--api-key", default=None, help="API key for authentication (Bearer)")
    p.add_argument(
        "-c", "--ctx-size", type=int, default=131072,
        help="total context size, split across slots (default: 131072 = 128K)",
    )
    p.add_argument(
        "--parallel", type=int, default=1,
        help="number of server slots; the context is divided by this (default: 1)",
    )
    p.add_argument(
        "-b", "--batch-size", type=int, default=2048, dest="batch_size",
        help="prompt processing chunk size (default: 2048)",
    )
    p.add_argument(
        "-n", "--n-predict", type=int, default=-1, dest="n_predict",
        help="max tokens to predict per request, -1 = unlimited (default: -1)",
    )
    p.add_argument("--cache-type-k", default="q8_0", choices=["auto", "f32", "f16", "q8_0", "q4_0"],
                   help="KV cache data type, keys (default: q8_0 — fast on Metal and halves"
                        " the KV memory; 'auto' = the model's compute dtype like mlx-lm)")
    p.add_argument("--cache-type-v", default="q8_0", choices=["auto", "f32", "f16", "q8_0", "q4_0"],
                   help="KV cache data type, values (default: q8_0)")
    p.add_argument("--chat-template", default="", help="jinja chat template override")
    p.add_argument(
        "--jinja", action="store_true", default=False,
        help="(accepted for llama.cpp compatibility; the tokenizer chat template is always used)",
    )
    p.add_argument("--threads", type=int, default=None, help="(no-op, MLX manages its own dispatch)")
    p.add_argument("--mlock", action="store_true", default=False, help="(no-op: weights are mmap'd by MLX)")
    p.add_argument("--no-mmap", action="store_true", default=False, help="(no-op)")
    p.add_argument("--verbose", "-v", action="store_true", default=False, help="verbose logging")

    # ---- sampling defaults (llama.cpp style) ----
    p.add_argument("--temp", "--temperature", dest="temp", type=float, default=0.8)
    p.add_argument("--top-k", dest="top_k", type=int, default=40)
    p.add_argument("--top-p", dest="top_p", type=float, default=0.95)
    p.add_argument("--min-p", dest="min_p", type=float, default=0.05)
    p.add_argument("--xtc-probability", dest="xtc_probability", type=float, default=0.0)
    p.add_argument("--xtc-threshold", dest="xtc_threshold", type=float, default=0.1)
    p.add_argument("--repeat-penalty", dest="repeat_penalty", type=float, default=1.0)
    p.add_argument("--repeat-last-n", dest="repeat_last_n", type=int, default=64)
    p.add_argument("--presence-penalty", dest="presence_penalty", type=float, default=0.0)
    p.add_argument("--frequency-penalty", dest="frequency_penalty", type=float, default=0.0)
    p.add_argument("--seed", type=int, default=None)

    # ---- reasoning (AgrillaMoE fork options) ----
    p.add_argument(
        "--reasoning-budget", dest="reasoning_budget", type=int, default=0,
        help="tokens allowed for the <think> phase before a wrap-up message is"
             " forced (0 = unlimited; required to be finite at low bit-rates)",
    )
    p.add_argument(
        "--reasoning-format", dest="reasoning_format", default="auto",
        choices=["auto", "none"],
        help="auto: split <think>…</think> into reasoning_content; none: keep it in content",
    )

    # ---- MoE expert expansion (the feature) ----
    p.add_argument("--moe-experts", "--q35-experts", dest="moe_experts", type=int, default=0,
                   metavar="N", help="absolute max routed experts per token (2 <= N <= expert_count; 0 = off)")
    p.add_argument("--moe-experts-add", dest="moe_experts_add", type=int, default=0,
                   metavar="N", help="experts added on top of the model's native top-K")
    p.add_argument("--moe-expert-threshold", "--q35-expert-threshold", dest="moe_expert_threshold",
                   type=float, default=0.0, metavar="T",
                   help="adaptive count: keep experts while p >= T x p(rank N/2); T in (0, 10]")
    p.add_argument("--moe-expert-decay-end", dest="moe_expert_decay_end", type=float, default=0.50,
                   metavar="D", help="influence of the last extra rank; linear 0.99..D; D in (0, 0.99)")
    p.add_argument("--moe-no-expert-decay", "--q35-no-expert-decay", dest="moe_no_expert_decay",
                   action="store_true", default=False, help="extra experts at full influence")
    p.add_argument("--moe-expert-renorm", dest="moe_expert_renorm", default="auto",
                   choices=["auto", "always", "never"],
                   help="kept-weight renormalization after cut+decay (default: auto)")
    p.add_argument("--moe-expert-layer-start", dest="moe_expert_layer_start", type=float, default=0.0,
                   metavar="I", help="first layer the expansion applies to; < 1: fraction of n_layer, >= 1: index")
    p.add_argument("--moe-expert-layer-end", dest="moe_expert_layer_end", type=float, default=-1.0,
                   metavar="I", help="last layer (inclusive); < 0: last, < 1: fraction, >= 1: index")
    p.add_argument("--moe-stats-every", dest="moe_stats_every", type=int, default=0, metavar="N",
                   help="print per-layer experts/token stats every N processed tokens"
                        " (0 = use MOE_EXPERT_STATS_EVERY, default 128; only meaningful with expansion active)")
    p.add_argument("--moe-expert-log", dest="moe_expert_log", default="", metavar="FILE",
                   help="log the kept expert ids per token/layer to FILE (moe_predict.py format)")
    p.add_argument("--no-moe-expansion", "--agrilla-no-moe-expansion", dest="no_moe_expansion",
                   action="store_true", default=False,
                   help="disable the MoE expansion entirely (also AGRIMLXO_NO_MOE_EXPANSION=1)")
    # moe-reduce: the inverse of the expansion (dynamic pruning)
    p.add_argument("--moe-reduce", dest="moe_reduce", type=float, default=0.0, metavar="X",
                   help="keep only experts with weight >= X x weight(top-1) (mutually exclusive with --moe-experts)")
    p.add_argument("--moe-reduce-layers", dest="moe_reduce_layers", default="0--1", metavar="S-E",
                   help="layer range for --moe-reduce, inclusive 0-based; -1 = last layer (default: all)")

    # ---- model selection helper ----
    p.add_argument("--suggest-quant", action="store_true", default=False,
                   help="only suggest a mlx-community quant for the unified memory and exit")

    # ---- optional native kernels ----
    p.add_argument("--fast-moe-prefill", dest="fast_moe_prefill", action="store_true",
                   default=None,
                   help="use the fused Metal weighted-sum kernel (package agrimlxo-kernels)"
                        " for large prefill batches; reorders the fp accumulation vs the"
                        " stock ops (also AGRIMLXO_FAST_MOE=1; default off)")

    # ---- tools ----
    p.add_argument("--tools", dest="tools", action="store_true", default=False,
                   help="enable the local tool loop: the model can call the python"
                        " scripts in --tools-dir (the chat template advertises them;"
                        " a <tool_call> block is executed and the result fed back)")
    p.add_argument("--tools-dir", dest="tools_dir", default="./tools",
                   help="directory of tool scripts (default: ./tools)")

    # ---- speed ----
    p.add_argument("--no-compile", dest="no_compile", action="store_true", default=False,
                   help="disable mx.compile of the pure FFN sub-calls (decode/prefill speed)")
    p.add_argument("--no-warmup", dest="no_warmup", action="store_true", default=False,
                   help="skip the startup warmup generation (first request then pays"
                        " the Metal JIT/autotune costs)")


def user_requested_moe(argv: List[str]) -> bool:
    """True when the user passed any --moe-* / --q35-* / --no-moe-expansion flag."""
    for i, tok in enumerate(argv):
        base = tok.split("=", 1)[0]
        if base == "--no-moe-expansion" or base == "--agrilla-no-moe-expansion":
            return True
        if base in _MOE_FLAGS:
            return True
        if base == "--moe-reduce" or base == "--moe-reduce-layers":
            return True
    return False


def env_disabled_moe() -> bool:
    return os.environ.get("AGRIMLXO_NO_MOE_EXPANSION", "") not in ("", "0")


def fast_moe_prefill_requested(args) -> bool:
    """--fast-moe-prefill flag or AGRIMLXO_FAST_MOE=1 env."""
    if getattr(args, "fast_moe_prefill", None):
        return True
    return os.environ.get("AGRIMLXO_FAST_MOE", "") not in ("", "0")


def _parse_reduce_layers(spec: str):
    try:
        start, end = spec.split("-")
        return int(start), int(end)
    except ValueError:
        raise SystemExit(
            f"invalid --moe-reduce-layers '{spec}': expected 'S-E' (e.g. 25-39 or 0--1)"
        )


def expansion_configs_from_args(args) -> tuple:
    """Build (MoEExpansionConfig | None, MoEReduceConfig | None) from parsed args.

    Applies the AgrillaMoE rules: --no-moe-expansion / env disable the feature;
    the default profile injection is decided by the caller (it needs the model
    geometry), never here.
    """
    if args.no_moe_expansion or env_disabled_moe():
        return None, None
    reduce = None
    if getattr(args, "moe_reduce", 0.0) > 0.0:
        reduce = MoEReduceConfig(
            factor=args.moe_reduce, layers=_parse_reduce_layers(args.moe_reduce_layers)
        )
    expansion = MoEExpansionConfig(
        max_experts=args.moe_experts,
        add_experts=args.moe_experts_add,
        threshold=args.moe_expert_threshold,
        decay_end=args.moe_expert_decay_end,
        no_decay=args.moe_no_expert_decay,
        renorm=args.moe_expert_renorm,
        layer_start=args.moe_expert_layer_start,
        layer_end=args.moe_expert_layer_end,
        stats_every=args.moe_stats_every,
        expert_log=args.moe_expert_log,
    )
    if not expansion.requested:
        expansion = None
    return expansion, reduce


def parse_args(argv: Optional[List[str]] = None):
    argv = list(sys.argv[1:] if argv is None else argv)
    parser = _build_parser()
    # `agrimlxo ...` (no subcommand) == `agrimlxo serve ...`
    first = argv[0] if argv else None
    subcommands = ("serve", "download", "list", "mixed-quantize")
    if first is None or first.startswith("-") or first not in subcommands:
        argv = ["serve"] + argv
    args = parser.parse_args(argv)
    return args
