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
    "--moe-liquid",
    "--moe-liquid-rules",
    "--moe-liquid-index",
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

    # ---- capture (liquid protocol phase 1) ----
    cap = sub.add_parser(
        "capture",
        help="run prompts and capture the full router distributions to .npz"
             " (liquid protocol phase 1; stock routing unless --moe-liquid)",
    )
    _add_capture_args(cap)

    # ---- moe-analyze (liquid protocol phase 2) ----
    an = sub.add_parser(
        "moe-analyze",
        help="per-layer perplexity-index statistics of captured distributions"
             " (liquid protocol phase 2: pick the index, calibrate thresholds)",
    )
    _add_analyze_args(an)

    # ---- moe-compare (liquid protocol phase 3) ----
    cp = sub.add_parser(
        "moe-compare",
        help="compare two captures (stock vs liquid): perplexity per layer"
             " and generated outputs (liquid protocol phase 3)",
    )
    _add_compare_args(cp)

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


def _add_liquid_args(p: argparse.ArgumentParser):
    """Liquid expansion flags (serve + capture)."""
    p.add_argument("--moe-liquid", dest="moe_liquid", action="store_true", default=False,
                   help="enable the liquid expansion: the per-token expert budget"
                        " grows while the router perplexity ladder says the selection"
                        " is undecided (default bands: layers 2-13 T 0.80->0.70 topE 10,"
                        " 14-27 T 0.60 topE 12, 28-39 T 0.50 topE 20;"
                        " see docs/moe-liquid-expansion.md)")
    p.add_argument("--moe-liquid-rules", dest="moe_liquid_rules", default="", metavar="FILE",
                   help="JSON rules file for --moe-liquid (bands + index; implies"
                        " --moe-liquid)")
    p.add_argument("--moe-liquid-index", dest="moe_liquid_index", default="", metavar="NAME",
                   help="perplexity index for --moe-liquid: tail_rel (default), tail,"
                        " entropy, kept_entropy, eff")


def _add_capture_args(p: argparse.ArgumentParser):
    p.add_argument("-m", "--model", default=None,
                   help="model: local directory or HF repo id (mlx-community format)")
    p.add_argument("--models-dir", default=None, help="extra directory to scan for local models")
    p.add_argument("--prompts", default="", metavar="FILE",
                   help="file with one prompt per line (default: the built-in prompt set)")
    p.add_argument("--out", default="./routercapture", metavar="DIR",
                   help="output directory, one .npz per run (default: ./routercapture)")
    p.add_argument("--runs", type=int, default=0, metavar="N",
                   help="number of runs (default: one per prompt)")
    p.add_argument("--temp", type=float, default=0.7,
                   help="sampling temperature (default: 0.7, per the protocol)")
    p.add_argument("--max-tokens", dest="max_tokens", type=int, default=256, metavar="N")
    p.add_argument("--seed", type=int, default=1234,
                   help="base seed; run i samples with seed+i (default: 1234)")
    p.add_argument("-c", "--ctx-size", dest="ctx_size", type=int, default=8192)
    p.add_argument("-b", "--batch-size", dest="batch_size", type=int, default=2048)
    p.add_argument("--chat-template", default="", help="jinja chat template override")
    p.add_argument("--no-warmup", dest="no_warmup", action="store_true", default=False,
                   help="skip the startup warmup generation")
    p.add_argument("--verbose", "-v", action="store_true", default=False)
    _add_liquid_args(p)


def _add_analyze_args(p: argparse.ArgumentParser):
    p.add_argument("files", nargs="*", help="capture .npz files (agrimlxo capture output)")
    p.add_argument("--dir", default="", metavar="DIR",
                   help="analyze every .npz in DIR (alternative to the file list)")
    p.add_argument("--index", default="", metavar="NAMES",
                   help="comma-separated index names to report (default: all of"
                        " tail_rel, tail, entropy, kept_entropy, eff)")
    p.add_argument("--top-k", dest="top_k", type=int, default=8,
                   help="native top-K of the model (default: 8)")
    p.add_argument("--max-experts", dest="max_experts", type=int, default=20,
                   help="largest candidate budget to evaluate (default: 20)")
    p.add_argument("--include-prefill", dest="include_prefill", action="store_true",
                   default=False, help="include the prompt rows (default: generated tokens only)")
    p.add_argument("--calibrate", type=float, default=None, metavar="P",
                   help="suggest band thresholds as the P-quantile (0..1) of the"
                        " default index over each band's layers/tokens")
    p.add_argument("--rules-out", dest="rules_out", default="", metavar="FILE",
                   help="write the default (or calibrated) rules JSON to FILE")


def _add_compare_args(p: argparse.ArgumentParser):
    p.add_argument("base", help="stock capture .npz (phase-1 run)")
    p.add_argument("liquid", help="liquid capture .npz (phase-3 run, same prompt/seed)")
    p.add_argument("--index", default="", metavar="NAME",
                   help="perplexity index for the per-layer table (default: the"
                        " liquid run's index, else tail_rel)")
    p.add_argument("--top-k", dest="top_k", type=int, default=8)
    p.add_argument("--max-experts", dest="max_experts", type=int, default=20)
    p.add_argument("--show-text", dest="show_text", action="store_true", default=False,
                   help="print both generated texts")


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
    p.add_argument("--verbose", "-v", action="store_true", default=False,
                   help="verbose startup logging (phases, per-layer progress)"
                        " + uvicorn info")

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
    p.add_argument(
        "--no-thinking", dest="no_thinking", action="store_true", default=False,
        help="disable the reasoning phase by default (enable_thinking=false in the"
             " chat template: the model answers directly); a request can still"
             " override it with enable_thinking/chat_template_kwargs",
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
    p.add_argument("--moe-expansion", dest="moe_expansion", action="store_true", default=False,
                   help="enable the default expansion profile (Qwen3.5/3.6-35B-A3B:"
                        " experts 12, threshold 0.75, layers 25-39) — OFF by default;"
                        " use --moe-experts/--moe-expert-threshold for custom settings")
    p.add_argument("--no-moe-expansion", "--agrilla-no-moe-expansion", dest="no_moe_expansion",
                   action="store_true", default=False,
                   help="redundant (the expansion is already off by default); kept"
                        " for backward compatibility")
    # moe-reduce: the inverse of the expansion (dynamic pruning)
    p.add_argument("--moe-reduce", dest="moe_reduce", type=float, default=0.0, metavar="X",
                   help="keep only experts with weight >= X x weight(top-1) (mutually exclusive with --moe-experts)")
    p.add_argument("--moe-reduce-layers", dest="moe_reduce_layers", default="0--1", metavar="S-E",
                   help="layer range for --moe-reduce, inclusive 0-based; -1 = last layer (default: all)")

    # ---- liquid MoE expansion (perplexity-driven dynamic budget) ----
    _add_liquid_args(p)

    # ---- model selection helper ----
    p.add_argument("--suggest-quant", action="store_true", default=False,
                   help="only suggest a mlx-community quant for the unified memory and exit")

    # ---- optional native kernels ----
    p.add_argument("--fast-moe-prefill", dest="fast_moe_prefill", action="store_true",
                   default=None,
                   help="use the fused Metal weighted-sum kernel (package agrimlxo-kernels)"
                        " for large prefill batches; reorders the fp accumulation vs the"
                        " stock ops (also AGRIMLXO_FAST_MOE=1; default off)")

    # ---- load-time mixed quantization from usage ----
    p.add_argument("--mixed-from-usage", dest="mixed_from_usage", action="store_true", default=False,
                   help="at load: quantize the routed experts per usage (the all-time"
                        " ranking in ./expertusage) — top-M hot experts at --hi-bits,"
                        " the tail at --lo-bits; the checkpoint on disk stays uniform")
    p.add_argument("--top-m", dest="top_m", type=int, default=64,
                   help="hot experts per layer for --mixed-from-usage (default 64)")
    p.add_argument("--mem-target", dest="mem_target", default=None,
                   help="4-tier mode (8/6/4/2 bit): memory target for the loaded"
                        " model, e.g. 24GB — the tier bands follow the usage"
                        " ranking (hot first) and the solver fits the budget")
    p.add_argument("--hi-bits", dest="hi_bits", type=int, default=8)
    p.add_argument("--lo-bits", dest="lo_bits", type=int, default=4)
    p.add_argument("--save-mixed", dest="save_mixed", default=None, metavar="DIR",
                   help="save the mixed model to DIR after the load-time mix"
                        " (the next boots load it directly: -m DIR — no re-mix,"
                        " no 28GB peak)")

    # ---- vision / pdf ----
    p.add_argument("--pdf-dpi", dest="pdf_dpi", type=int, default=100,
                   help="resolution used to render PDF pages as images (default: 100);"
                        " the webui settings can override it per browser")

    # ---- kache (persistent conversation KV snapshots) ----
    p.add_argument("--kache", dest="kache", action="store_true", default=False,
                   help="persist conversation KV snapshots in --kache-dir so a"
                        " continuing conversation skips the prefill even after a"
                        " server restart")
    p.add_argument("--kache-dir", dest="kache_dir", default="./kache",
                   help="directory of the persisted snapshots (default: ./kache)")
    p.add_argument("--kache-budget", dest="kache_budget", default="8GB",
                   help="max size of the kache directory, evicting the least"
                        " recently used conversations (default: 8GB; e.g. 4GB, 512MB)")
    p.add_argument("--kache-restore", dest="kache_restore", action="store_true", default=False,
                   help="EXPERIMENTAL: restore conversation snapshots from the kache"
                        " directory into new slots (a native crash was observed in"
                        " some setups; keep off unless verified)")

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
    subcommands = (
        "serve", "download", "list", "mixed-quantize",
        "capture", "moe-analyze", "moe-compare",
    )
    if first is None or first.startswith("-") or first not in subcommands:
        argv = ["serve"] + argv
    args = parser.parse_args(argv)
    return args


def liquid_rules_from_args(args):
    """LiquidRules from the parsed args, or None when the liquid expansion
    is not requested (--moe-liquid / --moe-liquid-rules / --moe-liquid-index).

    Fail fast (SystemExit) on mutual exclusion with the static expansion /
    reduce knobs and on invalid rule files.
    """
    from .moe_liquid import INDEX_NAMES, LiquidRules, default_rules

    requested = (
        getattr(args, "moe_liquid", False)
        or bool(getattr(args, "moe_liquid_rules", ""))
        or bool(getattr(args, "moe_liquid_index", ""))
    )
    if not requested:
        return None
    clashes = []
    if getattr(args, "moe_experts", 0):
        clashes.append("--moe-experts")
    if getattr(args, "moe_experts_add", 0):
        clashes.append("--moe-experts-add")
    if getattr(args, "moe_expert_threshold", 0.0):
        clashes.append("--moe-expert-threshold")
    if getattr(args, "moe_reduce", 0.0):
        clashes.append("--moe-reduce")
    if clashes:
        raise SystemExit(
            f"--moe-liquid is mutually exclusive with {', '.join(clashes)}"
        )
    rules_file = getattr(args, "moe_liquid_rules", "")
    if rules_file:
        try:
            rules = LiquidRules.load(rules_file)
        except ValueError as e:
            raise SystemExit(str(e))
    else:
        rules = default_rules()
    idx = getattr(args, "moe_liquid_index", "")
    if idx:
        if idx not in INDEX_NAMES:
            raise SystemExit(
                f"--moe-liquid-index: unknown index '{idx}'"
                f" (expected one of {', '.join(INDEX_NAMES)})"
            )
        rules.index = idx
    return rules
