# SPDX-License-Identifier: Apache-2.0
"""AgrimlxO command-line entry points: serve / download / list."""

import os
import sys
from pathlib import Path
from typing import List, Optional

from ._version import __version__
from .args import (
    expansion_configs_from_args,
    fast_moe_prefill_requested,
    parse_args,
    user_requested_moe,
)
from .moe_expansion import default_profile_for

_BANNER = r"""
    _                    _      _  __  __  _   _ ___
   / \   __ _  ___ _ __ | |    / \|  \/  |/ / | |_ _|
  / _ \ / _` |/ _ \ '_ \| |   / _ \ |\/| | || | || |
 / ___ \ (_| |  __/ | | | |  / ___ \ |  | | || | || |
/_/   \_\__, |\___|_| |_|_| /_/   \_\_| |_| \__\_|
        |___/   dedicated Qwen3.x-35B MoE server
                MLX inference + MoE expansion
"""


def _log(msg: str):
    print(msg, file=sys.stderr, flush=True)


# -- model discovery -----------------------------------------------------------


def _hf_cache_models() -> List[str]:
    """Repo ids already present in the local HF cache (with safetensors)."""
    root = Path(os.environ.get("HF_HOME", str(Path.home() / ".cache" / "huggingface")))
    hub = root / "hub"
    found = []
    if hub.is_dir():
        for d in hub.glob("models--*--*"):
            parts = d.name.split("--")
            if len(parts) < 3:
                continue
            repo = "/".join(parts[1:])
            snaps = d / "snapshots"
            if not snaps.is_dir():
                continue
            for snap in snaps.iterdir():
                if (snap / "config.json").exists() and list(snap.glob("model*.safetensors")):
                    found.append(repo)
                    break
    return sorted(set(found))


def _dir_models(extra_dirs: List[str]) -> List[Path]:
    """Local directories that directly contain a model."""
    found = []
    for d in extra_dirs:
        if not d:
            continue
        p = Path(d).expanduser()
        if p.is_dir() and (p / "config.json").exists() and list(p.glob("model*.safetensors")):
            found.append(p)
    return found


def discover_models(extra_dir: Optional[str] = None) -> List[str]:
    """All locally usable models: explicit dir, $AGRIMLXO_MODELS_DIR, ~/models,
    ./models and the HF cache."""
    dirs = [
        extra_dir,
        os.environ.get("AGRIMLXO_MODELS_DIR"),
        str(Path.home() / "models"),
        "models",
    ]
    entries: List[str] = [str(p) for p in _dir_models([d for d in dirs if d])]
    entries.extend(_hf_cache_models())
    return entries


def resolve_model_arg(model: Optional[str], extra_dir: Optional[str], argv: List[str]):
    """Resolve the -m value to a local path (downloading an HF repo when
    needed). Mirrors AgrillaMoE's interactive selection when no -m is given."""
    from .loader import resolve_model_path, peek_moe_geometry

    if model:
        return resolve_model_path(model)

    candidates = discover_models(extra_dir)
    qwen = [c for c in candidates if "Qwen3" in str(c) and "35B" in str(c)]
    qwen += [c for c in candidates if "qwen" in str(c).lower() and "moe" in str(c).lower()]
    seen = set()
    ordered = [c for c in qwen + candidates if not (c in seen or seen.add(c))]

    print()
    print("AgrimlxO — model selection")
    print("  local models found:")
    for i, c in enumerate(ordered[:20], 1):
        geom = ""
        g = peek_moe_geometry(Path(c)) if _is_path(c) else None
        if g:
            geom = f"  [{g[0]} layers, {g[1]} experts, top-{g[2]}]"
        print(f"   {i:2d}. {c}{geom}")
    if not ordered:
        print("   (none — pass -m <hf-repo-id> to download one, e.g.")
        print("       agrimlxo -m mlx-community/Qwen3.6-35B-A3B-8bit)")
    print()
    if not ordered:
        raise SystemExit(1)
    if len(ordered) == 1 or not sys.stdin.isatty():
        choice = ordered[0]
        print(f"  -> {choice}")
        return resolve_model_path(choice)
    try:
        raw = input("select model [1]: ").strip()
    except EOFError:
        raw = ""
    idx = int(raw) if raw.isdigit() and 1 <= int(raw) <= len(ordered) else 1
    return resolve_model_path(ordered[idx - 1])


def _is_path(candidate: str) -> bool:
    return os.path.sep in candidate or Path(candidate).is_dir()


# -- memory suggestion (AgrillaMoE-style) --------------------------------------


def unified_memory_bytes() -> int:
    if sys.platform == "darwin":
        try:
            import subprocess

            out = subprocess.check_output(["sysctl", "-n", "hw.memsize"])
            return int(out.strip())
        except Exception:
            return 0
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemTotal:"):
                    return int(line.split()[1]) * 1024
    except Exception:
        pass
    return 0


def suggest_quant(repo_prefix: str = "mlx-community/Qwen3.6-35B-A3B") -> Optional[str]:
    """Suggest the largest mlx-community quant that fits ~92% of memory."""
    try:
        from huggingface_hub import HfApi
    except Exception:
        return None
    api = HfApi()
    budget = unified_memory_bytes() * 0.92
    if budget <= 0:
        budget = 64 * 1024**3
    best = None
    try:
        models = api.list_models(search=repo_prefix, limit=40, sort="downloads")
    except Exception:
        return None
    for m in models:
        rid = m.id
        if "GGUF" in rid or "MLX" not in rid and "mlx" not in rid.lower():
            continue
        try:
            size = _repo_safetensors_size(rid)
        except Exception:
            continue
        if size <= budget and (best is None or size > best[1]):
            best = (rid, size)
    return best[0] if best else None


def _repo_safetensors_size(repo_id: str) -> int:
    from huggingface_hub import HfApi

    api = HfApi()
    total = 0
    for f in api.list_repo_tree(repo_id, recursive=False):
        if getattr(f, "path", "").endswith(".safetensors") and hasattr(f, "size"):
            total += int(f.size or 0)
    return total


# -- commands ------------------------------------------------------------------


def cmd_download(model: str):
    from .loader import resolve_model_path

    _log(f"agrimlxo: downloading {model} ...")
    path = resolve_model_path(model)
    _log(f"agrimlxo: downloaded to {path}")
    geom = None
    from .loader import peek_moe_geometry

    g = peek_moe_geometry(path)
    if g:
        _log(
            f"agrimlxo: {g[0]} layers, {g[1]} experts, native top-{g[2]}"
            + ("" if g[1] else " (not a MoE model)")
        )


def cmd_list(extra_dir: Optional[str]):
    found = discover_models(extra_dir)
    if not found:
        print("no local models found (pass --models-dir or use `agrimlxo download`)")
        return
    from .loader import peek_moe_geometry

    for c in found:
        geom = ""
        if _is_path(str(c)):
            g = peek_moe_geometry(Path(c))
        else:
            g = None
            try:
                from .loader import resolve_model_path

                g = peek_moe_geometry(resolve_model_path(c))
            except Exception:
                g = None
        if g:
            geom = f"  [{g[0]} layers, {g[1]} experts, top-{g[2]}]"
        print(f"{c}{geom}")


def cmd_mixed_quantize(args):
    from pathlib import Path as _P

    from .mixed_quantize import mixed_quantize

    src = resolve_model_arg(args.model, None, ["serve", "-m", args.model])
    _log(f"mixed-quantize: {src} -> {args.out}")
    summary = mixed_quantize(
        _P(src),
        _P(args.out).expanduser(),
        order_file=args.order,
        layers_spec=args.layers,
        top_m=args.top_m,
        hi_bits=args.hi_bits,
        lo_bits=args.lo_bits,
        hi_group_size=args.hi_group_size,
        lo_group_size=args.lo_group_size,
    )
    for layer in summary.get("layers", []):
        s = summary[layer]
        _log(
            f"  layer {layer}: {s['hi']} experts @ {s['hi_bits']}b"
            f" (top: {s['hi_ids']}...), {s['lo']} @ {s['lo_bits']}b"
        )
    _log(f"mixed-quantize: written {summary['out_dir']}")


def apply_default_profile(args, model_path: Path, argv: List[str]):
    """AgrillaMoE behavior: inject the signature-matched default expansion
    profile only when the user passed no --moe-* flag and didn't disable it."""
    from .loader import peek_moe_geometry

    if getattr(args, "no_moe_expansion", False) or os.environ.get(
        "AGRIMLXO_NO_MOE_EXPANSION", ""
    ) not in ("", "0"):
        _log("moe: expansion disabled (--no-moe-expansion)")
        return
    if user_requested_moe(argv):
        return
    geom = peek_moe_geometry(model_path)
    if geom is None:
        return
    n_layer, n_expert, top_k, _ = geom
    profile = default_profile_for(n_layer, n_expert, top_k)
    if profile is None:
        return
    args.moe_experts = profile.get("max_experts", 0)
    args.moe_experts_add = 0
    args.moe_expert_threshold = profile.get("threshold", 0.0)
    args.moe_expert_layer_start = profile.get("layer_start", 0.0)
    args.moe_expert_layer_end = profile.get("layer_end", -1.0)
    _log(
        f"moe: default expansion profile injected ({profile.get('name', 'generic')}):"
        f" --moe-experts {args.moe_experts} --moe-expert-threshold"
        f" {args.moe_expert_threshold:g}"
        + (
            f" --moe-expert-layer-start {args.moe_expert_layer_start}"
            f" --moe-expert-layer-end {args.moe_expert_layer_end}"
            if "layer_start" in profile
            else ""
        )
    )


def cmd_serve(args, argv: List[str]):
    from .loader import LoadedModel, resolve_model_path, build_expansion_runtime
    from .tokenizer import Tokenizer
    from .engine import Engine
    from . import kernels

    print(_BANNER)
    _log(f"agrimlxo {__version__}  (MLX server, MoE expert expansion)")

    if fast_moe_prefill_requested(args):
        if kernels.native_available():
            kernels.set_fast_prefill(True)
            _log("kernels: fused MoE weighted-sum enabled for large prefill batches")
        else:
            _log(
                "kernels: native kernels not available"
                + (
                    f" ({kernels.native_error()})"
                    if kernels.native_error()
                    else " (build on macOS with: AGRIMLXO_WITH_KERNELS=1 pip install ./native)"
                )
                + "; keeping the stock MLX ops"
            )

    model_path = resolve_model_arg(args.model, args.models_dir, argv)
    apply_default_profile(args, model_path, argv)

    expansion_cfg, reduce_cfg = expansion_configs_from_args(args)
    expansion = build_expansion_runtime(
        _load_config_safe(model_path), expansion_cfg, reduce_cfg
    )

    _log(f"loading model: {model_path}")
    loaded = LoadedModel(model_path, expansion=expansion)
    tokenizer = Tokenizer(model_path, args.chat_template)

    if expansion is not None and (expansion.active or expansion.reduce_active):
        for line in expansion.banner_lines():
            _log(line)

    import threading as _threading

    last_line = {"n": 0}

    def progress_cb(ev):
        # riga singola aggiornata con \r: fase, barra, %, tok/s, stato
        import time as _time

        phase = ev["phase"]
        done, total, elapsed = ev["done"], ev["total"], max(1e-6, ev["elapsed"])
        tps = done / elapsed
        if phase == "prefill":
            pct = min(100.0, done * 100.0 / max(1, total))
            bar = "#" * int(pct // 5)
            line = f"\rprefill [{bar:<20}] {pct:5.1f}%  {done}/{total} tok  {tps:.0f} tok/s"
        else:
            state = ev.get("state") or ""
            tag = "🧠 thinking" if state == "thinking" else "✍ answering"
            if total:
                pct = min(100.0, done * 100.0 / max(1, total))
                bar = "#" * int(pct // 5)
                line = f"\rdecode  [{bar:<20}] {pct:5.1f}%  {done} tok  {tps:.1f} tok/s  {tag}"
            else:
                line = f"\rdecode  {done} tok  {tps:.1f} tok/s  {tag}"
        sys.stderr.write(line)
        sys.stderr.flush()
        last_line["n"] = len(line)

    engine = Engine(
        loaded,
        tokenizer,
        ctx_size=args.ctx_size,
        prefill_step_size=args.batch_size,
        parallel=args.parallel,
        cache_type=args.cache_type_k,
        usage_dir="./expertusage",
        model_name=model_label,
    )
    import atexit

    model_label = str(args.model or "model")
    engine.progress_cb = progress_cb
    if getattr(args, "kache", False):
        from .kache import Kache

        def _parse_budget(spec: str) -> int:
            spec = spec.strip().upper()
            mult = 1
            for suffix, m in (("GB", 1 << 30), ("MB", 1 << 20), ("KB", 1 << 10)):
                if spec.endswith(suffix):
                    mult = m
                    spec = spec[: -len(suffix)]
                    break
            return int(float(spec) * mult)

        engine.kache = Kache(args.kache_dir, _parse_budget(args.kache_budget))
        engine.kache_restore = getattr(args, "kache_restore", False)
        _log(
            f"kache: persisted snapshots in {args.kache_dir}"
            f" (budget {args.kache_budget})"
        )
    if args.cache_type_k != args.cache_type_v:
        _log(
            "note: --cache-type-k and --cache-type-v differ; using the K type for"
            " both (per-tensor types are not supported yet)"
        )

    from .server import run_server

    atexit.register(engine.usage.flush)

    if not args.no_compile:
        from .speed import compile_ffn

        n = compile_ffn(engine.loaded.model)
        if n:
            _log(f"speed: mx.compile applied to {n} FFN sub-calls")
    if not args.no_warmup:
        w = engine.warmup()
        mem = w.get("memory") or {}
        _log(
            "speed: warmup decode — greedy {g} tok/s | sampled {s} tok/s"
            " (temp=0.8, top_p=0.95, top_k=40, min_p=0.05)".format(
                g=w["decode_tps"], s=w.get("decode_sampled_tps")
            )
        )
        _log(f"speed: warmup prefill — {w['prefill_tps']} tok/s")
        if mem:
            _log(
                "memory: active {active_gb} GB, cache {cache_gb} GB,"
                " peak {peak_gb} GB".format(**mem)
            )

    run_server(args, engine, tokenizer)


def _load_config_safe(model_path: Path) -> dict:
    try:
        import json

        with open(model_path / "config.json") as f:
            return json.load(f)
    except Exception:
        return {}


def main(argv: Optional[List[str]] = None):
    argv = list(sys.argv[1:] if argv is None else argv)
    args = parse_args(argv)

    if args.command == "download":
        cmd_download(args.model)
        return
    if args.command == "list":
        cmd_list(args.models_dir)
        return
    if args.command == "mixed-quantize":
        cmd_mixed_quantize(args)
        return
    if args.command == "serve":
        if args.suggest_quant:
            q = suggest_quant()
            print(q or "no suitable quant found")
            return
        cmd_serve(args, argv)
        return
    raise SystemExit(f"unknown command {args.command}")


if __name__ == "__main__":
    main()
