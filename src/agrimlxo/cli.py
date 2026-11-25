# SPDX-License-Identifier: Apache-2.0
"""AgrimlxO command-line entry points: serve / download / list /
mixed-quantize, plus the liquid protocol tools (capture / moe-analyze /
moe-compare)."""

import atexit
import logging
import os
import sys
import time
from pathlib import Path
from typing import List, Optional

from ._version import __version__
from .args import (
    expansion_configs_from_args,
    fast_moe_prefill_requested,
    liquid_rules_from_args,
    parse_args,
    user_requested_moe,
)
from .moe_expansion import default_profile_for


def _parse_budget(spec: str) -> int:
    spec = spec.strip().upper()
    mult = 1
    for suffix, m in (("GB", 1 << 30), ("MB", 1 << 20), ("KB", 1 << 10)):
        if spec.endswith(suffix):
            mult = m
            spec = spec[: -len(suffix)]
            break
    return int(float(spec) * mult)

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


# -- capture / liquid protocol (moe_liquid) ------------------------------------

# Built-in prompt set of the protocol phase 1: diverse domains/styles, run at
# temperature 0.7 so the router indecision distribution is sampled broadly.
_CAPTURE_PROMPTS = [
    "Explain how mixture-of-experts routing works in large language models.",
    "Write a short story about a lighthouse keeper who finds a message in a bottle.",
    "Summarize the causes and consequences of the 1929 stock market crash.",
    "Write a Python function that merges two sorted lists without duplicates.",
    "What are the main differences between nuclear fission and fusion?",
    "Translate 'the weather is beautiful today' into Italian, French and German.",
    "Draft a polite email asking a colleague to review a document by Friday.",
    "Describe step by step how to bake a simple focaccia bread at home.",
]


def _capture_prompt_ids(tokenizer, text: str) -> List[int]:
    try:
        ids = tokenizer.apply_chat_template(
            [{"role": "user", "content": text}]
        )
        if ids:
            return list(ids)
    except Exception:
        pass
    return tokenizer.encode(text)


def cmd_capture(args):
    from .engine import Engine, GenParams
    from .loader import LoadedModel, build_expansion_runtime
    from .moe_liquid import RouterCapture, set_capture_sink
    from .tokenizer import Tokenizer

    print(_BANNER)
    liquid_requested = (
        getattr(args, "moe_liquid", False)
        or getattr(args, "moe_liquid_rules", "")
        or getattr(args, "moe_liquid_index", "")
    )
    _log(
        "agrimlxo capture: routing "
        + ("liquid" if liquid_requested else "stock (no expansion)")
    )

    model_path = resolve_model_arg(args.model, args.models_dir, [])
    liquid_rules = liquid_rules_from_args(args)
    expansion = build_expansion_runtime(
        _load_config_safe(model_path), None, None, liquid_rules
    )

    _log(f"loading model: {model_path}")
    loaded = LoadedModel(model_path, expansion=expansion)
    tokenizer = Tokenizer(model_path, args.chat_template)
    if expansion is not None and expansion.liquid_active:
        for line in expansion.banner_lines():
            _log(line)

    text_args = getattr(loaded.model, "text_args", None)
    n_layer = getattr(text_args, "num_hidden_layers", 0) or 1
    n_expert = getattr(text_args, "num_experts", 0) or 1
    capture = RouterCapture(n_layer, n_expert)
    set_capture_sink(capture)

    out_dir = Path(args.out).expanduser()
    # il file usage all-time del motore resta fuori dal server: qui va nella
    # cartella di output per non inquinare ./expertusage
    engine = Engine(
        loaded,
        tokenizer,
        ctx_size=args.ctx_size,
        prefill_step_size=args.batch_size,
        usage_dir=str(out_dir / "expertusage"),
        model_name=str(args.model or model_path),
        capture=capture,
    )
    if not args.no_warmup:
        engine.warmup()

    if args.prompts:
        lines = Path(args.prompts).read_text(encoding="utf-8").splitlines()
        prompts = [ln.strip() for ln in lines if ln.strip()]
    else:
        prompts = list(_CAPTURE_PROMPTS)
    if not prompts:
        raise SystemExit("capture: no prompts (empty --prompts file?)")
    n_runs = args.runs if args.runs > 0 else len(prompts)

    _log(
        f"capture: {n_runs} run(s), temp {args.temp:g}, max_tokens"
        f" {args.max_tokens}, seed base {args.seed} -> {out_dir}"
    )
    saved = []
    try:
        for i in range(n_runs):
            prompt_text = prompts[i % len(prompts)]
            seed = args.seed + i
            prompt_ids = _capture_prompt_ids(tokenizer, prompt_text)
            if not prompt_ids:
                raise SystemExit(f"capture: empty prompt ids for run {i}")
            capture.start_run({
                "model": str(args.model or model_path),
                "prompt": prompt_text,
                "temp": args.temp,
                "seed": seed,
                "mode": "liquid" if liquid_rules is not None else "stock",
                "index": liquid_rules.index if liquid_rules else "",
                "n_prompt": len(prompt_ids),
            })
            params = GenParams(
                temp=args.temp, seed=seed, max_tokens=args.max_tokens,
                repeat_last_n=64,
            )
            tokens: List[int] = []
            parts: List[str] = []
            for step in engine.generate(prompt_ids, params):
                if step.token >= 0:
                    tokens.append(step.token)
                    parts.append(step.text)
            if capture.run is not None:
                capture.run.update({
                    "tokens": tokens,
                    "prompt_tokens": prompt_ids,
                    "text": "".join(parts),
                })
            path = out_dir / f"run{i:03d}.npz"
            meta = capture.save_run(path)
            saved.append(str(path))
            _log(
                f"capture: run {i:03d} -> {path.name}"
                f" ({meta['n_rows']} positions x {n_layer} layers,"
                f" {len(tokens)} generated tok)"
                + (f" [{meta['dropped_rows']} rows dropped]" if meta["dropped_rows"] else "")
            )
    finally:
        capture.abort_run()
        set_capture_sink(None)
    _log(f"capture: done ({len(saved)} files in {out_dir})")


def cmd_moe_analyze(args):
    import glob

    import numpy as np

    from .moe_liquid import (
        INDEX_NAMES,
        default_rules,
        index_series_np,
        layer_stats,
        load_capture,
        summarize_runs,
    )

    files = list(args.files)
    if args.dir:
        files += sorted(glob.glob(str(Path(args.dir) / "*.npz")))
    if not files:
        raise SystemExit("moe-analyze: no capture files (pass files or --dir)")
    runs = [load_capture(f) for f in files]
    total_rows = sum(r.decode_scores().shape[0] if not args.include_prefill
                     else r.scores.shape[0] for r in runs)
    _log(f"moe-analyze: {len(runs)} run(s), {total_rows} token rows, top-K"
         f" {args.top_k}, candidate budgets up to {args.max_experts}")

    indices = [s.strip() for s in args.index.split(",") if s.strip()] or list(INDEX_NAMES)
    for name in indices:
        try:
            stats = summarize_runs(
                runs, name, args.top_k, args.max_experts,
                decode_only=not args.include_prefill,
            )[name]
        except ValueError as e:
            raise SystemExit(f"moe-analyze: {e}")
        print(f"\nindex {name} — first check (budget K={args.top_k}), per layer:")
        print("  layer   mean     p50     p90")
        for l in range(stats["mean"].shape[0]):
            print(
                f"  {l:5d}  {stats['mean'][l]:7.4f}  {stats['p50'][l]:7.4f}"
                f"  {stats['p90'][l]:7.4f}"
            )

    # calibration + rules skeleton (the notes' bands)
    rules = default_rules()
    base_index = rules.index
    chunks = []
    for r in runs:
        scores = r.scores if args.include_prefill else r.decode_scores()
        if scores.shape[0]:
            chunks.append(index_series_np(base_index, scores, args.top_k, args.max_experts))
    if chunks and args.calibrate is not None:
        idx = np.concatenate(chunks, axis=0)
        first = idx[..., 0]  # [T, L] — the check that starts the ladder
        print(f"\ncalibration (index {base_index}, p={args.calibrate:g} quantile):")
        for b in rules.bands:
            lo, hi = max(0, b.start), min(first.shape[1] - 1, b.end)
            if lo > hi:
                continue
            thr = float(np.quantile(first[:, lo:hi + 1], args.calibrate))
            b.perp_t = round(thr, 4)
            b.perp_next = round(thr, 4)
            print(f"  band {b.start:2d}..{b.end:2d}: PerpT {thr:.4f}")
    if args.rules_out:
        rules.save(args.rules_out)
        _log(f"moe-analyze: rules written to {args.rules_out}")


def cmd_moe_compare(args):
    from .moe_liquid import compare_runs, load_capture

    base = load_capture(args.base)
    other = load_capture(args.liquid)
    index = args.index or other.meta.get("index") or base.meta.get("index") or "tail_rel"
    res = compare_runs(base, other, index, args.top_k, args.max_experts)

    print(f"moe-compare: index {index}")
    print(f"  base   : {args.base} ({res['tokens_base']} generated tok,"
          f" mode {base.meta.get('mode', '?')})")
    print(f"  liquid : {args.liquid} ({res['tokens_other']} generated tok,"
          f" mode {other.meta.get('mode', '?')})")
    print("\n  layer |  base mean/p90  |  liq mean/p90   |  d mean")
    sb, so = res["base"], res["other"]
    for l in range(sb["mean"].shape[0]):
        print(
            f"  {l:5d} |  {sb['mean'][l]:7.4f} / {sb['p90'][l]:6.4f} |"
            f"  {so['mean'][l]:7.4f} / {so['p90'][l]:6.4f} |"
            f"  {res['delta_mean'][l]:+7.4f}"
        )
    print(
        f"\noutputs: common prefix {res['common_prefix']} tok,"
        f" agreement {res['token_agreement'] * 100:.1f}%"
    )
    if args.show_text:
        print(f"\n--- base text ---\n{base.meta.get('text', '')}")
        print(f"\n--- liquid text ---\n{other.meta.get('text', '')}")


def apply_default_profile(args, model_path: Path, argv: List[str]):
    """Inject the signature-matched expansion profile ONLY when the user
    explicitly passes --moe-expansion (the expansion is OFF by default)."""
    from .loader import peek_moe_geometry

    if getattr(args, "no_moe_expansion", False) or os.environ.get(
        "AGRIMLXO_NO_MOE_EXPANSION", ""
    ) not in ("", "0"):
        _log("moe: expansion disabled (--no-moe-expansion)")
        return
    if not getattr(args, "moe_expansion", False):
        return  # OFF by default: serve --moe-expansion per attivarla
    if not getattr(args, "moe_expansion", False):
        return  # OFF by default: serve --moe-expansion per attivarla
    if user_requested_moe(argv):
        return  # flag espliciti dell'utente hanno la precedenza
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

    # -v: dettaglio startup (per-file/per-layer/per-fase) + logging INFO dei
    # moduli (es. la riga di memoria della passata mixed); senza -v i moduli
    # loggano solo warning e lo startup resta ai _log di fase.
    vlog = None
    if getattr(args, "verbose", False):
        logging.basicConfig(
            level=logging.INFO,
            stream=sys.stderr,
            format="%(levelname)s %(name)s: %(message)s",
        )
        vlog = _log
    t_start = time.perf_counter()

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
    liquid_rules = liquid_rules_from_args(args)
    expansion = build_expansion_runtime(
        _load_config_safe(model_path), expansion_cfg, reduce_cfg, liquid_rules
    )

    _log(f"loading model: {model_path}")
    t_load = time.perf_counter()
    loaded = LoadedModel(
        model_path, expansion=expansion,
        lazy=bool(getattr(args, "mixed_from_usage", False)),
        log=vlog,
    )
    tokenizer = Tokenizer(model_path, args.chat_template)
    load_s = time.perf_counter() - t_load
    if vlog:
        vlog(f"load: model + tokenizer in {load_s:.1f}s")

    if expansion is not None and (
        expansion.active or expansion.reduce_active or expansion.liquid_active
    ):
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

    model_label = str(args.model or "model")

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
    engine.progress_cb = progress_cb
    if engine.usage is not None:
        _log(f"usage: {engine.usage.tokens_all} tokens all-time"
             f" ({engine.usage.persist_path})")
    if getattr(args, "kache", False):
        from .kache import Kache


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

    mixed_s = 0.0
    if getattr(args, "mixed_from_usage", False):
        from .mixed_quantize import ExpertQuantCache, apply_usage_mix_in_memory

        t_mixed = time.perf_counter()
        if engine.usage is not None and engine.usage.tokens_all < 1000:
            _log(
                "warning: mixed-from-usage su un ranking quasi vuoto"
                f" ({engine.usage.tokens_all} tokens): l'ordine caldo/freddo"
                " segue quasi gli ID, non l'uso reale. Chatta col modello per"
                " accumulare usage, o riusa il file di una variante"
                " equivalente (cp expertusage/<variante>.json"
                f" {engine.usage.persist_path}) e riavvia."
            )

        quant_cache = ExpertQuantCache(
            "./expertcache",
            _parse_budget(args.kache_budget) if hasattr(args, "kache_budget")
            else 8 << 30,
        )
        ranking_src = engine.usage.snapshot()["counts"]
        orders = {
            l: sorted(range(len(row)), key=lambda e: (-row[e], e))
            for l, row in enumerate(ranking_src)
        }
        if getattr(args, "mem_target", None):
            from .mixed_quantize import apply_usage_mix_4tier

            def _parse_bytes(spec: str) -> int:
                spec = spec.strip().upper()
                for suffix, m in (("GB", 1 << 30), ("MB", 1 << 20)):
                    if spec.endswith(suffix):
                        return int(float(spec[:-len(suffix)]) * m)
                return int(float(spec))

            spec = apply_usage_mix_4tier(
                loaded, orders, target_bytes=_parse_bytes(args.mem_target),
                layers_spec="0--1", quant_cache=quant_cache,
                progress=vlog,
            )
            first = next(iter(spec.values()))
            detail = "/".join(f"{g['bits']}b×{len(g['experts'])}" for g in first)
            _log(f"mixed-from-usage: bande {detail} esperti/layer"
                 f" (su {len(spec)} layer), target {args.mem_target}")
        else:
            spec = apply_usage_mix_in_memory(
                loaded, orders, top_m=args.top_m,
                hi_bits=args.hi_bits, lo_bits=args.lo_bits,
                quant_cache=quant_cache,
                progress=vlog,
            )
            _log(
                f"mixed-from-usage: {len(spec)} MoE layer quantizzati per uso"
                f" (top-{args.top_m} @ {args.hi_bits}b, coda @ {args.lo_bits}b)"
            )
        mixed_s = time.perf_counter() - t_mixed

    if not args.no_compile:
        from .speed import compile_ffn

        n = compile_ffn(engine.loaded.model)
        if n:
            _log(f"speed: mx.compile applied to {n} FFN sub-calls")
    warmup_s = 0.0
    if not args.no_warmup:
        # sempre visibile: il primo eval del warmup paga la JIT dei kernel
        # compilati ed è il lungo silenzio che sembra un hang
        _log(
            "speed: warmup — the first eval pays the JIT compile of the"
            " compiled FFN sub-calls (can take minutes on a big model)"
        )
        t_warm = time.perf_counter()
        w = engine.warmup(progress=vlog)
        warmup_s = time.perf_counter() - t_warm
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

    parts = [f"load {load_s:.1f}s"]
    if mixed_s:
        parts.append(f"mixed {mixed_s:.1f}s")
    if warmup_s:
        parts.append(f"warmup {warmup_s:.1f}s")
    _log(
        f"startup: ready in {time.perf_counter() - t_start:.1f}s"
        f" ({', '.join(parts)})"
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
    if args.command == "capture":
        cmd_capture(args)
        return
    if args.command == "moe-analyze":
        cmd_moe_analyze(args)
        return
    if args.command == "moe-compare":
        cmd_moe_compare(args)
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
