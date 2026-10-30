# SPDX-License-Identifier: Apache-2.0
"""``agrimlxo mixed-quantize``: build a per-expert mixed-quantization model.

MLX counterpart of AgrillaMoE's ``expert-rank/make_mixed_gguf.py``: given a
uniform MLX checkpoint (any quant or bf16) and the per-layer expert ranking
produced by the ``--moe-expert-log`` / ``moe_predict.py`` workflow, split
each MoE layer's routed experts into quantization groups — the top-M ranked
experts ("hot") at high bits, the cold tail at low bits — and write a new
checkpoint that :class:`~agrimlxo.models.mixed_experts.MixedSwitchGLU`
loads natively.

The routed branch is the only thing touched: attention, shared experts,
router gate (kept at its 8-bit protection) and all other layers keep the
source checkpoint's quantization unchanged.
"""

import json
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import mlx.core as mx

_PROJECTIONS = ("gate_proj", "up_proj", "down_proj")


def load_order_file(path: str) -> Dict[int, List[int]]:
    """expert-order.txt format: one line per layer, comma-separated expert
    ids, ranked most-important first (missing lines fall back to identity)."""
    orders: Dict[int, List[int]] = {}
    with open(path) as f:
        for layer, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            orders[layer] = [int(x) for x in line.split(",") if x != ""]
    return orders


def parse_layers(spec: str, n_layer: int) -> List[int]:
    """'25-39' / '0--1' / '3,7,11' -> layer list (inclusive, 0-based)."""
    if "," in spec:
        return [int(x) for x in spec.split(",")]
    if "-" in spec:
        parts = spec.split("-")
        if len(parts) == 3 and parts[1] == "":
            # AgrillaMoE style 'S--E': E negative (e.g. '0--1' = last layer)
            start, end = int(parts[0]), -int(parts[2])
        else:
            start, end = int(parts[0]), int(parts[1])
        if end < 0:
            end = n_layer - 1
        return list(range(start, end + 1))
    return [int(spec)]


def _layer_prefix(model) -> str:
    return (
        "language_model.model"
        if hasattr(model, "language_model")
        else "model"
    )


def _moe_layer_indices(loaded) -> List[int]:
    layers = loaded.layers
    return [
        i
        for i, l in enumerate(layers)
        if hasattr(l, "mlp") and hasattr(l.mlp, "switch_mlp")
    ]


def mixed_quantize(
    src_dir: Path,
    out_dir: Path,
    order_file: Optional[str] = None,
    layers_spec: str = "0--1",
    top_m: int = 32,
    hi_bits: int = 8,
    lo_bits: int = 4,
    hi_group_size: int = 64,
    lo_group_size: int = 64,
) -> Dict:
    """Convert ``src_dir`` into a mixed-quant model in ``out_dir``.

    Returns a summary dict (per-layer groups, sizes).
    """
    from .loader import LoadedModel

    loaded = LoadedModel(src_dir)
    model = loaded.model
    prefix = _layer_prefix(model)
    # re-read the source tensors (post-sanitize, matching the module tree)
    weights = {}
    for wf in sorted(src_dir.glob("model*.safetensors")):
        weights.update(mx.load(wf))
    if hasattr(model, "sanitize"):
        weights = model.sanitize(weights)

    orders = load_order_file(order_file) if order_file else {}
    n_layer = len(loaded.layers)
    target_layers = set(parse_layers(layers_spec, n_layer))
    moe_layers = set(_moe_layer_indices(loaded))

    quant = loaded.config.get("quantization")
    mixed_spec: Dict[str, list] = {}
    summary = {}

    for layer in sorted(moe_layers & target_layers):
        mlp = loaded.layers[layer].mlp
        E = mlp.num_experts
        order = orders.get(layer, list(range(E)))
        if len(order) < E:
            order = order + [e for e in range(E) if e not in order]
        order = order[:E]
        hi_ids = order[: min(top_m, E)]
        lo_ids = [e for e in order if e not in set(hi_ids)]

        # keep dequantizing consistent with the checkpoint's uniform quant
        sm = mlp.switch_mlp
        projections = {}
        for name in _PROJECTIONS:
            proj = getattr(sm, name)
            w = getattr(proj, "weight")
            s = getattr(proj, "scales", None)
            b = getattr(proj, "biases", None)
            if s is not None:
                qcfg = {}
                if isinstance(quant, dict):
                    key = f"{prefix}.layers.{layer}.mlp.switch_mlp.{name}"
                    ov = quant.get(key)
                    if isinstance(ov, dict) and "bits" in ov:
                        qcfg = ov
                    else:
                        qcfg = {
                            k: quant[k]
                            for k in ("group_size", "bits", "mode")
                            if k in quant
                        }
                projections[name] = mx.dequantize(
                    w,
                    s,
                    b,
                    group_size=int(qcfg.get("group_size", 64)),
                    bits=int(qcfg.get("bits", 4)),
                    mode=qcfg.get("mode", "affine"),
                )
            else:
                projections[name] = w

        groups = [
            {
                "bits": hi_bits,
                "group_size": hi_group_size,
                "experts": [int(e) for e in hi_ids],
            },
            {"bits": lo_bits, "group_size": lo_group_size},
        ]

        mixed_prefix = f"{prefix}.layers.{layer}.mlp.switch_mlp"
        for gi, ids in ((0, hi_ids), (1, lo_ids)):
            idx = mx.array(ids)
            for name in _PROJECTIONS:
                wq, scales, *biases = mx.quantize(
                    projections[name][idx],
                    group_size=(hi_group_size if gi == 0 else lo_group_size),
                    bits=(hi_bits if gi == 0 else lo_bits),
                )
                base = f"{mixed_prefix}.groups.{gi}.{name}"
                weights[base + "_weight"] = wq
                weights[base + "_scales"] = scales
                if biases:
                    weights[base + "_biases"] = biases[0]
                # drop the uniform projection tensors
                weights.pop(f"{mixed_prefix}.{name}.weight", None)
                weights.pop(f"{mixed_prefix}.{name}.scales", None)
                weights.pop(f"{mixed_prefix}.{name}.biases", None)

        mixed_spec[str(layer)] = groups
        summary[layer] = {
            "hi": len(hi_ids),
            "lo": len(lo_ids),
            "hi_bits": hi_bits,
            "lo_bits": lo_bits,
            "hi_ids": [int(e) for e in hi_ids[:8]],
        }

    # remove quantization overrides that would re-quantize mixed switch_mlps
    new_config = dict(loaded.config)
    if isinstance(new_config.get("quantization"), dict):
        q = dict(new_config["quantization"])
        for layer in mixed_spec:
            for name in _PROJECTIONS:
                q.pop(f"{prefix}.layers.{layer}.mlp.switch_mlp.{name}", None)
            q[f"{prefix}.layers.{layer}.mlp.switch_mlp"] = False
        new_config["quantization"] = q
    elif hi_bits != lo_bits or hi_group_size != 64:
        # source was bf16: mixed layers are quantized, the rest stays bf16
        new_config["quantization"] = {
            "group_size": lo_group_size,
            "bits": lo_bits,
        }
        for layer in mixed_spec:
            new_config["quantization"][f"{prefix}.layers.{layer}.mlp.switch_mlp"] = False
    new_config.pop("quantization_config", None)
    new_config["moe_expert_quant"] = mixed_spec

    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "config.json", "w") as f:
        json.dump(new_config, f, indent=2)

    mx.save_safetensors(str(out_dir / "model.safetensors"), weights)

    # copy tokenizer / generation config side files
    for f in src_dir.iterdir():
        if f.name.startswith(("tokenizer", "generation_config", "special_tokens")):
            shutil.copy2(f, out_dir / f.name)

    summary["out_dir"] = str(out_dir)
    summary["layers"] = sorted(int(k) for k in mixed_spec)
    return summary
