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


def apply_usage_mix_in_memory(
    loaded,
    orders: Dict[int, List[int]],
    layers_spec: str = "0--1",
    top_m: int = 64,
    hi_bits: int = 8,
    lo_bits: int = 4,
    hi_group_size: int = 64,
    lo_group_size: int = 64,
) -> Dict[int, List[dict]]:
    """Load-time per-expert mixed quantization driven by the usage ranking.

    The model on disk stays uniform (e.g. 8-bit); the switch_mlps of the
    selected MoE layers are dequantized in memory, re-quantized into hot/cold
    groups per the ranking, and the module is swapped for MixedSwitchGLU.
    No disk round-trip: the uniform checkpoint remains the single source.

    Returns {layer: groups-spec} (the moe_expert_quant annotation).
    """
    from .models.mixed_experts import MixedSwitchGLU

    model = loaded.model
    prefix = _layer_prefix(model)
    quant = loaded.config.get("quantization")
    n_layer = len(loaded.layers)
    target_layers = set(parse_layers(layers_spec, n_layer))
    mixed_spec: Dict[int, List[dict]] = {}
    targs = getattr(model, "text_args", None)
    hidden = getattr(targs, "hidden_size", None)
    moe_hidden = getattr(targs, "moe_intermediate_size", None)

    for layer in sorted(set(_moe_layer_indices(loaded)) & target_layers):
        mlp = loaded.layers[layer].mlp
        sm = mlp.switch_mlp
        if hasattr(sm, "groups"):
            continue  # già mixed
        E = mlp.num_experts
        order = list(orders.get(layer, []))
        if len(order) < E:
            order += [e for e in range(E) if e not in order]
        order = order[:E]
        hi_ids = order[: min(top_m, E)]
        lo_ids = [e for e in range(E) if e not in set(hi_ids)]

        groups_spec = [
            {"bits": hi_bits, "group_size": hi_group_size,
             "experts": [int(e) for e in hi_ids]},
            {"bits": lo_bits, "group_size": lo_group_size},
        ]
        mixed_sm = MixedSwitchGLU(
            hidden or mlp.gate.in_features, moe_hidden or hidden, E, groups_spec
        )

        for gi, ids in ((0, hi_ids), (1, lo_ids)):
            gsize = hi_group_size if gi == 0 else lo_group_size
            bits = hi_bits if gi == 0 else lo_bits
            idx = mx.array(ids)
            for name in _PROJECTIONS:
                proj = getattr(sm, name)
                w = getattr(proj, "weight")
                sc = getattr(proj, "scales", None)
                b = getattr(proj, "biases", None)
                if sc is None:
                    src = w
                else:
                    qcfg = {}
                    if isinstance(quant, dict):
                        key = f"{prefix}.layers.{layer}.mlp.switch_mlp.{name}"
                        ov = quant.get(key)
                        if isinstance(ov, dict) and "bits" in ov:
                            qcfg = ov
                        else:
                            qcfg = {k: quant[k] for k in ("group_size", "bits", "mode") if k in quant}
                    src = mx.dequantize(
                        w, sc, b,
                        group_size=int(qcfg.get("group_size", 64)),
                        bits=int(qcfg.get("bits", 4)),
                        mode=qcfg.get("mode", "affine"),
                    )
                wq, scales, *biases = mx.quantize(src[idx], group_size=gsize, bits=bits)
                mixed_sm.load_group(gi, name, {
                    "weight": wq, "scales": scales,
                    "biases": biases[0] if biases else None,
                })

        mlp.switch_mlp = mixed_sm
        mixed_spec[layer] = groups_spec

    return mixed_spec


def _expert_bytes(mlp, layer_idx: int, prefix: str, quant) -> Tuple[int, int]:
    """(bytes per expert, source bits) of the uniform switch_mlp."""
    sm = mlp.switch_mlp
    E = sm.gate_proj.num_experts
    total = 0
    src_bits_w = 0
    for name in _PROJECTIONS:
        proj = getattr(sm, name)
        w = getattr(proj, "weight")
        sc = getattr(proj, "scales", None)
        total += int(w.nbytes) // E
        if sc is not None:
            total += (int(sc.nbytes) + (int(getattr(proj, "biases").nbytes)
                        if getattr(proj, "biases", None) is not None else 0)) // E
            dims_in = sc.shape[2] * getattr(proj, "group_size", 64)
            src_bits_w += int(round(w.nbytes * 8 / (E * w.shape[1] * w.shape[2])))
        else:
            dims_in = w.shape[2]
            src_bits_w += 16
    return total, max(2, src_bits_w // len(_PROJECTIONS))


def _bytes_at(dims_out: int, dims_in: int, bits: int, group_size: int = 64) -> int:
    """Packed weight + f16 scales/biases bytes of one expert at `bits`."""
    w = dims_out * dims_in * bits // 8
    groups = dims_in // group_size
    s = dims_out * groups * 2 * 2  # scales + biases f16
    return w + s


def solve_tiers(
    layer_infos: List[dict],
    target_bytes: int,
    tiers: List[int] = (8, 6, 4, 2),
    group_size: int = 64,
) -> Dict[int, List[dict]]:
    """Assign 8/6/4/2-bit tiers per rank position (hot first) so the total
    routed-expert memory fits `target_bytes`. Greedy: hottest ranks get the
    highest tier that still fits the running budget.

    layer_infos: [{layer, E, dims_out, dims_in, src_bits} per MoE layer].
    Returns {layer: [groups-spec]}.
    """
    # base: tutto a 2 bit (il minimo)
    base = sum(_bytes_at(i["dims_out"], i["dims_in"], tiers[-1], group_size)
               * i["E"] for i in layer_infos)
    if base > target_bytes:
        raise ValueError(
            f"mem target too low: all experts at {tiers[-1]} bit already take"
            f" {base / 1e9:.1f} GB > {target_bytes / 1e9:.1f} GB"
        )
    remaining = target_bytes - base
    tier_of_rank = [tiers[-1]] * max(i["E"] for i in layer_infos)
    # upgrade hot-first: per rank, prova i tier dal più alto
    max_E = max(i["E"] for i in layer_infos)
    n_layers = len(layer_infos)
    for rank in range(max_E):
        for tier in tiers[:-1]:
            cost = sum(
                _bytes_at(i["dims_out"], i["dims_in"], tier, group_size)
                - _bytes_at(i["dims_out"], i["dims_in"], tiers[-1], group_size)
                for i in layer_infos if rank < i["E"]
            )
            if cost <= remaining:
                tier_of_rank[rank] = tier
                remaining -= cost
                break
        else:
            continue
    # groups per layer: contigui per rank -> 4 gruppi
    out: Dict[int, List[dict]] = {}
    for i in layer_infos:
        E = i["E"]
        spec = []
        assigned = set()
        for tier in tiers:
            ids = [e for e in range(E)
                   if e not in assigned and tier_of_rank[e] == tier]
            if not ids:
                continue
            assigned.update(ids)
            spec.append({"bits": tier, "group_size": group_size, "experts": ids})
        if assigned != set(range(E)):
            missing = [e for e in range(E) if e not in assigned]
            spec.append({"bits": tiers[-1], "group_size": group_size,
                         "experts": missing})
        out[i["layer"]] = spec
    return out


def apply_usage_mix_4tier(
    loaded,
    orders: Dict[int, List[int]],
    target_bytes: int,
    layers_spec: str = "0--1",
    group_size: int = 64,
    tiers: List[int] = (8, 6, 4, 2),
) -> Dict[int, List[dict]]:
    """4-tier (8/6/4/2 bit) in-memory mixing with a memory target. The tier
    bands follow the usage ranking (hot first) and the solver fits the
    budget. Returns {layer: groups-spec} and swaps the modules."""
    from .models.mixed_experts import MixedSwitchGLU

    model = loaded.model
    prefix = _layer_prefix(model)
    n_layer = len(loaded.layers)
    target_layers = set(parse_layers(layers_spec, n_layer))
    moe_layers = sorted(set(_moe_layer_indices(loaded)) & target_layers)

    layer_infos = []
    for layer in moe_layers:
        mlp = loaded.layers[layer].mlp
        sm = mlp.switch_mlp
        _, src_bits = _expert_bytes(mlp, layer, prefix, loaded.config.get("quantization"))
        dims_in = sm.gate_proj.weight.shape[2]
        dims_out = sm.gate_proj.weight.shape[1]
        layer_infos.append({
            "layer": layer, "E": mlp.num_experts,
            "dims_out": dims_out, "dims_in": dims_in, "src_bits": src_bits,
        })

    spec = solve_tiers(layer_infos, target_bytes, tiers, group_size)

    quant = loaded.config.get("quantization")
    targs = getattr(model, "text_args", None)
    hidden = getattr(targs, "hidden_size", None)
    moe_hidden = getattr(targs, "moe_intermediate_size", None)

    for layer in moe_layers:
        mlp = loaded.layers[layer].mlp
        sm = mlp.switch_mlp
        if hasattr(sm, "groups"):
            continue
        groups_spec = spec[layer]
        mixed_sm = MixedSwitchGLU(
            hidden or mlp.gate.in_features, moe_hidden or hidden,
            mlp.num_experts, groups_spec,
        )
        for gi, g in enumerate(mixed_sm.groups):
            ids = g.expert_ids
            gsize, bits = g.group_size, g.bits
            idx = mx.array(ids)
            for name in _PROJECTIONS:
                proj = getattr(sm, name)
                w = getattr(proj, "weight")
                sc = getattr(proj, "scales", None)
                b = getattr(proj, "biases", None)
                if sc is None:
                    src = w
                else:
                    qcfg = {}
                    if isinstance(quant, dict):
                        key = f"{prefix}.layers.{layer}.mlp.switch_mlp.{name}"
                        ov = quant.get(key)
                        if isinstance(ov, dict) and "bits" in ov:
                            qcfg = ov
                        else:
                            qcfg = {k: quant[k] for k in ("group_size", "bits", "mode") if k in quant}
                    src = mx.dequantize(
                        w, sc, b,
                        group_size=int(qcfg.get("group_size", 64)),
                        bits=int(qcfg.get("bits", 4)),
                        mode=qcfg.get("mode", "affine"),
                    )
                wq, scales, *biases = mx.quantize(src[idx], group_size=gsize, bits=bits)
                mixed_sm.load_group(gi, name, {
                    "weight": wq, "scales": scales,
                    "biases": biases[0] if biases else None,
                })
        mlp.switch_mlp = mixed_sm

    return spec
