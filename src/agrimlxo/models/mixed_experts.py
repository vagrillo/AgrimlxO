# SPDX-License-Identifier: Apache-2.0
"""Per-expert mixed-quantization for the routed experts (MixedQuant).

The AgrillaMoE workflow (``expert-rank/``: per-layer expert rankings from
``--moe-expert-log`` + ``moe_predict.py``; ``make_mixed_gguf.py``) builds
GGUFs where the hot experts carry more bits than the cold tail. This module
is the MLX equivalent: the stacked ``switch_mlp`` weights of a layer are
split into quantization groups by expert id — e.g. the top-M ranked experts
at 8-bit, the rest at 4-bit — each group stored as its own stacked
quantized tensor.

On-disk layout per MoE layer (weights are the standard ``mx.quantize``
tuples, stacked over the group's experts):

    ...mlp.switch_mlp.groups.0.gate_proj_weight / _scales / _biases
    ...mlp.switch_mlp.groups.0.up_proj_weight   / _scales / _biases
    ...mlp.switch_mlp.groups.0.down_proj_weight / _scales / _biases
    ...mlp.switch_mlp.groups.1....

annotated in config.json (per layer index; missing layers stay uniform):

    "moe_expert_quant": {
      "25": [{"bits": 8, "group_size": 64, "experts": [0, 41, ...]},
             {"bits": 4, "group_size": 64}]
    }

(the last group without ``experts`` takes every remaining expert; the
router gate keeps its uniform 8-bit protection).

``MixedSwitchGLU`` is a drop-in ``switch_mlp`` for ``routed_forward``: it
accepts ``[B, T, N]`` expert ids exactly like ``SwitchGLU`` — including
expanded budgets — and partitions the slots by group with a single global
argsort on the composite key ``group_id * num_experts + local_id``, so each
group runs one contiguous ``gather_qmm`` with sorted indices. Since the
groups are static per layer, the per-group slot counts are Python ints.
"""

from typing import Dict, List, Optional, Tuple

import mlx.core as mx
import mlx.nn as nn

from .activations import swiglu

_PROJECTIONS = ("gate_proj", "up_proj", "down_proj")


class QuantExpertGroup(nn.Module):
    """One quantization group: stacked quantized weights over its experts."""

    def __init__(self, group_index: int, expert_ids: List[int], bits: int, group_size: int):
        super().__init__()
        self.group_index = group_index
        self.expert_ids = list(expert_ids)
        self.bits = bits
        self.group_size = group_size
        for name in _PROJECTIONS:
            setattr(self, f"{name}_weight", None)
            setattr(self, f"{name}_scales", None)
            setattr(self, f"{name}_biases", None)

    def load_projection(self, name: str, tensors: dict):
        for suffix in ("weight", "scales", "biases"):
            setattr(self, f"{name}_{suffix}", tensors.get(suffix))

    def projection(self, name: str) -> Tuple[mx.array, mx.array, Optional[mx.array]]:
        return (
            getattr(self, f"{name}_weight"),
            getattr(self, f"{name}_scales"),
            getattr(self, f"{name}_biases"),
        )


def resolve_groups(
    num_experts: int, groups: List[dict]
) -> List[Tuple[List[int], int, int]]:
    """Validate and complete a group spec: returns [(expert_ids, bits, gs)].
    The last group without ``experts`` takes the remaining experts."""
    resolved: List[Tuple[List[int], int, int]] = []
    assigned: set = set()
    catchall_index = None
    for gi, g in enumerate(groups):
        bits = int(g["bits"])
        gs = int(g.get("group_size", 64))
        ids = [int(e) for e in g.get("experts", [])]
        if not ids:
            if catchall_index is not None:
                raise ValueError(
                    f"mixed-quant group {gi}: only one catch-all group allowed"
                )
            catchall_index = gi
            resolved.append(([], bits, gs))
            continue
        overlap = assigned.intersection(ids)
        if overlap:
            raise ValueError(
                f"mixed-quant group {gi}: experts {sorted(overlap)} appear in"
                " more than one group"
            )
        for e in ids:
            if not (0 <= e < num_experts):
                raise ValueError(
                    f"mixed-quant group {gi}: expert {e} out of range"
                    f" [0, {num_experts})"
                )
        assigned.update(ids)
        resolved.append((ids, bits, gs))
    missing = [e for e in range(num_experts) if e not in assigned]
    if missing:
        if catchall_index is None:
            raise ValueError(
                f"mixed-quant: experts without a group and no catch-all"
                f" group (missing {len(missing)})"
            )
        ids, bits, gs = resolved[catchall_index]
        resolved[catchall_index] = (missing, bits, gs)
    return resolved


class MixedSwitchGLU(nn.Module):
    """Drop-in ``SwitchGLU`` with per-expert quantization groups."""

    def __init__(
        self,
        input_dims: int,
        hidden_dims: int,
        num_experts: int,
        groups: List[dict],
    ):
        super().__init__()
        self.input_dims = input_dims
        self.hidden_dims = hidden_dims
        self.num_experts = num_experts

        resolved = resolve_groups(num_experts, groups)
        self.groups: List[QuantExpertGroup] = [
            QuantExpertGroup(gi, ids, bits, gs)
            for gi, (ids, bits, gs) in enumerate(resolved)
        ]
        # initialize the quantized buffers with their exact shapes so the
        # parameter tree matches the checkpoint (strict weight loading)
        dims = {
            "gate_proj": (input_dims, hidden_dims),
            "up_proj": (input_dims, hidden_dims),
            "down_proj": (hidden_dims, input_dims),
        }
        for g in self.groups:
            for name, (dims_in, dims_out) in dims.items():
                packed_in = dims_in * g.bits // 32
                groups_n = dims_in // g.group_size
                setattr(g, f"{name}_weight", mx.zeros((len(g.expert_ids), dims_out, packed_in), dtype=mx.uint32))
                setattr(g, f"{name}_scales", mx.zeros((len(g.expert_ids), dims_out, groups_n), dtype=mx.float32))
                setattr(g, f"{name}_biases", mx.zeros((len(g.expert_ids), dims_out, groups_n), dtype=mx.float32))

        # routing tables: expert id -> (group index, local id). Python lists
        # (plain attributes, invisible to the MLX parameter tree); the tiny
        # lookup arrays are rebuilt per call.
        group_of = [0] * num_experts
        local_of = [0] * num_experts
        for g in self.groups:
            for li, e in enumerate(g.expert_ids):
                group_of[e] = g.group_index
                local_of[e] = li
        self._group_list = group_of
        self._local_list = local_of
        # static per-group slot counts after the global sort
        self._group_sizes = [len(g.expert_ids) for g in self.groups]
        self._offsets = []
        acc = 0
        for size in self._group_sizes:
            self._offsets.append(acc)
            acc += size

    # -- loading -------------------------------------------------------------

    def load_group(self, group_index: int, name: str, tensors: dict):
        if name not in _PROJECTIONS:
            raise ValueError(f"unknown projection '{name}'")
        self.groups[group_index].load_projection(name, tensors)

    # -- forward -------------------------------------------------------------

    def _project(self, x_slots, gid, lid, name):
        """Per-group gather over all routed slots, masked to the group's
        members. ``x_slots``: [S, 1, K]; ``gid``/``lid``: [S]; returns
        [S, 1, O] (zero at non-member slots).

        The per-group slot counts are data-dependent, so static per-group
        slicing is impossible in a lazy graph: every group computes the full
        slot set and non-members are masked away (G x the routed compute on
        mixed layers — the trade-off buys the memory win; documented in
        docs/mixed-experts.md)."""
        total = None
        for gi, g in enumerate(self.groups):
            if self._group_sizes[gi] == 0:
                continue
            w, s, b = g.projection(name)
            if w is None:
                raise ValueError(
                    f"mixed-quant layer: group {gi} projection '{name}' is not"
                    " loaded"
                )
            member = gid == gi
            # gather_qmm 1:1 contract: lhs [S, 1, K], rhs_indices [S] -> [S,1,O]
            local = mx.where(member, lid, 0)
            out = mx.gather_qmm(
                x_slots,
                w,
                s,
                b,
                rhs_indices=local,
                transpose=True,
                group_size=g.group_size,
                bits=g.bits,
            )
            out = out * member[:, None, None].astype(out.dtype)
            total = out if total is None else total + out
        return total

    def __call__(self, x: mx.array, indices: mx.array) -> mx.array:
        B, T, D = x.shape
        N = indices.shape[-1]
        S = B * T * N
        # gather_qmm 1:1 contract: lhs [S, 1, K], rhs_indices [S] -> [S,1,O]
        x_slots = mx.broadcast_to(
            mx.expand_dims(x, -2), (B, T, N, D)
        ).reshape(S, 1, D)
        idx = indices.reshape(-1)
        gid = mx.array(self._group_list)[idx]  # [S]
        lid = mx.array(self._local_list)[idx]  # [S]

        gate = self._project(x_slots, gid, lid, "gate_proj")
        up = self._project(x_slots, gid, lid, "up_proj")
        act = swiglu(gate, up)  # [S, 1, H]; only member slots survive
        down = self._project(act, gid, lid, "down_proj")

        return down.reshape(B, T, N, D)
