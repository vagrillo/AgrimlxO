# SPDX-License-Identifier: Apache-2.0
"""Optional native kernels for AgrimlxO (graceful degradation).

The fused MoE routed weighted sum comes from the vendored omlx Metal kernel
(``native/``, package ``agrimlxo-kernels``) and is generalized for the MoE
expert expansion: any routed budget in [2, 64] per token. Everything here
degrades to pure MLX ops when the extension is missing (Linux, CPU-only,
no ``agrimlxo-kernels`` installed) or was built against a mismatched
nanobind ABI (the omlx ``abi_probe`` canary, issue #2139 over there).

Use in ``models/moe_block.py`` for large prefill batches: the routed expert
outputs are computed on sorted rows (no [B, T, N, D] scatter) and the router
scores — already cut/decayed/renormalized by the expansion post-pass — are
folded by the native kernel.
"""

from __future__ import annotations

import logging
from typing import Optional

import mlx.core as mx

logger = logging.getLogger(__name__)

_EXT = None
_EXT_ERROR: Optional[Exception] = None
_PROBED = False

# The fused weighted sum reorders the fp accumulation vs the stock ops, so
# it is opt-in (--fast-moe-prefill / AGRIMLXO_FAST_MOE=1) to keep the
# flags-off bit-identical guarantee airtight.
FAST_PREFILL_ENABLED = False

# Prefill batches shorter than this keep the stock ops (kernel overhead is
# not worth it), mirroring omlx's _MIN_TOKENS.
MIN_FAST_TOKENS = 1024
# Maximum routed budget the native kernel accepts (fixed-N mask keeps the
# graph within this; the expansion validator rejects N > 64 anyway).
MAX_NATIVE_TOPK = 64


def set_fast_prefill(enabled: bool):
    global FAST_PREFILL_ENABLED
    FAST_PREFILL_ENABLED = bool(enabled)


def _probe():
    global _EXT, _EXT_ERROR, _PROBED
    if _PROBED:
        return
    _PROBED = True
    try:
        import agrimlxo_kernels as pkg
    except Exception as exc:
        _EXT_ERROR = exc
        return
    ext, import_error = pkg.load_ext()
    if ext is None:
        _EXT_ERROR = import_error
        return
    probe = getattr(ext, "abi_probe", None)
    if probe is not None:
        try:
            probe(mx.zeros((1,)))
        except TypeError as exc:
            logger.warning(
                "agrimlxo-kernels: native kernels disabled — the extension"
                " was built with a nanobind ABI that does not match this mlx"
                " wheel; rebuild with AGRIMLXO_WITH_KERNELS=1 pip install"
                " ./native"
            )
            _EXT_ERROR = exc
            return
    if not hasattr(ext, "moe_weighted_sum"):
        _EXT_ERROR = RuntimeError("moe_weighted_sum symbol missing")
        return
    _EXT = ext


def native_available() -> bool:
    """True when the fused weighted-sum kernel is importable and ABI-clean."""
    _probe()
    return _EXT is not None and mx.metal.is_available()


def native_error() -> Optional[Exception]:
    _probe()
    return _EXT_ERROR


def sort_routes(x: mx.array, indices: mx.array):
    """mlx-lm's ``_gather_sort`` without the gather (omlx moe_routes).

    Returns ``(x_tok, row_map, idx, inv_order)``: the token rows
    ``x.flatten(0, -3)``, the sorted slot -> token-row map ``order // k``,
    the sorted expert indices, and the inverse permutation, so
    ``x_tok[row_map]`` equals ``_gather_sort(x, indices)[0]`` and the native
    weighted sum can consume the sorted rows directly.
    """
    *_, top_k = indices.shape
    indices = indices.flatten()
    order = mx.argsort(indices)
    # order is a permutation: a scatter inverts it without a second sort
    # (argsort stays cheaper for short route lists).
    if order.size >= 4096:
        inv_order = mx.put_along_axis(
            mx.zeros_like(order),
            order,
            mx.arange(order.size, dtype=order.dtype),
            axis=0,
        )
    else:
        inv_order = mx.argsort(order)
    return x.flatten(0, -3), order // top_k, indices[order], inv_order


def should_use_fast_weighted_sum(
    x: mx.array, n_routed: int, training: bool = False
) -> bool:
    """Whether the native fused weighted sum applies to this MoE call."""
    if not FAST_PREFILL_ENABLED:
        return False
    if training or not native_available():
        return False
    if x.ndim != 3 or x.shape[-2] < MIN_FAST_TOKENS:
        return False
    if x.dtype not in (mx.float16, mx.bfloat16):
        return False
    if not (2 <= n_routed <= MAX_NATIVE_TOPK):
        return False
    return True


def fast_switch_weighted_sum(
    switch_mlp,
    x: mx.array,
    inds: mx.array,
    scores: mx.array,
) -> mx.array:
    """Routed branch computed on sorted rows + fused weighted sum.

    Equivalent to ``switch_mlp(x, inds)`` scatter followed by
    ``(y * scores[..., None]).sum(-2)``, without materializing the
    [B, T, N, D] intermediate. Any exception propagates: the caller falls
    back to the stock ops.
    """
    x_tok, row_map, idx, inv_order = sort_routes(mx.expand_dims(x, (-2, -3)))
    x_sorted = x_tok[row_map]
    y_sorted = switch_mlp(x_sorted, idx, sorted_indices=True)
    return _EXT.moe_weighted_sum(
        mx.contiguous(y_sorted),
        mx.contiguous(inv_order.astype(mx.uint32)),
        mx.contiguous(scores.astype(mx.float32)),
    )


def fallback_weighted_sum(
    y_sorted: mx.array, inv_order: mx.array, scores: mx.array
) -> mx.array:
    """Pure-MLX reference of the fused weighted sum (also the CPU fallback).

    ``y_sorted``: [tokens*n, 1, D] expert outputs sorted by expert id;
    ``inv_order``: sorted slot -> router-order slot; ``scores``:
    [tokens, n] float weights. y[t] = sum_k y_sorted[inv[t*n+k]] * scores[t,k].
    """
    tokens = scores.shape[0]
    n = scores.shape[-1]
    D = y_sorted.shape[-1]
    y = y_sorted.reshape(tokens * n, D)
    y = y[inv_order]  # router order, grouped per token
    y = (y.reshape(tokens, n, D) * scores.astype(y.dtype)[..., None]).sum(axis=-2)
    return y.astype(y_sorted.dtype).reshape(tokens, 1, D)
