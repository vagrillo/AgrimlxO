# SPDX-License-Identifier: Apache-2.0
"""Routed-expert forward shared by the vendored MoE models.

Three code paths:
  * stock — byte-for-byte the upstream mlx-lm computation (argpartition top-K
    + optional renormalization). Taken whenever the expansion does not apply
    to this layer call, which is the "flags off → bit-identical to stock"
    guarantee of the MoE expert expansion feature.
  * expansion — rank-ordered top-N selection (fixed-N mask): the graph always
    carries N slots per token and dropped ranks get zero weight, then the
    post-pass from ``agrimlxo.moe_expansion`` (threshold cut, linear decay on
    the extra ranks, kept-weight renormalization) shapes the final scores.
    Compute scales with N, not with the adaptive kept count, exactly like the
    llama.cpp implementation.
  * liquid — same fixed-N mask at the band's topE, but the per-token kept
    count comes from the perplexity ladder (``agrimlxo.moe_liquid``): the
    router distribution is re-checked after every added expert, per band
    thresholds, up to the band budget.
  * optional native fast path (--fast-moe-prefill, Metal only): with the
    optional ``agrimlxo-kernels`` extension installed, large prefill batches
    compute the expert outputs on rows sorted by expert id and fold the
    (already cut/decayed/renormalized) router scores with the fused
    ``moe_weighted_sum`` kernel — no [B, T, N, D] scatter intermediate. The
    score folding is numerically identical in exact arithmetic but reorders
    the floating-point accumulation, hence it stays opt-in. Any failure
    falls back to the stock ops silently.

The mixed-quantization variant (``models.mixed_experts.MixedSwitchGLU``) is
a drop-in ``switch_mlp``: the same three paths apply unchanged.
"""

import logging

import mlx.core as mx

from .. import kernels, moe_liquid, speed
from ..moe_expansion import (
    expansion_post_pass,
    observe_usage,
    reduce_post_pass,
)
from ..moe_liquid import liquid_post_pass

logger = logging.getLogger(__name__)


def routed_forward(
    x: mx.array,
    gate,
    switch_mlp,
    top_k: int,
    n_expert: int,
    norm_topk_prob: bool,
    runtime=None,
    layer_idx: int = 0,
):
    """Compute the routed-expert branch of one MoE layer.

    Returns (y, kept_count_or_None).
    """
    gates = gate(x)
    gates = mx.softmax(gates, axis=-1, precise=True)

    # router-distribution capture (liquid protocol phase 1): the FULL
    # post-softmax distribution, every layer, every processed position
    _sink = moe_liquid.CAPTURE_SINK
    if _sink is not None:
        _sink.observe_lazy(layer_idx, gates)

    use_expansion = runtime is not None and runtime.active_for(
        layer_idx, top_k, n_expert
    )
    use_liquid = runtime is not None and runtime.liquid_active_for(layer_idx)
    use_reduce = runtime is not None and runtime.reduce_active_for(layer_idx)

    if not use_expansion and not use_liquid and not use_reduce:
        # ---- stock path (upstream mlx-lm, untouched) ----
        k = top_k
        inds = mx.argpartition(gates, kth=-k, axis=-1)[..., -k:]
        inds = mx.stop_gradient(inds)
        scores = mx.take_along_axis(gates, inds, axis=-1)
        if norm_topk_prob:
            scores = scores / scores.sum(axis=-1, keepdims=True)
        cnt = None
        observe_usage(layer_idx, inds, k)
    else:
        # ---- expansion / liquid / reduce path: rank-ordered selection ----
        if use_expansion:
            n = runtime.n
            # rank order by descending router probability (argsort, per the
            # llama.cpp note on tie-breaking)
            order = mx.argsort(-gates, axis=-1)[..., :n]
            order = mx.stop_gradient(order)
            w = mx.take_along_axis(gates, order, axis=-1)  # non-increasing
            w, cnt = expansion_post_pass(
                w,
                n,
                top_k,
                runtime.config.threshold,
                runtime.config.decay_end,
                runtime.config.no_decay,
                runtime.renormalize(),
            )
        elif use_liquid:
            # liquid expansion: the band's topE slots per token, the kept
            # count decided per token by the perplexity ladder
            band = runtime.liquid_band_for(layer_idx)
            n = min(band.top_e, n_expert)
            order = mx.argsort(-gates, axis=-1)[..., :n]
            order = mx.stop_gradient(order)
            w = mx.take_along_axis(gates, order, axis=-1)  # non-increasing
            w, cnt = liquid_post_pass(
                w,
                gates,
                band,
                runtime.liquid.index,
                top_k,
                runtime.config.decay_end,
                runtime.config.no_decay,
                runtime.renormalize(),
            )
        else:
            n = top_k
            order = mx.argsort(-gates, axis=-1)[..., :n]
            order = mx.stop_gradient(order)
            w = mx.take_along_axis(gates, order, axis=-1)
            w, cnt = reduce_post_pass(
                w, runtime.reduce.factor, renormalize=norm_topk_prob
            )
        inds, scores = order, w
        # liquid: the kept set varies per token, so the usage heatmap counts
        # only the native top-K ranks (the expansion records the full N)
        observe_usage(layer_idx, inds, n if use_expansion else top_k)

    if runtime is not None and cnt is not None:
        runtime.record(layer_idx, cnt, inds)

    # ---- optional native fast path (large prefill, Metal, opt-in) ----
    n_routes = inds.shape[-1]
    if kernels.should_use_fast_weighted_sum(x, n_routes):
        try:
            y = kernels.fast_switch_weighted_sum(switch_mlp, x, inds, scores)
            return y, cnt
        except Exception:
            logger.warning(
                "native weighted sum failed; falling back to stock ops",
                exc_info=True,
            )

    y = speed.maybe_compiled(switch_mlp, x, inds)
    y = (y * scores[..., None]).sum(axis=-2)
    return y, cnt
