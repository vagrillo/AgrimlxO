# SPDX-License-Identifier: Apache-2.0
"""Sampling for AgrimlxO.

Adapted from mlx-lm (ml-explore/mlx-lm @ 0d4d3d04, mlx_lm/sample_utils.py)
with llama.cpp-server-style parameter semantics (temp/top-k/top-p/min-p,
XTC, repetition/presence/frequency penalties, logit bias, seed).
"""

import math
from functools import partial
from typing import Callable, Dict, List, Optional

import mlx.core as mx

Sampler = Callable[[mx.array], mx.array]
LogitsProcessor = Callable[[mx.array, List[int]], mx.array]


def greedy_sampler(logprobs: mx.array) -> mx.array:
    return mx.argmax(logprobs, axis=-1)


@partial(mx.compile, inputs=mx.random.state, outputs=mx.random.state)
def apply_top_k(logprobs: mx.array, top_k: int) -> mx.array:
    vocab_size = logprobs.shape[-1]
    if not isinstance(top_k, int) or not (0 < top_k < vocab_size):
        raise ValueError(
            f"`top_k` has to be an integer in the (0, {vocab_size}) interval,"
            f" but is {top_k}."
        )
    mask_idx = mx.argpartition(-logprobs, kth=top_k - 1, axis=-1)[..., top_k:]
    masked_logprobs = mx.put_along_axis(
        logprobs, mask_idx, mx.array(-float("inf"), logprobs.dtype), axis=-1
    )
    return masked_logprobs


@partial(mx.compile, inputs=mx.random.state, outputs=mx.random.state)
def apply_min_p(
    logprobs: mx.array,
    min_p: float,
    min_tokens_to_keep: int = 1,
) -> mx.array:
    if not (0 <= min_p <= 1.0):
        raise ValueError(
            f"`min_p` has to be a float in the [0, 1] interval, but is {min_p}"
        )
    top_logprobs = mx.max(logprobs, axis=-1, keepdims=True)
    scaled_min_p = top_logprobs + math.log(min_p)
    tokens_to_remove = logprobs < scaled_min_p

    if min_tokens_to_keep > 1:
        top_indices = mx.argpartition(logprobs, kth=-min_tokens_to_keep, axis=-1)
        top_indices = top_indices[..., -min_tokens_to_keep:]
        tokens_to_remove = mx.put_along_axis(
            tokens_to_remove,
            top_indices,
            mx.array(False),
            axis=-1,
        )

    return mx.where(tokens_to_remove, -float("inf"), logprobs)


@partial(mx.compile, inputs=mx.random.state, outputs=mx.random.state)
def apply_top_p(logprobs: mx.array, top_p: float) -> mx.array:
    sorted_logprobs = mx.sort(logprobs, axis=-1)
    sorted_probs = mx.exp(sorted_logprobs.astype(mx.float32))
    mass_above = mx.cumsum(sorted_probs, axis=-1, reverse=True, inclusive=False)
    total_mass = mass_above[..., :1] + sorted_probs[..., :1]
    num_dropped = (mass_above >= top_p * total_mass).sum(axis=-1, keepdims=True)
    threshold = mx.take_along_axis(sorted_logprobs, num_dropped, axis=-1)
    return mx.where(logprobs < threshold, -float("inf"), logprobs)


@partial(mx.compile, inputs=mx.random.state, outputs=mx.random.state)
def apply_xtc(
    logits: mx.array,
    xtc_probability: float,
    xtc_threshold: float,
    xtc_special_tokens: List[int],
) -> mx.array:
    if not (0 <= xtc_threshold <= 0.5):
        raise ValueError(
            f"`xtc_threshold` has to be a float in the [0, 0.5] interval, but is"
            f" {xtc_threshold}"
        )
    if not (0 <= xtc_probability <= 1.0):
        raise ValueError(
            f"`xtc_probability` has to be a float in the [0, 1] interval, but is"
            f" {xtc_probability}"
        )

    probs = mx.softmax(logits, -1)
    mask = probs > mx.where(probs > xtc_threshold, probs, mx.inf).min(
        axis=-1, keepdims=True
    )
    if xtc_special_tokens:
        mask[..., xtc_special_tokens] = False

    return mx.where(
        mx.random.uniform(0, 1) > xtc_probability,
        logits,
        mx.where(mask, -mx.inf, logits),
    )


@partial(mx.compile, inputs=mx.random.state, outputs=mx.random.state)
def categorical_sampling(logits: mx.array, temp: float) -> mx.array:
    return mx.random.categorical(logits * (1.0 / temp))


def make_sampler(
    temp: float = 0.0,
    top_p: float = 0.0,
    min_p: float = 0.0,
    min_tokens_to_keep: int = 1,
    top_k: int = 0,
    xtc_probability: float = 0.0,
    xtc_threshold: float = 0.1,
    xtc_special_tokens: Optional[List[int]] = None,
) -> Sampler:
    if temp == 0:
        return greedy_sampler

    xtc_special_tokens = [] if xtc_special_tokens is None else xtc_special_tokens
    sampling_methods = []
    if 0 < top_p < 1.0:
        sampling_methods.append(lambda x: apply_top_p(x, top_p))
    if min_p != 0.0:
        sampling_methods.append(lambda x: apply_min_p(x, min_p, min_tokens_to_keep))
    if xtc_probability > 0.0:
        sampling_methods.append(
            lambda x: apply_xtc(x, xtc_probability, xtc_threshold, xtc_special_tokens)
        )
    if top_k > 0:
        sampling_methods.append(lambda x: apply_top_k(x, top_k))

    def sampler(logprobs: mx.array) -> mx.array:
        for method in sampling_methods:
            logprobs = method(logprobs)
        return categorical_sampling(logprobs, temp)

    return sampler


def make_repetition_penalty(penalty: float, context_size: int = 20):
    if penalty <= 0:
        raise ValueError(f"penalty must be positive, got {penalty}")

    def repetition_penalty_processor(tokens: List[int], logits: mx.array):
        if len(tokens) > 0:
            tokens = tokens[-context_size:]
            selected_logits = logits[:, tokens]
            selected_logits = mx.where(
                selected_logits < 0,
                selected_logits * penalty,
                selected_logits / penalty,
            )
            logits[:, tokens] = selected_logits
        return logits

    return repetition_penalty_processor


def make_presence_penalty(penalty: float, context_size: int = 20):
    def presence_penalty_processor(tokens: List[int], logits: mx.array):
        if len(tokens) > 0:
            tokens = tokens[-context_size:]
            logits[:, tokens] -= penalty
        return logits

    return presence_penalty_processor


def make_frequency_penalty(penalty: float, context_size: int = 20):
    def frequency_penalty_processor(tokens: List[int], logits: mx.array):
        if len(tokens) > 0:
            tokens = tokens[-context_size:]
            logits = logits.at[:, tokens].subtract(penalty)
        return logits

    return frequency_penalty_processor


def make_logits_processors(
    logit_bias: Optional[Dict[int, float]] = None,
    repetition_penalty: Optional[float] = None,
    repetition_context_size: int = 20,
    presence_penalty: Optional[float] = None,
    presence_context_size: int = 20,
    frequency_penalty: Optional[float] = None,
    frequency_context_size: int = 20,
) -> List[LogitsProcessor]:
    logits_processors = []
    if logit_bias:
        indices = mx.array(list(logit_bias.keys()))
        values = mx.array(list(logit_bias.values()))

        def logit_bias_processor(tokens: List[int], logits):
            return logits.at[:, indices].add(values)

        logits_processors.append(logit_bias_processor)

    penalties = [
        (make_repetition_penalty, repetition_penalty, repetition_context_size),
        (make_presence_penalty, presence_penalty, presence_context_size),
        (make_frequency_penalty, frequency_penalty, frequency_context_size),
    ]

    for make_penalty, penalty, context_size in penalties:
        if penalty is not None and penalty != 0:
            logits_processors.append(make_penalty(penalty, context_size))

    return logits_processors
