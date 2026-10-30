# Copyright © 2023 Apple Inc.
# SPDX-License-Identifier: Apache-2.0
"""Vendored from mlx-lm (ml-explore/mlx-lm @ 0d4d3d04, mlx_lm/models/cache.py).

KV cache classes used by the vendored models. Trimmed to the pieces AgrimlxO
needs (KVCache, QuantizedKVCache, ArraysCache) plus a small AgrimlxO extension:
KVCache accepts an optional ``dtype`` backing the --cache-type-k/v server flags
(f16/f32), matching llama.cpp's cache-type semantics.
"""

from typing import List, Optional

import mlx.core as mx
from mlx.utils import tree_map, tree_reduce

from .base import create_causal_mask


def create_attention_mask(
    N: int, offset: int, return_array: bool, window_size: Optional[int]
):
    if window_size is not None:
        return create_causal_mask(N, offset, window_size=window_size)
    elif N == 1:
        return None
    elif return_array:
        return create_causal_mask(N, offset, window_size=window_size)
    else:
        return "causal"


class _BaseCache:
    @property
    def state(self):
        return []

    @state.setter
    def state(self, v):
        if v is not None and v:
            raise ValueError("This cache has no state but a state was set.")

    def is_trimmable(self):
        return False

    def size(self):
        """Return the size (i.e. sequence length) of the cache."""
        return 0

    @property
    def nbytes(self):
        raise NotImplementedError("Cache sub-class must implement nbytes")

    def empty(self):
        raise NotImplementedError("Cache sub-class must implement this.")


class QuantizedKVCache(_BaseCache):
    step = 256

    def __init__(self, group_size: int = 64, bits: int = 8):
        self.keys = None
        self.values = None
        self.offset = 0
        self.group_size = group_size
        self.bits = bits

    def update_and_fetch(self, keys, values):
        B, n_kv_heads, num_steps, k_head_dim = keys.shape
        v_head_dim = values.shape[-1]
        prev = self.offset

        if self.keys is None or (prev + num_steps) > self.keys[0].shape[-2]:
            el_per_int = 8 * mx.uint32.size // self.bits
            new_steps = (self.step + num_steps - 1) // self.step * self.step
            shape = (B, n_kv_heads, new_steps)

            def init_quant(dim):
                return (
                    mx.zeros((*shape, dim // el_per_int), dtype=mx.uint32),
                    mx.zeros((*shape, dim // self.group_size), dtype=keys.dtype),
                    mx.zeros((*shape, dim // self.group_size), dtype=keys.dtype),
                )

            def expand_quant(x):
                new_x = mx.zeros((*shape, x.shape[-1]), dtype=x.dtype)
                return mx.concatenate([x, new_x], axis=-2)

            if self.keys is not None:
                if prev % self.step != 0:
                    self.keys, self.values = tree_map(
                        lambda x: x[..., :prev, :], (self.keys, self.values)
                    )

                self.keys, self.values = tree_map(
                    expand_quant, (self.keys, self.values)
                )
            else:
                self.keys, self.values = init_quant(k_head_dim), init_quant(v_head_dim)

        self.offset += num_steps

        keys = mx.quantize(keys, group_size=self.group_size, bits=self.bits)
        values = mx.quantize(values, group_size=self.group_size, bits=self.bits)
        for i in range(len(self.keys)):
            self.keys[i][..., prev : self.offset, :] = keys[i]
            self.values[i][..., prev : self.offset, :] = values[i]

        return self.keys_and_values()

    def keys_and_values(self):
        if self.offset == self.keys[0].shape[2]:
            return self.keys, self.values
        else:
            return tree_map(
                lambda x: x[..., : self.offset, :], (self.keys, self.values)
            )

    @property
    def state(self):
        return self.keys, self.values, self.offset, self.group_size, self.bits

    @state.setter
    def state(self, v):
        self.keys, self.values, self.offset, self.group_size, self.bits = v

    def is_trimmable(self):
        return True

    def trim(self, n):
        n = min(self.offset, n)
        self.offset -= n
        return n

    def make_mask(self, *args, **kwargs):
        return create_attention_mask(*args, offset=self.offset, **kwargs)

    def empty(self):
        return self.keys is None

    @property
    def nbytes(self):
        return tree_reduce(lambda a, x: a + x.nbytes, (self.keys, self.values), 0)


class KVCache(_BaseCache):
    step = 256

    def __init__(self, dtype: Optional[mx.Dtype] = None):
        self.keys = None
        self.values = None
        self.offset = 0
        self.dtype = dtype

    def _cast(self, x: mx.array) -> mx.array:
        if self.dtype is not None and x.dtype != self.dtype:
            return x.astype(self.dtype)
        return x

    def update_and_fetch(self, keys, values):
        prev = self.offset
        keys = self._cast(keys)
        values = self._cast(values)
        if self.keys is None or (prev + keys.shape[2]) > self.keys.shape[2]:
            B, n_kv_heads, _, k_head_dim = keys.shape
            v_head_dim = values.shape[3]
            n_steps = (self.step + keys.shape[2] - 1) // self.step
            k_shape = (B, n_kv_heads, n_steps * self.step, k_head_dim)
            v_shape = (B, n_kv_heads, n_steps * self.step, v_head_dim)
            new_k = mx.zeros(k_shape, keys.dtype)
            new_v = mx.zeros(v_shape, values.dtype)
            if self.keys is not None:
                if prev % self.step != 0:
                    self.keys = self.keys[..., :prev, :]
                    self.values = self.values[..., :prev, :]
                self.keys = mx.concatenate([self.keys, new_k], axis=2)
                self.values = mx.concatenate([self.values, new_v], axis=2)
            else:
                self.keys, self.values = new_k, new_v

        self.offset += keys.shape[2]
        self.keys[..., prev : self.offset, :] = keys
        self.values[..., prev : self.offset, :] = values
        return self.keys_and_values()

    def keys_and_values(self):
        if self.offset < self.keys.shape[2]:
            return self.keys[..., : self.offset, :], self.values[..., : self.offset, :]
        else:
            return self.keys, self.values

    def size(self):
        return self.offset

    @property
    def state(self):
        return self.keys, self.values, self.offset

    @state.setter
    def state(self, v):
        self.keys, self.values, self.offset = v

    def is_trimmable(self):
        return True

    def trim(self, n):
        n = min(self.offset, n)
        self.offset -= n
        return n

    def make_mask(self, *args, **kwargs):
        return create_attention_mask(*args, offset=self.offset, **kwargs)

    def empty(self):
        return self.keys is None

    @property
    def nbytes(self):
        if self.keys is None:
            return 0
        return self.keys.nbytes + self.values.nbytes


class ArraysCache(_BaseCache):
    """Cache for arbitrary per-layer arrays (used by the linear-attention
    layers: [0] conv state, [1] recurrent state)."""

    def __init__(self, size, left_padding: Optional[List[int]] = None):
        self.cache = [None] * size
        self.left_padding = mx.array(left_padding) if left_padding else None
        self.lengths = None
        self.offset = 0

    @property
    def batch_size(self):
        for c in self.cache:
            if c is not None:
                return c.shape[0]
        if self.left_padding is not None:
            return self.left_padding.size
        elif self.lengths is not None:
            return self.lengths.size
        else:
            return 1

    def __setitem__(self, idx, value):
        self.cache[idx] = value

    def __getitem__(self, idx):
        return self.cache[idx]

    @property
    def state(self):
        return self.cache, self.left_padding, self.lengths

    @state.setter
    def state(self, v):
        self.cache, self.left_padding, self.lengths = v

    def prepare(self, lengths=None, **kwargs):
        self.lengths = mx.array(lengths)

    def finalize(self):
        self.lengths = None
        self.left_padding = None

    def advance(self, N):
        self.offset += N
        if self.lengths is not None:
            self.lengths -= N
        if self.left_padding is not None:
            self.left_padding -= N

    def make_mask(self, N: int):
        pos = mx.arange(N)
        mask = None
        if self.left_padding is not None:
            mask = pos >= self.left_padding[:, None]
        if self.lengths is not None:
            in_length = pos < self.lengths[:, None]
            mask = in_length if mask is None else (mask & in_length)
        return mask

    def empty(self):
        return self.cache[0] is None

    @property
    def nbytes(self):
        return sum(c.nbytes for c in self.cache if c is not None)


def make_cache_for_layers(layer_kinds: List[str], cache_type: str = "f16"):
    """Build the per-layer cache list for the vendored models.

    ``layer_kinds``: "full_attention" or "linear_attention" per layer.
    ``cache_type``: f16 | f32 | q8_0 | q4_0 (llama.cpp-style --cache-type-k/v).
    """
    dtype = None
    kv_kwargs = {}
    if cache_type == "f32":
        dtype = mx.float32
    elif cache_type == "f16":
        dtype = mx.float16
    elif cache_type == "q8_0":
        kv_kwargs = dict(group_size=32, bits=8)
    elif cache_type == "q4_0":
        kv_kwargs = dict(group_size=32, bits=4)
    else:
        raise ValueError(f"unsupported cache type '{cache_type}'")

    caches = []
    for kind in layer_kinds:
        if kind == "linear_attention":
            caches.append(ArraysCache(size=2))
        elif kv_kwargs:
            caches.append(QuantizedKVCache(**kv_kwargs))
        else:
            caches.append(KVCache(dtype=dtype))
    return caches
