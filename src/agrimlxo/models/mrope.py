# SPDX-License-Identifier: Apache-2.0
"""MRoPE rotary embedding for Qwen3.5/3.6 (interleaved multimodal RoPE).

Clean reimplementation of the mlx-vlm semantics
(mlx_vlm/models/rope_utils.py): position_ids come as [3, B, L] (t/h/w axes)
or [B, L] (text-only); each rotary frequency reads one of the three axes
selected by the interleaved layout built from ``mrope_section``
(Qwen3.5: [11, 11, 0] - frequency f feeds axis 1 when f % 3 == 1 below
section[1]*3, axis 2 when f % 3 == 2 below section[2]*3, axis 0 elsewhere).
Only the first ``dim`` head features rotate; the rest pass through
(partial rotary).
"""

from typing import List, Sequence

import mlx.core as mx


def _interleaved_position_selector(
    mrope_section: Sequence[int], freq_dim: int
) -> mx.array:
    selector = [0] * freq_dim
    for dim, offset in enumerate((1, 2), start=1):
        for idx in range(offset, min(mrope_section[dim] * 3, freq_dim), 3):
            selector[idx] = dim
    return mx.array(selector, dtype=mx.int32)


def rotate_half(x: mx.array) -> mx.array:
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return mx.concatenate([-x2, x1], axis=-1)


class MRoPERotaryEmbedding:
    """Plain object (not an nn.Module): it holds no trainable parameters, and
    keeping it out of the module tree keeps inv_freq/position_selector out of
    the strict checkpoint load."""

    def __init__(
        self,
        dim: int,
        max_position_embeddings: int = 262144,
        base: float = 10000.0,
        mrope_section: Sequence[int] = (11, 11, 0),
    ):
        super().__init__()
        self.dim = dim  # number of rotated head features (2 * half)
        self.base = base
        self._mrope_section: List[int] = list(mrope_section)
        self.inv_freq = 1.0 / (
            base ** (mx.arange(0, dim, 2, dtype=mx.float32) / dim)
        )
        self.position_selector = _interleaved_position_selector(
            self._mrope_section, dim // 2
        )
        mx.eval(self.inv_freq)

    def _cos_sin(self, q: mx.array, position_ids: mx.array):
        if position_ids.ndim == 2:
            angle = position_ids.astype(mx.float32)[..., None] * self.inv_freq
        else:
            positions = mx.take(position_ids, self.position_selector, axis=0)
            angle = positions.transpose(1, 2, 0).astype(mx.float32) * self.inv_freq
        cos = mx.cos(angle)[:, None, :, :]
        sin = mx.sin(angle)[:, None, :, :]
        # half-split pairing: [cos | cos], [sin | sin] over the rotated dims
        cos = mx.concatenate([cos, cos], axis=-1)
        sin = mx.concatenate([sin, sin], axis=-1)
        return cos, sin

    def __call__(self, q: mx.array, k: mx.array, position_ids: mx.array):
        """q/k: [B, H, L, head_dim]; position_ids: [3, B, L] or [B, L]."""
        cos, sin = self._cos_sin(q, position_ids)
        d = self.dim

        def apply(x):
            x_rot = x[..., :d]
            x_pass = x[..., d:]
            x_embed = x_rot * cos + rotate_half(x_rot) * sin
            x_embed = x_embed.astype(x.dtype)
            if x_pass.shape[-1] == 0:
                return x_embed
            return mx.concatenate([x_embed, x_pass], axis=-1)

        return apply(q), apply(k)
