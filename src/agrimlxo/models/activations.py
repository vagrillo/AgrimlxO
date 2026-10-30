# Copyright © 2023 Apple Inc.
# SPDX-License-Identifier: Apache-2.0
"""Vendored from mlx-lm (ml-explore/mlx-lm @ 0d4d3d04, mlx_lm/models/activations.py).

Trimmed to the activations used by the vendored models.
"""

from functools import partial

import mlx.core as mx
import mlx.nn as nn


@partial(mx.compile, shapeless=True)
def swiglu(gate, x):
    return nn.silu(gate) * x


@partial(mx.compile, shapeless=True)
def precise_swiglu(h, gate, x):
    gate = nn.silu(gate.astype(mx.float32))
    x = x.astype(mx.float32)
    return (gate * x).astype(h.dtype)
