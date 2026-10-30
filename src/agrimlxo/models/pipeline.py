# SPDX-License-Identifier: Apache-2.0
"""Minimal single-node stand-in for mlx-lm's PipelineMixin.

AgrimlxO serves one model on one device, so pipelining is a no-op: every
layer is local and the graph runs end-to-end. The models only read the
``pipeline_*`` properties (upstream code paths), while ``self.layers`` stays
a plain MLX module attribute.
"""


class PipelineMixin:
    @property
    def pipeline_rank(self) -> int:
        return 0

    @property
    def pipeline_size(self) -> int:
        return 1

    @property
    def pipeline_layers(self):
        return self.layers
