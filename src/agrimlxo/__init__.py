# SPDX-License-Identifier: Apache-2.0
"""AgrimlxO: dedicated MLX inference server for Qwen3.x-35B MoE with runtime
MoE expert expansion (layer-scoped expert-budget expansion).

The engine serves OpenAI-compatible HTTP APIs on Apple Silicon via Apple's
MLX framework. The MoE expansion feature is the MLX port of the runtime-only
routing modification described in

    Vincenzo Agrillo, "Succinct convergence ..." (Zenodo, 2026)
    https://zenodo.org/records/22255483

originally implemented in the llama.cpp fork `vagrillo/llama.cpp` (branch
`moe-expansion`) and shipped as AgrillaMoE.
"""

from ._version import __version__

__all__ = ["__version__"]
