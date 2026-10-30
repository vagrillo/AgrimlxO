# SPDX-License-Identifier: Apache-2.0
"""Model loading for AgrimlxO.

Follows the mlx-lm load flow (vendored model classes):
  config.json -> ModelArgs -> Model -> sanitize(weights) -> nn.quantize with
  per-layer overrides -> load_weights -> eval.

Model paths can be a local directory or a HuggingFace repo id (downloaded into
the HF cache via snapshot_download).
"""

import json
from pathlib import Path
from typing import Optional, Tuple

import mlx.core as mx
import mlx.nn as nn

from .models import get_model_class, MoEGeometry
from .moe_expansion import MoERuntime, MoEExpansionConfig, MoEReduceConfig


def resolve_model_path(model_id: str) -> Path:
    """Local directory or HF repo id -> local snapshot directory."""
    p = Path(model_id).expanduser()
    if p.is_dir():
        return p
    if "/" in model_id:
        from huggingface_hub import snapshot_download

        return Path(
            snapshot_download(
                model_id,
                allow_patterns=[
                    "*.json",
                    "*.safetensors",
                    "tokenizer*",
                    "*.py",  # rare remote code; sanitized away otherwise
                ],
                ignore_patterns=["*.gguf", "*.bin", "vision*", "*.pth"],
            )
        )
    raise FileNotFoundError(
        f"model '{model_id}' is not a local directory and not a HF repo id"
        " (expected 'org/name')"
    )


def load_config(model_path: Path) -> dict:
    with open(model_path / "config.json") as f:
        return json.load(f)


def _quantize_model(model: nn.Module, config: dict, weights: dict) -> None:
    quantization = config.get("quantization")
    if quantization is None:
        text = config.get("text_config")
        if isinstance(text, dict):
            quantization = text.get("quantization")
        if quantization is None:
            return
    if not isinstance(quantization, dict):
        return

    per_layer = {
        k: v for k, v in quantization.items() if isinstance(v, dict)
    }
    group_size = int(quantization.get("group_size", 64))
    bits = int(quantization.get("bits", 4))
    mode = quantization.get("mode", "affine")

    def class_predicate(p, m):
        # custom per-layer quantizations baked into the config
        if p in per_layer:
            override = dict(per_layer[p])
            if override.get("linear") is False and "bits" not in override:
                return False
            return override
        if not hasattr(m, "to_quantized"):
            return False
        # quantized checkpoints carry their scales in the weights
        if f"{p}.scales" not in weights:
            return False
        return {"group_size": group_size, "bits": bits, "mode": mode}

    nn.quantize(
        model,
        group_size=group_size,
        bits=bits,
        mode=mode,
        class_predicate=class_predicate,
    )


class LoadedModel:
    """A model ready to generate, plus its routing geometry and expansion
    runtime."""

    def __init__(
        self,
        model_path: Path,
        expansion: Optional[MoERuntime] = None,
        lazy: bool = False,
    ):
        self.path = model_path
        self.config = load_config(model_path)
        model_cls, geometry = get_model_class(self.config)
        self.geometry: MoEGeometry = geometry

        # expansion setup against the model geometry (fail fast)
        if expansion is not None:
            expansion.setup(
                is_moe=geometry.is_moe,
                k_native=geometry.top_k,
                n_expert=geometry.n_expert,
            )
        self.expansion = expansion

        import importlib

        model_module = importlib.import_module(model_cls.__module__)
        ModelArgs = getattr(model_module, "ModelArgs", None)
        if ModelArgs is None:
            raise ValueError(f"no ModelArgs for {model_cls}")
        model_args = ModelArgs.from_dict(self.config)
        model = model_cls(model_args, expansion=expansion)

        weight_files = sorted(model_path.glob("model*.safetensors"))
        if not weight_files:
            raise FileNotFoundError(f"no model*.safetensors found in {model_path}")
        weights = {}
        for wf in weight_files:
            weights.update(mx.load(wf))

        if hasattr(model, "sanitize"):
            weights = model.sanitize(weights)

        _quantize_model(model, self.config, weights)

        model.eval()
        model.load_weights(list(weights.items()), strict=True)
        if not lazy:
            mx.eval(model.parameters())
        self.model = model

    @property
    def layers(self):
        return self.model.layers

    def make_cache(self, cache_type: str = "f16"):
        return self.model.make_cache(cache_type)

    @property
    def layer_kinds(self):
        return [
            "linear_attention" if getattr(l, "is_linear", False) else "full_attention"
            for l in self.layers
        ]


def build_expansion_runtime(
    config: dict,
    expansion_cfg: Optional[MoEExpansionConfig],
    reduce_cfg: Optional[MoEReduceConfig],
) -> Optional[MoERuntime]:
    """Create the MoERuntime for a raw config dict (fail-fast validation
    happens in MoERuntime.setup at model load)."""
    geometry = None
    try:
        _, geometry = get_model_class(config)
    except ValueError:
        return None
    if expansion_cfg is None and reduce_cfg is None:
        return None
    return MoERuntime(
        n_layer=geometry.n_layer,
        norm_topk_prob=geometry.norm_topk_prob,
        config=expansion_cfg or MoEExpansionConfig(),
        reduce=reduce_cfg,
    )


def peek_moe_geometry(model_path: Path) -> Optional[Tuple[int, int, int, int]]:
    """(n_layer, n_expert, top_k, norm_topk_prob) from config.json without
    loading weights; used for the AgrillaMoE-style default profile injection."""
    try:
        config = load_config(model_path)
    except Exception:
        return None
    geometry = None
    try:
        _, geometry = get_model_class(config)
    except ValueError:
        return None
    if geometry is None:
        return None
    return (
        geometry.n_layer,
        geometry.n_expert,
        geometry.top_k,
        int(geometry.norm_topk_prob),
    )
