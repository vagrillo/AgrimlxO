# SPDX-License-Identifier: Apache-2.0
"""Model registry: model_type -> implementation + MoE geometry access."""

from dataclasses import dataclass
from typing import Optional, Tuple

from .qwen3_5_moe import Model as Qwen3_5MoeModel
from .qwen3_moe import Model as Qwen3MoeModel


@dataclass
class MoEGeometry:
    """Routing geometry of a loaded model, for the expansion runtime."""

    is_moe: bool
    n_layer: int
    n_expert: int
    top_k: int
    norm_topk_prob: bool
    # First MoE layer index, for config parsing before load.
    layer_start_hint: int = 0


def _geometry_qwen35(config: dict) -> MoEGeometry:
    text = config.get("text_config", config)
    n_expert = int(text.get("num_experts", 0) or 0)
    top_k = int(text.get("num_experts_per_tok", 0) or 0)
    return MoEGeometry(
        is_moe=n_expert > 0,
        n_layer=int(text.get("num_hidden_layers", 0)),
        n_expert=n_expert,
        top_k=top_k,
        norm_topk_prob=bool(text.get("norm_topk_prob", True)),
    )


def _geometry_qwen3moe(config: dict) -> MoEGeometry:
    n_expert = int(config.get("num_experts", 0) or 0)
    top_k = int(config.get("num_experts_per_tok", 0) or 0)
    return MoEGeometry(
        is_moe=n_expert > 0,
        n_layer=int(config.get("num_hidden_layers", 0)),
        n_expert=n_expert,
        top_k=top_k,
        norm_topk_prob=bool(config.get("norm_topk_prob", True)),
    )


_MODEL_REGISTRY = {
    "qwen3_5_moe": (Qwen3_5MoeModel, _geometry_qwen35),
    "qwen3_moe": (Qwen3MoeModel, _geometry_qwen3moe),
}

# Display aliases for the same implementations.
_TYPE_ALIASES = {
    "qwen3_5_moe_text": "qwen3_5_moe",
    "qwen35_moe": "qwen3_5_moe",
    "qwen3_moe_35b": "qwen3_moe",
}


def get_model_class(config: dict):
    """Return the Model class + geometry extractor for a config dict."""
    model_type = config.get("model_type", "")
    model_type = _TYPE_ALIASES.get(model_type, model_type)
    if model_type not in _MODEL_REGISTRY:
        raise ValueError(
            f"unsupported model_type '{model_type}': AgrimlxO is dedicated to the"
            " Qwen3.x MoE family (qwen3_5_moe, qwen3_moe)"
        )
    model_cls, geometry_fn = _MODEL_REGISTRY[model_type]
    return model_cls, geometry_fn(config)


def infer_geometry(config: dict) -> Optional[MoEGeometry]:
    """MoE geometry from a raw config dict (used before/independent of load)."""
    try:
        _, geometry = get_model_class(config)
        return geometry
    except ValueError:
        return None
