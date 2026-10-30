# Copyright © 2025 Apple Inc.
# SPDX-License-Identifier: Apache-2.0
"""Qwen3 MoE (``qwen3_moe``) — MLX implementation with the AgrimlxO MoE expert
expansion hooks.

Vendored from mlx-lm (ml-explore/mlx-lm @ 0d4d3d04, mlx_lm/models/qwen3_moe.py),
trimmed to single-device serving, with the routed branch going through the
shared ``routed_forward`` (stock path when the expansion is off, rank-ordered
top-N with the expansion post-pass when active).

This covers the classic dense-attention Qwen MoE family (Qwen3-30B-A3B and
friends: every layer full attention with q/k RMSNorm, sparse MoE on the layers
selected by decoder_sparse_step / mlp_only_layers).
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union

import mlx.core as mx
import mlx.nn as nn

from .base import (
    BaseModelArgs,
    create_attention_mask,
    scaled_dot_product_attention,
)
from .cache import KVCache, make_cache_for_layers
from .pipeline import PipelineMixin
from .switch_layers import SwitchGLU
from .moe_block import routed_forward
from ..moe_expansion import MoERuntime


@dataclass
class ModelArgs(BaseModelArgs):
    model_type: str
    hidden_size: int
    num_hidden_layers: int
    intermediate_size: int
    num_attention_heads: int
    num_experts: int
    num_experts_per_tok: int
    moe_intermediate_size: int
    rms_norm_eps: float
    vocab_size: int
    num_key_value_heads: int
    rope_theta: float
    tie_word_embeddings: bool
    max_position_embeddings: int
    norm_topk_prob: bool
    # optional in some checkpoints
    decoder_sparse_step: int = 1
    mlp_only_layers: List[int] = field(default_factory=list)
    head_dim: Optional[int] = None
    rope_scaling: Optional[Dict[str, Union[float, str]]] = None
    attention_bias: bool = False

    def __post_init__(self):
        if self.head_dim is None:
            self.head_dim = self.hidden_size // self.num_attention_heads


class Attention(nn.Module):
    def __init__(self, args: ModelArgs, layer_idx: int):
        super().__init__()

        dim = args.hidden_size
        self.n_heads = n_heads = args.num_attention_heads
        assert args.num_key_value_heads is not None
        self.n_kv_heads = n_kv_heads = args.num_key_value_heads

        head_dim = getattr(args, "head_dim", args.hidden_size // args.num_attention_heads)
        self.scale = head_dim**-0.5

        self.q_proj = nn.Linear(
            dim, n_heads * head_dim, bias=args.attention_bias
        )
        self.k_proj = nn.Linear(
            dim, n_kv_heads * head_dim, bias=args.attention_bias
        )
        self.v_proj = nn.Linear(
            dim, n_kv_heads * head_dim, bias=args.attention_bias
        )
        self.o_proj = nn.Linear(n_heads * head_dim, dim, bias=False)

        self.q_norm = nn.RMSNorm(head_dim, eps=args.rms_norm_eps)
        self.k_norm = nn.RMSNorm(head_dim, eps=args.rms_norm_eps)

        self.rope = nn.RoPE(
            head_dim,
            traditional=False,
            base=args.rope_theta,
        )

    def __call__(
        self,
        x: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[Any] = None,
    ) -> mx.array:
        B, L, D = x.shape

        queries, keys, values = self.q_proj(x), self.k_proj(x), self.v_proj(x)

        # Prepare the queries, keys and values for the attention computation
        queries = self.q_norm(queries.reshape(B, L, self.n_heads, -1)).transpose(
            0, 2, 1, 3
        )
        keys = self.k_norm(keys.reshape(B, L, self.n_kv_heads, -1)).transpose(
            0, 2, 1, 3
        )
        values = values.reshape(B, L, self.n_kv_heads, -1).transpose(0, 2, 1, 3)

        if cache is not None:
            queries = self.rope(queries, offset=cache.offset)
            keys = self.rope(keys, offset=cache.offset)
            keys, values = cache.update_and_fetch(keys, values)
        else:
            queries = self.rope(queries)
            keys = self.rope(keys)

        output = scaled_dot_product_attention(
            queries, keys, values, cache=cache, scale=self.scale, mask=mask
        )
        output = output.transpose(0, 2, 1, 3).reshape(B, L, -1)
        return self.o_proj(output)


class MLP(nn.Module):
    def __init__(self, dim, hidden_dim):
        super().__init__()
        self.gate_proj = nn.Linear(dim, hidden_dim, bias=False)
        self.down_proj = nn.Linear(hidden_dim, dim, bias=False)
        self.up_proj = nn.Linear(dim, hidden_dim, bias=False)

    def __call__(self, x) -> mx.array:
        from .activations import swiglu

        return self.down_proj(swiglu(self.gate_proj(x), self.up_proj(x)))


class SparseMoeBlock(nn.Module):
    def __init__(
        self,
        args: ModelArgs,
        expansion: Optional[MoERuntime] = None,
        layer_idx: int = 0,
        mixed_groups: Optional[List[dict]] = None,
    ):
        super().__init__()
        dim = args.hidden_size
        intermediate_size = args.moe_intermediate_size

        self.norm_topk_prob = args.norm_topk_prob
        self.num_experts = num_experts = args.num_experts
        self.top_k = args.num_experts_per_tok
        self.expansion = expansion
        self.layer_idx = layer_idx

        self.gate = nn.Linear(dim, num_experts, bias=False)
        if mixed_groups:
            from .mixed_experts import MixedSwitchGLU

            self.switch_mlp = MixedSwitchGLU(
                dim, intermediate_size, num_experts, mixed_groups
            )
        else:
            self.switch_mlp = SwitchGLU(dim, intermediate_size, num_experts)

    def __call__(self, x: mx.array) -> mx.array:
        y, _ = routed_forward(
            x,
            self.gate,
            self.switch_mlp,
            self.top_k,
            self.num_experts,
            self.norm_topk_prob,
            runtime=self.expansion,
            layer_idx=self.layer_idx,
        )
        return y


class Qwen3MoeDecoderLayer(nn.Module):
    def __init__(
        self,
        args: ModelArgs,
        layer_idx: int,
        expansion: Optional[MoERuntime] = None,
        mixed_groups: Optional[List[dict]] = None,
    ):
        super().__init__()
        self.hidden_size = args.hidden_size
        self.self_attn = Attention(args, layer_idx)

        self.input_layernorm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)
        self.post_attention_layernorm = nn.RMSNorm(
            args.hidden_size, eps=args.rms_norm_eps
        )
        self.args = args

        if (layer_idx not in args.mlp_only_layers) and (
            args.num_experts > 0 and (layer_idx + 1) % args.decoder_sparse_step == 0
        ):
            self.mlp = SparseMoeBlock(
                args,
                expansion=expansion,
                layer_idx=layer_idx,
                mixed_groups=mixed_groups,
            )
        else:
            self.mlp = MLP(args.hidden_size, args.intermediate_size)

    def __call__(
        self,
        x: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[Any] = None,
    ) -> mx.array:
        r = self.self_attn(self.input_layernorm(x), mask, cache)
        h = x + r
        r = self.mlp(self.post_attention_layernorm(h))
        out = h + r
        return out


class Qwen3MoeModel(PipelineMixin, nn.Module):
    def __init__(
        self,
        args: ModelArgs,
        expansion: Optional[MoERuntime] = None,
        mixed_quant: Optional[Dict[int, list]] = None,
    ):
        super().__init__()
        self.args = args
        self.vocab_size = args.vocab_size
        self.num_hidden_layers = args.num_hidden_layers
        assert self.vocab_size > 0
        self.embed_tokens = nn.Embedding(args.vocab_size, args.hidden_size)
        self.layers = [
            Qwen3MoeDecoderLayer(
                args=args,
                layer_idx=i,
                expansion=expansion,
                mixed_groups=(mixed_quant or {}).get(i),
            )
            for i in range(args.num_hidden_layers)
        ]
        self.norm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)

    def __call__(
        self,
        inputs: mx.array,
        cache=None,
        input_embeddings: Optional[mx.array] = None,
    ) -> mx.array:
        if input_embeddings is not None:
            h = input_embeddings
        else:
            h = self.embed_tokens(inputs)

        if cache is None:
            cache = [None] * len(self.layers)

        mask = create_attention_mask(h, cache[0])

        for layer, c in zip(self.layers, cache):
            h = layer(h, mask, c)

        return self.norm(h)


class Model(nn.Module):
    def __init__(
        self,
        args: ModelArgs,
        expansion: Optional[MoERuntime] = None,
        mixed_quant: Optional[Dict[int, list]] = None,
    ):
        super().__init__()
        self.args = args
        self.model_type = args.model_type
        self.model = Qwen3MoeModel(args, expansion=expansion, mixed_quant=mixed_quant)
        if not args.tie_word_embeddings:
            self.lm_head = nn.Linear(args.hidden_size, args.vocab_size, bias=False)

    def __call__(
        self,
        inputs: mx.array,
        cache=None,
        input_embeddings: Optional[mx.array] = None,
    ) -> mx.array:
        out = self.model(inputs, cache, input_embeddings)
        if self.args.tie_word_embeddings:
            out = self.model.embed_tokens.as_linear(out)
        else:
            out = self.lm_head(out)
        return out

    @property
    def text_args(self) -> ModelArgs:
        return self.args

    @property
    def layers(self):
        return self.model.layers

    def make_cache(self, cache_type: str = "f16"):
        return make_cache_for_layers(["full_attention"] * len(self.layers), cache_type)

    def sanitize(self, weights):
        if self.args.tie_word_embeddings:
            weights.pop("lm_head.weight", None)
        moe_layers = sorted(
            int(k.split(".")[2])
            for k in weights
            if k.startswith("model.layers.")
            and k.endswith(".mlp.experts.0.up_proj.weight")
        )
        for l in moe_layers:
            prefix = f"model.layers.{l}"
            for n in ["up_proj", "down_proj", "gate_proj"]:
                to_join = [
                    weights.pop(f"{prefix}.mlp.experts.{e}.{n}.weight")
                    for e in range(self.args.num_experts)
                ]
                weights[f"{prefix}.mlp.switch_mlp.{n}.weight"] = mx.stack(to_join)
        return weights

    @property
    def quant_predicate(self):
        def predicate(path, _):
            if path.endswith("mlp.gate"):
                return {"group_size": 64, "bits": 8}
            return True

        return predicate
