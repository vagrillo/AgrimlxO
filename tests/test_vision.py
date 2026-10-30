# SPDX-License-Identifier: Apache-2.0
"""Vision pipeline tests: toy qwen3_5_moe VLM checkpoint with a small real
vision tower (random weights — validates the wiring: tower build, image
splice, interleaved mrope positions, engine passthrough), plus the mrope
invariants."""

import json
from pathlib import Path

import mlx.core as mx
import numpy as np
import pytest

from agrimlxo.loader import LoadedModel
from agrimlxo.models.mrope import MRoPERotaryEmbedding

from test_engine_toy import PROMPT

VOCAB = 64
IMAGE_TOKEN, VIDEO_TOKEN, VISION_START = 62, 61, 60


def build_toy_vlm(base: Path):
    from agrimlxo.models.qwen3_5_moe import (
        Model as Q35Model,
        TextModelArgs,
        ModelArgs as Q35Args,
    )
    from agrimlxo.models.vision import VisionModel
    from agrimlxo.models.vision_config import VisionConfig
    from dataclasses import asdict
    from mlx.utils import tree_flatten

    text_args = TextModelArgs(
        model_type="qwen3_5_moe_text",
        hidden_size=64,
        num_hidden_layers=4,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        rms_norm_eps=1e-6,
        vocab_size=VOCAB,
        linear_num_value_heads=8,
        linear_num_key_heads=4,
        linear_key_head_dim=16,
        linear_value_head_dim=16,
        linear_conv_kernel_dim=4,
        full_attention_interval=4,
        num_experts=8,
        num_experts_per_tok=2,
        moe_intermediate_size=32,
        shared_expert_intermediate_size=32,
        norm_topk_prob=True,
        tie_word_embeddings=True,
        rope_parameters={
            "type": "default",
            "rope_theta": 10000.0,
            "partial_rotary_factor": 0.5,
            "mrope_section": [11, 11, 0],
        },
        max_position_embeddings=256,
    )
    vcfg = VisionConfig(
        depth=2, hidden_size=64, intermediate_size=128, out_hidden_size=64,
        num_heads=4, patch_size=14, spatial_merge_size=2, temporal_patch_size=2,
        num_position_embeddings=16, window_size=28, fullatt_block_indexes=[1],
        deepstack_visual_indexes=[], in_channels=3, layer_norm_eps=1e-6,
    )
    text_config = asdict(text_args)
    text_config["model_type"] = "qwen3_5_moe_text"
    config = {
        "model_type": "qwen3_5_moe",
        "text_config": text_config,
        "image_token_id": IMAGE_TOKEN,
        "video_token_id": VIDEO_TOKEN,
        "vision_start_token_id": VISION_START,
        "vision_config": asdict(vcfg),
    }
    args = Q35Args(
        model_type="qwen3_5_moe", text_config=text_config,
        image_token_id=IMAGE_TOKEN, video_token_id=VIDEO_TOKEN,
        vision_start_token_id=VISION_START, vision_config=asdict(vcfg),
    )
    mx.random.seed(11)
    model = Q35Model(args)
    model.vision_tower = VisionModel(vcfg)
    model.eval()

    d = base / "toy-vlm"
    d.mkdir(parents=True, exist_ok=True)
    all_weights = dict(tree_flatten(model.parameters()))
    text_w = {k: v for k, v in all_weights.items() if not k.startswith("vision_tower")}
    vision_w = {k: v for k, v in all_weights.items() if k.startswith("vision_tower")}
    mx.save_safetensors(str(d / "model.safetensors"), text_w)
    mx.save_safetensors(str(d / "model-vision.safetensors"), vision_w)
    (d / "config.json").write_text(json.dumps(config))
    return d


@pytest.fixture(scope="module")
def vlm_model(tmp_path_factory):
    d = build_toy_vlm(tmp_path_factory.mktemp("vlm"))
    lm = LoadedModel(d)
    assert lm.model.vision_tower is not None
    return lm


def test_vision_tower_forward_shapes(vlm_model):
    grid = mx.array([[1, 4, 4]])  # t,h,w patches -> 2x2 merged tokens
    pv = mx.random.normal((16, 3, 2, 14, 14))
    feats = vlm_model.model.run_vision(pv, grid)
    mx.eval(feats)
    assert feats.shape == (4, 64)  # (t*h/2)*(w/2) tokens, hidden 64


def test_splice_replaces_image_tokens(vlm_model):
    prompt = [1, 2, VISION_START, IMAGE_TOKEN, IMAGE_TOKEN, 3]
    embeds = vlm_model.model.language_model.model.embed_tokens(mx.array([prompt]))
    feats = mx.random.normal((2, 64))
    out = vlm_model.model.splice_image_features(
        feats, embeds, mx.array([prompt]), IMAGE_TOKEN
    )
    mx.eval(out)
    assert out.shape == embeds.shape
    # the two image positions now carry the two features
    assert np.allclose(np.array(out)[0, 3], np.array(feats)[0], atol=1e-6)
    assert np.allclose(np.array(out)[0, 4], np.array(feats)[1], atol=1e-6)
    # non-image positions untouched
    assert np.allclose(
        np.array(out)[0, 0], np.array(embeds)[0, 0], atol=1e-6
    )


def test_get_rope_index_mrope_pattern(vlm_model):
    # image grid 1x4x4 -> merged 1x2x2 = 4 image tokens after vision_start
    n_img_tokens = 4
    prompt = (
        [1, 2, VISION_START] + [IMAGE_TOKEN] * n_img_tokens + [3, 4]
    )
    grid = mx.array([[1, 4, 4]])
    pos, delta = vlm_model.model.get_rope_index(mx.array([prompt]), grid)
    mx.eval(pos)
    p = np.array(pos)
    assert pos.shape == (3, 1, len(prompt))
    # before the image: plain text positions on all axes
    assert list(p[:, 0, 0]) == [0, 0, 0]
    assert list(p[:, 0, 1]) == [1, 1, 1]
    # image positions continue after the text: first image token at
    # text_len + st_idx = 3, then t/h/w inside the 1x2x2 merged grid
    assert list(p[:, 0, 3]) == [3, 3, 3]
    assert list(p[:, 0, 4]) == [3, 3, 4]
    assert list(p[:, 0, 5]) == [3, 4, 3]
    assert list(p[:, 0, 6]) == [3, 4, 4]
    # the token right after the image block resumes after the max position
    assert p[0, 0, -2] == p[1, 0, -2] and p[0, 0, -2] > 4
    # delta: max position + 1 - L
    assert delta == int(p.max()) + 1 - len(prompt)


def test_multimodal_forward_and_decode(vlm_model):
    prompt = [1, 2, VISION_START] + [IMAGE_TOKEN] * 4 + [3, 4, 5]
    grid = mx.array([[1, 4, 4]])
    pv = mx.random.normal((16, 3, 2, 14, 14))
    out = vlm_model.model(
        mx.array([prompt]),
        cache=vlm_model.make_cache(),
        pixel_values=pv,
        image_grid_thw=grid,
    )
    mx.eval(out)
    assert out.shape == (1, len(prompt), VOCAB)
    assert np.isfinite(np.array(out)).all()


def test_mrope_equals_plain_rope_when_axes_equal():
    r = MRoPERotaryEmbedding(64, base=1e7)
    q = mx.random.normal((1, 4, 8, 256))
    k = mx.random.normal((1, 2, 8, 256))
    pos1 = mx.arange(8).reshape(1, 1, 8)
    qq, kk = r(q, k, mx.tile(pos1, (3, 1, 1)))
    from mlx.nn.layers.positional_encoding import RoPE

    ref = RoPE(64, base=1e7)(q, offset=0)
    assert np.allclose(np.array(qq[..., :64]), np.array(ref[..., :64]), atol=1e-3)
    assert np.allclose(np.array(qq[..., 64:]), np.array(q[..., 64:]), atol=1e-6)
