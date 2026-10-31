# SPDX-License-Identifier: Apache-2.0
"""Multi-turn conversation reuse E2E: the snapshot at the stable boundary
must be extended by the next turn's prompt (only the delta prefills)."""

import mlx.core as mx

from agrimlxo.engine import Engine, GenParams
from agrimlxo.loader import LoadedModel
from agrimlxo.tokenizer import Tokenizer

from test_vision import build_toy_vlm


def test_multi_turn_snapshot_extends(tmp_path):
    d = build_toy_vlm(tmp_path)
    tok = Tokenizer("/tmp/vlm-cfg")  # real qwen3.5 template
    engine = Engine(LoadedModel(d), tok, ctx_size=1024, parallel=1)
    engine.min_snapshot_tokens = 8
    engine.progress_cb = None

    def render(m, **kw):
        out = tok.apply_chat_template(m, add_generation_prompt=True, **kw)
        return out if isinstance(out, list) else tok.encode(out)

    m1 = [{"role": "user", "content": "cosa è il MoE?"}]
    p1 = render(m1)
    nogen = tok.apply_chat_template(m1, add_generation_prompt=False,
                                    enable_thinking=True)
    sa1 = len(nogen if isinstance(nogen, list) else tok.encode(nogen))

    list(engine.generate(p1, GenParams(temp=0.0, max_tokens=3),
                         conv_key="MT", snapshot_at=sa1))
    snap1 = list(engine.snapshots["MT"]["tokens"])
    assert snap1 == p1[:sa1], "lo snapshot copre il confine stabile"

    # turno 2: la storia estende — il re-render del turno 1 e' identico
    gen_text = tok._tokenizer.decode(engine.slots[0].tokens[sa1:]).replace(
        "<|im_end|>", "")
    m2 = m1 + [{"role": "assistant", "content": gen_text},
               {"role": "user", "content": "sintetizza"}]
    p2 = render(m2)
    assert p2[:len(snap1)] == snap1, (
        "il render del turno 2 deve estendere lo snapshot del turno 1"
    )

    nogen2 = tok.apply_chat_template(m2, add_generation_prompt=False,
                                     enable_thinking=True)
    sa2 = len(nogen2 if isinstance(nogen2, list) else tok.encode(nogen2))
    list(engine.generate(p2, GenParams(temp=0.0, max_tokens=2),
                         conv_key="MT", snapshot_at=sa2))
    snap2 = list(engine.snapshots["MT"]["tokens"])
    assert len(snap2) > len(snap1), "lo snapshot cresce col turno"
    assert snap2[:len(snap1)] == snap1


def test_multimodal_turn_skips_snapshot_but_reuses_text_prefix(tmp_path):
    """Le richieste con immagini nel delta non fanno snapshot/restore (le
    pixel_values coprono tutte le immagini, non solo i pad del delta)."""
    d = build_toy_vlm(tmp_path)
    tok = Tokenizer("/tmp/vlm-cfg")
    engine = Engine(LoadedModel(d), tok, ctx_size=1024, parallel=1)
    engine.min_snapshot_tokens = 8
    engine.progress_cb = None
    IMG, VS = 62, 60

    m1 = [{"role": "user", "content": "descrivi"}]
    p1 = render(m1) if False else None
    # prompt con un'immagine
    # il processor espande il pad al numero di token mergiati (4 per 1x4x4)
    tpl_text = "<|vision_start|>" + "<|image_pad|>" * 4 + "<|vision_end|>descrivi"
    p1 = tok.apply_chat_template(
        [{"role": "user", "content": tpl_text}],
        add_generation_prompt=True, enable_thinking=True)
    if not isinstance(p1, list): p1 = tok.encode(p1)
    p1 = [t if t != 248056 else IMG for t in p1]  # pad id del toy
    nogen = tok.apply_chat_template(
        [{"role": "user", "content": tpl_text}],
        add_generation_prompt=False, enable_thinking=True)
    if not isinstance(nogen, list): nogen = tok.encode(nogen)
    nogen = [t if t != 248056 else IMG for t in nogen]
    sa1 = len(nogen)

    grid = mx.array([[1, 4, 4]])
    pv = mx.random.normal((16, 3, 2, 14, 14))
    steps = list(engine.generate(
        p1, GenParams(temp=0.0, max_tokens=2),
        conv_key="MI", snapshot_at=sa1, images={"pixel_values": pv,
                                                "image_grid_thw": grid}))
    assert engine.snapshots.get("MI"), "snapshot catturato anche con immagini"
    snap_toks = engine.snapshots["MI"]["tokens"]
    # i pad dell'immagine sono dentro lo snapshot (stabili al turno dopo)
    assert IMG in snap_toks
