# Per-expert mixed quantization (MixedQuant)

The AgrillaMoE workflow measures which experts actually fire (server run
with `--moe-expert-log run.jsonl` → `moe_predict.py` → per-layer rankings,
see the fork's `expert-rank/`) and builds GGUFs where the hot experts carry
more bits than the cold tail (`make_mixed_gguf.py`). AgrimlxO brings the
same idea to MLX checkpoints: each MoE layer's routed experts are split
into quantization **groups by expert id** — e.g. the top-M ranked experts
at 8-bit, the rest at 4-bit — each group stored as its own stacked
quantized tensor, loaded natively by
`models/mixed_experts.MixedSwitchGLU`.

Memory, not speed, is the point: the expansion wants headroom for N ranks
per token; the cold tail is where the bits are cheapest to cut. See the
paper (README) for the low-bit deployment recipe.

## Format

Per MoE layer, the stacked `switch_mlp` projections are replaced by groups:

```
...mlp.switch_mlp.groups.0.gate_proj_weight / _scales / _biases   (hot experts, e.g. 8-bit)
...mlp.switch_mlp.groups.1.gate_proj_weight / _scales / _biases   (cold tail, e.g. 4-bit)
...up_proj..., ...down_proj...
```

(standard `mx.quantize` tuples stacked over the group's experts) annotated
in `config.json`:

```json
"moe_expert_quant": {
  "25": [{"bits": 8, "group_size": 64, "experts": [41, 190, 78, ...]},
         {"bits": 4, "group_size": 64}]
}
```

Layer keys are 0-based indices; missing layers stay uniformly quantized;
the last group without `experts` takes every remaining expert; the router
gate keeps its uniform 8-bit protection.

## Tool

```bash
# rank the experts (server run with --moe-expert-log, then e.g.
# moe_predict.py from the AgrillaMoE tooling) → expert-order.txt:
#   one line per layer, comma-separated expert ids, most-important first
agrimlxo mixed-quantize <model-dir> --order expert-order.txt \
    --top-m 64 --hi-bits 8 --lo-bits 4 --layers 0--1 --out model-mixed

# or a plain plan without rankings (natural order):
agrimlxo mixed-quantize <model-dir> --top-m 32 --out model-mixed
```

The conversion loads the source checkpoint (uniform quant or bf16),
dequantizes each `switch_mlp`, re-quantizes the two groups and writes the
new safetensors + config. Everything else (attention, shared experts,
router gate, other layers) keeps the source quantization untouched.

## Serving

Nothing changes at serve time: the loader reads `moe_expert_quant`, builds
`MixedSwitchGLU` for the annotated layers and loads the group tensors. The
expansion composes with it directly — the rank-ordered top-N selection
feeds `MixedSwitchGLU` exactly like `SwitchGLU` (verified by
`test_mixed_model_with_expansion`).

## Trade-offs (read before converting)

* **Compute**: the per-group slot counts are data-dependent, so static
  per-group slicing is impossible in a lazy graph. Every group computes
  the full slot set and non-members are masked away: G× the routed compute
  on mixed layers (G = 2 typically). The memory win is what buys the
  bigger expert budget on a constrained Mac; the compute cost is the price.
  A variable-count execution (group-sorted gathers with dynamic
  boundaries, or the omlx block-sorting kernels) is the future fix.
* **The hot/cold split is only as good as the ranking** — measure per
  model with the paired protocol (paper, README) before and after.

## Tests

`tests/test_kernels_mixed.py`: group validation, exact per-slot math
against a dequantized reference (same-bits groups), the full conversion
roundtrip (uniform toy → mixed-quantize → load → deterministic generation),
and expansion-through-mixed-layers composition.
