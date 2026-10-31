# Real-time per-expert quantization driven by usage

Hypothesis implemented in two stages: the model on disk stays **uniform**
(e.g. 8-bit single source), and the per-expert bit-width is applied in
memory.

## Stage 1 — implemented: load-time mixing

`--mixed-from-usage --top-m 64 --hi-bits 8 --lo-bits 4`:

1. The all-time usage ranking (`./expertusage/<model>.json`, fed by the
   USAGE heatmap) orders the experts per layer, hot first.
2. At load, each MoE layer's switch_mlp is dequantized in memory,
   re-quantized into two groups (top-M hot at hi-bits, tail at lo-bits) and
   swapped for `MixedSwitchGLU`. No disk round-trip; the uniform checkpoint
   remains the single source and can be re-mixed with a different
   `--top-m` at any restart.
3. Round-trip verified: dequantize(uniform) -> per-group quantize ->
   MixedSwitchGLU forward (tests/test_conv_kache.py covers the session
   flow; the toy mix test the swap + forward).

## Stage 2 — design: runtime re-mixing

While serving, usage drifts: a background "re-mixer" (thread, every N
tokens or on demand from the USAGE panel) would:

1. Recompute hot/cold from `counts_session + counts_all` (weighted).
2. Only swap the experts whose group CHANGED (dequantize that expert from
   the uniform source, requantize into the target group tensor slice) —
   `MixedSwitchGLU` groups are stacked per expert, so a single expert moves
   with a slice assignment, not a full re-quantize.
3. Swap under the engine lock between requests: the KV cache stays valid
   (weights-only change), outputs may shift slightly mid-conversation —
   acceptable for a quantization change, documented.
4. Cost per swap: dequantize+requantize of the moved experts only
   (~1-2 MB/expert at 6-bit → sub-second).

## Why not re-quantize per token

Per-token bit-width changes would require dequantizing the whole expert on
the fly (the disk format is uniform): that's a gather + dequant of ~1 MB per
expert per token — not viable on Metal today. The load-time split plus
occasional runtime re-mixing captures the memory/performance win without
per-token cost.
