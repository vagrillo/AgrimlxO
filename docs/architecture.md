# AgrimlxO — architecture

This document describes how the standalone engine is put together, which
invariants it maintains, and where the vendored code comes from.

## Goals (from the project instructions)

1. A **server-only** engine (no external GUI) like AgrillaMoE, with a
   built-in chat interface like llama.cpp's webui.
2. Optimized **exclusively for Qwen3.x-35B MoE** (reference: Qwen3.6-35B-A3B)
   using the **MLX** inference approach (as omlx does), adapted with the
   **MoE expansion** functionality defined in AgrillaMoE.
3. A **new standalone project** in `./src` with **no dependencies on the
   other projects** (AgrillaMoE, omlx) — kernels rewritten where needed.
4. Manual execution tests on the maintainer's Mac.

## Package layout

```
src/agrimlxo/
├── cli.py            entry point: serve / download / list, banner, profile injection
├── args.py           full flag surface (llama.cpp-server + AgrillaMoE moe-*)
├── server.py         FastAPI/uvicorn app: OpenAI API, SSE, /props, /slots, /metrics
├── engine.py         generation loop: chunked prefill, prefix-reuse cache,
│                     stop handling, reasoning budget, MoE stats driving
├── sampler.py        temp/top-k/top-p/min-p/XTC samplers + logits processors
├── tokenizer.py      AutoTokenizer wrapper + monotonic incremental detokenizer
├── loader.py         model resolution (local dir or HF), mlx-lm-style loading
├── moe_expansion.py  ★ MoERuntime, expansion/reduce post-passes, validation,
│                     banner, stats window, default profiles
├── webui/index.html  chat interface (vanilla JS, offline, no CDN)
└── models/
    ├── __init__.py     registry (model_type → class + MoE geometry)
    ├── qwen3_5_moe.py  ★ hybrid target architecture (text-only)
    ├── qwen3_moe.py    classic dense-attention MoE family
    ├── moe_block.py    ★ routed_forward(): stock vs expansion code paths
    ├── cache.py        KVCache (+dtype), QuantizedKVCache, ArraysCache
    ├── base.py         masks, SDPA (incl. quantized-KV path), BaseModelArgs
    ├── gated_delta.py  linear-attention kernels (Metal + pure-MLX fallback)
    ├── rope_utils.py   RoPE variants (default/linear/dynamic/llama3/yarn/longrope)
    ├── switch_layers.py SwitchLinear/QuantizedSwitchLinear/SwitchGLU
    ├── activations.py  swiglu / precise_swiglu
    └── pipeline.py     no-op PipelineMixin (single device)
```

★ = where the MoE expansion lives.

## Request flow

```
HTTP request → FastAPI (server.py)
  → chat template (tokenizer.py) → prompt token ids
  → slot acquisition (ServerState) → GenParams (server defaults + request overrides)
  → engine.generate(prompt, params, slot)  [worker thread, engine lock]
      → prefix-reuse check (engine._reuse_len): a cached sequence that is a
        strict prefix of the prompt continues from cache; else fresh cache
      → chunked prefill (batch-size chunks, mx.eval between chunks)
      → decode loop: logits → logits processors → sampler → detokenizer
        - eos / stop strings / max tokens / context limit
        - reasoning budget: at budget−24 tokens inside <think>, the wrap-up
          message is fed and streamed as generated tokens
        - MoE stats window advances per processed token
  → SSE events (delta content / delta reasoning_content / final usage)
```

The engine runs on a worker thread behind a lock (MLX decode serializes on
the GPU stream); further requests queue. Slot bookkeeping mirrors llama.cpp's
`/slots`.

## Cache model

* **Full-attention layers** (every 4th layer in Qwen3.6-35B-A3B): `KVCache`
  (step-256 growth) with optional `--cache-type-k/v` f32/f16/q8_0/q4_0.
  KV is small by design here (2 KV heads × 256 head dim × 10 layers).
* **Linear-attention layers** (gated delta net): `ArraysCache` holding the
  short-conv state and the recurrent state. This state is a *running
  summary*: it is only valid for exact continuations, therefore prefix reuse
  requires the cached sequence to be a strict prefix of the new prompt —
  otherwise the engine re-prefills. (This matches chat behavior: the next
  turn's template continues exactly where the previous generation stopped.)
* Context: `-c/--ctx-size` is the total, split across `--parallel` slots
  (llama.cpp semantics); requests that would exceed it fail fast with a
  clear error.

## MoE expansion invariants

1. **Stock path untouched**: with the feature off (or a layer outside the
   configured range), `moe_block.routed_forward` executes the upstream
   mlx-lm computation verbatim (`argpartition` top-K → take_along_axis →
   norm). Flags-off output is bit-identical to stock.
2. **`N == K` + threshold 0 + `auto` renorm** also routes stock (no-op
   expansion is never instantiated as a graph).
3. The post-pass is a pure-function port of `build_moe_expansion_weights`
   (`vagrillo/llama.cpp`, `src/llama-moe-expansion.h`), pinned by golden
   tests (`tests/test_moe_expansion.py`).
4. Router weights, softmax, shared expert/gate and output scales are never
   modified; dropped ranks carry zero weight inside the fixed-N slot set.
5. Compute scales with N (fixed-N mask), not with the adaptive kept count —
   documented as a routing change, not a speedup (same trade-off as the
   llama.cpp implementation; ds4-style true variable-count execution is a
   possible future optimization for the MLX graph).

## Loading pipeline (`loader.py`)

config.json → registry (`models/__init__.py`) → `ModelArgs.from_dict` →
`Model(args, expansion=runtime)` → **sanitize** (vision/MTP weights dropped;
`model.language_model.*` remap; per-expert HF layout stacked into
`switch_mlp.*`, or merged `experts.gate_up_proj` split — both layouts
supported; conv1d axis fix and RMS-norm +1 fix for unconverted checkpoints)
→ `nn.quantize` with the checkpoint's per-layer quantization overrides
(mlx-community quants load quantized, the router gate at 8-bit as upstream)
→ `model.load_weights(...)` → `mx.eval`. Expansion parameters are validated
against the model geometry **before** any weight is read (fail fast).

## Vendored code (provenance)

The engine pieces under `src/agrimlxo/models/` are vendored from
[mlx-lm](https://github.com/ml-explore/mlx-lm) @ `0d4d3d04` (Apache-2.0,
© Apple Inc.), trimmed/adapted for a single-device text server; each file
carries a provenance header. `omlx` served as the reference for the
server/engine approach (OpenAI API surface, SSE streaming, incremental
detokenization) but no omlx code is copied. The MoE expansion algorithm is
ported from the `moe-expansion` branch of `vagrillo/llama.cpp`
(`src/llama-moe-expansion.h`, AgrillaMoE), per the Zenodo paper cited in the
README and `docs/moe-expansion.md`.

## What is intentionally out of scope

* Vision tower / MTP weights (text-only server; both are filtered at load).
* Multi-node pipeline/distributed sharding (single Mac target).
* Continuous batching across requests (single-stream decode; slots queue).
* GUI beyond the embedded chat page (per instructions: server-only).
