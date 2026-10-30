# AgrimlxO

```
    _                    _      _  __  __  _   _ ___
   / \   __ _  ___ _ __ | |    / \|  \/  |/ / | |_ _|
  / _ \ / _` |/ _ \ '_ \| |   / _ \ |\/| | || | || |
 / ___ \ (_| |  __/ | | | |  / ___ \ |  | | || | || |
/_/   \_\__, |\___|_| |_|_| /_/   \_\_| |_| \__\_|
        |___/   dedicated Qwen3.x-35B MoE server
                MLX inference + MoE expansion
```

**AgrimlxO** is a dedicated, GUI-free inference **server** for the
**Qwen3.x-35B MoE** family (reference model: **Qwen3.6-35B-A3B**, 40 hybrid
layers, 256 routed experts, native top-8), optimized for **Apple Silicon**
via Apple's [MLX](https://github.com/ml-explore/mlx). It brings together:

* the **inference engine approach of [omlx](https://github.com/pmarreck/omlx)**
  (MLX-native, OpenAI-compatible server), rewritten as a focused
  single-model, standalone engine, and
* the **MoE expert expansion** feature of
  [AgrillaMoE](https://github.com/vagrillo/AgrimlxO) / the llama.cpp fork
  `vagrillo/llama.cpp` (branch `moe-expansion`), ported to the MLX graph.

The MoE expansion is a runtime-only routing modification described in

> Vincenzo Agrillo, *Succinct convergence of mixture-of-experts language
> models by layer-scoped expert-budget expansion*, Zenodo, 2026.
> <https://zenodo.org/records/22255483>

No model-file changes, no MLX changes: **with the feature off the engine
behaves bit-identically to stock**.

The project is fully **standalone** — it has no dependency on the AgrillaMoE
or omlx repositories (the model implementations are vendored from mlx-lm,
Apache-2.0, and adapted).

---

## Why

On Qwen3.6-35B-A3B (the paper's reference model), raising the routed-expert
budget above the native top-8 with decaying influence on the extra ranks
measured on GPQA-Diamond (Q8_0, paired runs): **84.34% expanded vs 81.82%
native** (+2.52 pts), overtaking the dense Qwen3.8-27B reference — see the
paper and `docs/moe-expansion.md`. Gains are model-dependent: always measure
per model.

## Quick start (macOS)

> 🧪 Manual Mac test plan (setup, per-parameter checks, kernels, mixed quant,
> paired quality protocol): **[docs/mac-testing.md](docs/mac-testing.md)**

```bash
# 1) install
pip install .

# 2) download a model (mlx-community format, 4/8-bit recommended for 128 GB,
#    bfloat16 needs ~64+ GB of unified memory)
agrimlxo download mlx-community/Qwen3.6-35B-A3B-8bit

# 3) serve: the AgrillaMoE default expansion profile (experts 12, T=0.75,
#    layers 25-39) is injected automatically for Qwen3.6-35B-A3B
agrimlxo serve -m mlx-community/Qwen3.6-35B-A3B-8bit

#    ...or keep everything stock:
agrimlxo serve -m mlx-community/Qwen3.6-35B-A3B-8bit --no-moe-expansion
```

Open **http://127.0.0.1:8080** for the built-in chat interface (llama.cpp
webui style) or point any OpenAI client at it:

```bash
curl http://127.0.0.1:8080/v1/chat/completions -H 'Content-Type: application/json' -d '{
  "messages": [{"role":"user","content":"Explain mixture-of-experts routing."}],
  "stream": true, "temperature": 0.6, "max_tokens": 2048
}'
```

Not on macOS? MLX also ships CPU/Linux wheels: `pip install "mlx[cpu]"` —
useful for smoke tests; the production target is Apple Silicon (Metal).

## The MoE expansion in one paragraph

Three knobs (all off → stock):

1. **`--moe-experts N`** (or `--moe-experts-add N`): raise the routed-expert
   budget above the model's native top-K (8 → 20 on the reference model).
2. **`--moe-expert-threshold T`**: adaptive per-token expert count — keep
   experts while their router probability stays above `T × p(rank N/2)`
   (dense tokens route to ~N experts, peaked tokens to fewer; floor
   `max(1, N/4)`).
3. **`--moe-expert-decay-end D`**: the extra ranks get a linearly decaying
   influence factor from 0.99 down to D, applied to the raw probability
   before renormalization — the native experts keep the output scale, the
   extras fade out.

Plus `--moe-expert-renorm auto|always|never`, `--moe-no-expert-decay`,
`--moe-expert-layer-start/end` (the *layer-scoped* part: the budget expansion
matters most in late layers) and `--moe-experts` ds4 aliases
(`--q35-experts`, …). Observability: a startup banner and per-window
`experts/token` stats (`--moe-stats-every`, `MOE_EXPERT_STATS_EVERY`).
Full documentation, the algorithm and the benchmark protocol:
**[docs/moe-expansion.md](docs/moe-expansion.md)**.

## Server options

llama.cpp-server-style surface: `-m/--model`, `--host/--port`, `--api-key`,
`-c/--ctx-size`, `--parallel` (slots; the context is split across them),
`-b/--batch-size` (prefill chunk), `-n/--n-predict`, `--cache-type-k/v`
(f16/f32/q8_0/q4_0), `--temp/--top-k/--top-p/--min-p/--xtc-*`,
`--repeat-penalty/--presence-penalty/--frequency-penalty`, `--seed`,
`--reasoning-budget N` (AgrillaMoE: force a wrap-up message when the
`<think>` phase exceeds the budget), `--reasoning-format auto|none`
(split `<think>` into `reasoning_content`), `--chat-template`,
`--no-moe-expansion` (+ `AGRIMLXO_NO_MOE_EXPANSION=1`) and the whole
`--moe-*` family above. `agrimlxo download <repo>` and `agrimlxo list`
manage local models; without `-m` the server offers the locally found ones.

## Endpoints

| endpoint | description |
|---|---|
| `GET /` | chat interface (llama.cpp webui style) |
| `POST /v1/chat/completions` | OpenAI-compatible chat, streaming via SSE, `reasoning_content` split |
| `POST /v1/completions` | OpenAI-compatible completions |
| `GET /v1/models` | the served model |
| `GET /health` | health check |
| `GET /props` | llama.cpp-style server properties (incl. MoE expansion state) |
| `GET /slots` | slot occupancy |
| `GET /metrics` | minimal Prometheus metrics (incl. MoE expansion info) |

## Architecture (short version)

```
src/agrimlxo/
├── cli.py / args.py          flags, AgrillaMoE default-profile injection, banner
├── server.py                 FastAPI/uvicorn, OpenAI API + SSE streaming
├── engine.py                 chunked prefill, KV-cache prefix reuse, stop
│                             handling, reasoning budget, slot state
├── sampler.py                temp/top-k/top-p/min-p/XTC, penalties, seed
├── tokenizer.py              AutoTokenizer wrapper + incremental detokenizer
├── loader.py                 mlx-lm-style load (sanitize → quantize → weights)
├── moe_expansion.py          ★ the expansion: post-pass, validation, stats,
│                             AgrillaMoE default profiles
├── models/
│   ├── qwen3_5_moe.py        ★ Qwen3.6-35B-A3B (hybrid gated-delta linear
│   │                           attention + full attention, MoE + shared expert)
│   ├── qwen3_moe.py          classic dense-attention Qwen3 MoE family
│   ├── moe_block.py          ★ routed forward: stock path vs rank-ordered
│   │                           top-N fixed-mask expansion path
│   └── (cache/base/gated_delta/…)   vendored mlx-lm engine pieces
└── webui/index.html          chat interface
```

Details, invariants and vendoring notes:
**[docs/architecture.md](docs/architecture.md)**.

## Optional native kernels (Metal)

The fused MoE routed weighted-sum kernel from omlx is vendored under
`native/` and **generalized for the expansion**: the upstream kernel only
accepted 6/8/10 routed experts per token, AgrimlxO's accepts any budget in
`[2, 64]` (tiled kernels for 6/8/10/12/16/20/24/32, a runtime-topk dynamic
kernel for the rest). Off by default — the fused sum reorders the fp
accumulation vs the stock ops:

```bash
AGRIMLXO_WITH_KERNELS=1 pip install ./native   # build on the Mac
agrimlxo serve -m <model> --fast-moe-prefill   # enable (or AGRIMLXO_FAST_MOE=1)
```

Missing kernels / ABI mismatch degrade to the stock MLX ops with one log
line. Docs: **[docs/kernels.md](docs/kernels.md)**.

## Per-expert mixed quantization (MixedQuant)

Follow AgrillaMoE's expert-rank workflow on MLX checkpoints: rank the
experts per layer (`--moe-expert-log` + `moe_predict.py`), then cut the
cold tail to fewer bits while the hot experts keep 8:

```bash
agrimlxo mixed-quantize <model> --order expert-order.txt \
    --top-m 64 --hi-bits 8 --lo-bits 4 --out model-mixed
agrimlxo serve -m model-mixed --moe-experts 20 --moe-expert-threshold 0.8
```

`MixedSwitchGLU` loads the annotated layers natively and composes with the
expansion. Memory-first trade-off (G× routed compute on mixed layers):
**[docs/mixed-experts.md](docs/mixed-experts.md)**.

## Tests

```bash
pip install -e ".[dev]"
pytest tests/
```

The suite (61 tests, runs on the CPU backend too) pins the expansion
post-pass against hand-derived golden values from the llama.cpp
implementation (threshold cut, floor, decay endpoints, midpoint factor,
renorm guards, per-token independence, 32-configuration sweep), checks
flags-off == stock on a toy model end-to-end, KV-prefix-reuse equivalence,
stop/context handling, the HTTP API, the expert log, the fast-path
fallback and the mixed-quant roundtrip.

## Status & compatibility

* **Reference model**: Qwen3.6-35B-A3B (`qwen3_5_moe`; MLX-format
  checkpoints, e.g. `mlx-community/Qwen3.6-35B-A3B-{4,8}bit`). The classic
  `qwen3_moe` family (e.g. Qwen3-30B-A3B) is also supported.
* Vision tower and MTP weights of the checkpoint are ignored (text-only
  server). Draft/MTP decoding is not implemented.
* Generation is single-stream (MLX serializes decode on the GPU stream);
  multiple slots share the context and queue requests.
* **Execution/compilation tests on a Mac are done manually by the
  maintainer** (per project instructions).

## License

Apache-2.0 — like both parents. The vendored mlx-lm modules keep their
Apple Inc. copyright headers (`docs/architecture.md` lists the provenance).
