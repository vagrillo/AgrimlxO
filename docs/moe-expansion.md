# MoE Expert Expansion — AgrimlxO

Runtime-only routing modification for sparse MoE models: raise the
routed-expert budget above the model's native top-K with a linear influence
decay on the extra ranks, optionally with an adaptive (threshold-based)
expert count and a layer range. This is the MLX port of the
"layer-scoped expert-budget expansion" feature of AgrillaMoE / the llama.cpp
fork `vagrillo/llama.cpp` (branch `moe-expansion`), described in

> Vincenzo Agrillo, *Succinct convergence of mixture-of-experts language
> models by layer-scoped expert-budget expansion* (Zenodo, 2026)
> <https://zenodo.org/records/22255483>

No model-file changes, no MLX changes, no rewritten expert kernels: with the
feature off, the engine behaves bit-identically to stock. The routed branch
keeps the stock computation (softmax router → top-K → renorm) untouched;
when the expansion applies, the layer runs the rank-ordered fixed-N mask
path described below.

**Reference model**: Qwen3.6-35B-A3B (`qwen3_5_moe`: 40 hybrid
linear/full-attention layers, 256 routed experts, native top-8,
`norm_topk_prob=true`, shared expert per layer). Reference results in the
paper: GPQA-Diamond 84.34% (expansion, N=16/20 recipes) vs 81.82% (native
top-8) on Q8_0 — paired net +5, with the gain concentrated in Chemistry.
Gains are model-dependent: always measure per model.

## The three knobs

1. **Max experts N** (`--moe-experts N`, or `--moe-experts-add N` to add on
   top of the native K): raise the routed expert budget above the model's
   native top-K (e.g. 8 → 20). Values below the native K are also allowed
   (fixed pruning).
2. **Dynamic threshold T** (`--moe-expert-threshold T`): instead of a fixed
   count, keep experts while their router probability stays above
   `T × p_ref`, where `p_ref` is the probability of rank `N/2` (1-indexed,
   i.e. 0-based index `N/2 − 1`). The per-token expert count becomes
   adaptive: dense tokens route to ~N experts, peaked tokens to fewer.
   `T ≤ 1` only ever expands past the native K; `T > 1` can also prune
   below it, down to the floor `max(1, N/4)`.
3. **Expansion decay D** (`--moe-expert-decay-end D`, default 0.50): when
   N > K, ranks above the native K get a linearly decaying influence factor
   from 0.99 down to D (a single extra rank gets the midpoint
   `(0.99 + D)/2` = 0.745 with the default D), applied to the raw
   probability *before* renormalization, so the native experts keep the
   output scale and the extras fade out. `--moe-no-expert-decay` disables
   the decay (extra experts at full influence).

All flags off → bit-identical to stock (hard requirement, enforced by
construction: the stock routing code path is taken verbatim).
`N == K` with no threshold and `auto` renorm is also exactly stock.

## CLI flags

| flag | alias (ds4) | default | meaning |
|---|---|---|---|
| `--moe-experts N` | `--q35-experts` | 0 (off) | absolute max routed experts per token N (2 ≤ N ≤ expert_count) |
| `--moe-experts-add N` | — | 0 (off) | experts added on top of the model's native top-K |
| `--moe-expert-threshold T` | `--q35-expert-threshold` | 0 (off) | adaptive count: keep while p ≥ T × p(rank N/2); T ∈ (0, 10] |
| `--moe-expert-decay-end D` | — | 0.50 | influence of the last extra rank; linear 0.99..D; D ∈ (0, 0.99) |
| `--moe-no-expert-decay` | `--q35-no-expert-decay` | off | extra experts at full influence |
| `--moe-expert-renorm MODE` | — | `auto` | kept-weight renormalization after cut+decay: `auto` follows the model's stock normalization (renorm iff `norm_topk_prob`), `always` forces sum-to-1, `never` keeps the raw decayed score scale |
| `--moe-expert-layer-start I` | — | 0 | first layer the expansion applies to; `< 1`: fraction of n_layer, `≥ 1`: layer index |
| `--moe-expert-layer-end I` | — | −1 | last layer (inclusive); `< 0`: last layer, `< 1`: fraction, `≥ 1`: index |
| `--moe-stats-every N` | — | 128 | print the experts/token stats every N processed tokens (0 = use `MOE_EXPERT_STATS_EVERY`) |
| `--moe-expert-log FILE` | — | off | JSONL log of kept expert ids per token/layer (`moe_predict.py`-compatible) |
| `--no-moe-expansion` | `--agrilla-no-moe-expansion` | off | disable the expansion entirely (also `AGRIMLXO_NO_MOE_EXPANSION=1`) |
| `--moe-reduce X` | — | 0 (off) | the inverse of the expansion: keep only experts with weight ≥ X × weight(top-1) |
| `--moe-reduce-layers S-E` | — | `0--1` | layer range for `--moe-reduce` (inclusive, 0-based, −1 = last) |

`--moe-reduce` and the expansion flags are mutually exclusive (fail fast).

Invalid parameters (N out of range, non-MoE model, threshold outside
(0, 10], decay end outside (0, 0.99), layer start > end, reduce+expansion
together) fail fast at model load with a clear message and a non-zero exit
code.

## Default profiles (AgrillaMoE behavior)

The default expansion profile is injected **only if the user passes no
`--moe-*` / `--q35-*` flag**; `--no-moe-expansion` (or
`AGRIMLXO_NO_MOE_EXPANSION=1`) disables it entirely. Profiles are
signature-matched on the model config (layers / experts / top-K), like the
AgrillaMoE launcher does on GGUF metadata:

| model | signature | profile injected |
|---|---|---|
| Qwen3.6-35B-A3B | 40 layers, 256 experts, top-8 | experts 20, thr 0.80, layers 25–39 |
| other MoE | any | experts 20, thr 0.80 (no layer range) |

The Qwen3.6 profile is exactly the one benchmarked in the paper (GPQA-Diamond
84.34% vs 81.82% stock).

## Mechanics (MLX implementation notes)

The routed branch of a sparse MoE layer normally runs the stock path
(softmax over 256 router logits → `argpartition` top-K → optional weight
norm → `SwitchGLU` over the selected experts). The expansion replaces the
selection step:

1. **Rank-ordered selection (fixed-N mask)**: take the top-N router
   probabilities per token in rank order (`argsort` descending; ties follow
   argsort, per the llama.cpp note). The graph always carries N slots per
   token and dropped ranks simply get zero weight, so the expert kernels
   stay stock. Consequence: compute scales with N (not with the adaptive
   count), i.e. speed ~N/K of native — this is a routing change, not a
   speedup.
2. **Post-pass** (pure MX ops, `agrimlxo/moe_expansion.py::
   expansion_post_pass`, a faithful port of `build_moe_expansion_weights` in
   `src/llama-moe-expansion.h` of the fork):
   * selection: `keep = min(step(F − j) + step(w > T · p_ref), 1)` with the
     floor `F = max(1, N/4)` and `p_ref = w[rank N/2]` (the kept set is a
     contiguous prefix because the weights are non-increasing);
   * decay: ranks ≥ K get `0.99 + t·(D − 0.99)` with
     `t = clamp((j − K)/(N − K − 1), 0, 1)`, midpoint with a single extra
     rank; computed against the maximum N, so a rank's factor does not jump
     when the adaptive cut moves;
   * renormalization of the kept weights to sum 1 (with the 2⁻¹⁴ sum guard),
     controlled by `--moe-expert-renorm` (`auto` = follow
     `norm_topk_prob`).
3. **Untouched**: router weights, softmax, shared expert and its gate, the
   output scales. The warmup graph sizes buffers for N ≤ expert_count.
   Attention, KV cache and all other layers are stock.

Layer scoping is decided per layer call: outside `[layer_start, layer_end]`
(0-based, inclusive) the layer runs the stock path — that is what makes the
feature "layer-scoped" (per the paper, the budget expansion matters most in
late layers).

## Observability

With the feature active, every `--moe-stats-every` processed tokens (prompt
+ generated; default 128, `0` = off) the engine prints to stderr the average
number of routed experts per token, per expanded layer:

```
moe: routed experts per token: 20 (model default 8; ranks 9..20 get linear influence decay 0.99..0.5)
moe: expert threshold: keep experts while p >= 0.8 x p(rank 10) (experts 5..20 per token)
moe: expansion active on layers 25..39 of 40
...
moe: experts/token avg: L25:16.1 L26:15.4 ... | mean 15.5
```

This is the primary health signal: with `N=20, T=0.8` on Qwen3.6-35B-A3B
expect a mean around 15–16; exactly 20.0 with `T=0`; exactly `N/4` with
`T=10`.

The startup banner (`/props` exposes the same state for the webui and
`/metrics` for monitoring) reports the active configuration; with
`--moe-expert-log FILE` the kept expert ids are logged per token/layer in
the AgrillaMoE `LLAMA_MOE_EXPERT_LOG` format (consumable by the fork's
`moe_predict.py` analysis).

## Examples

```bash
# quality-leaning expansion, as measured in the paper (Qwen3.6-35B-A3B):
agrimlxo serve -m mlx-community/Qwen3.6-35B-A3B-8bit --temp 0 \
    --moe-experts 20 --moe-expert-threshold 0.8          # ~15.5 experts/token

# same, ds4-compatible spelling:
agrimlxo serve -m model --q35-experts 20 --q35-expert-threshold 0.8

# expansion restricted to the late layers (layer-scoped), decay fading to 0.30:
agrimlxo serve -m model --moe-experts-add 12 --moe-expert-decay-end 0.30 \
    --moe-expert-layer-start 0.5

# aggressive pruning below the native top-K (c ∈ [2..8] on an 8-expert model):
agrimlxo serve -m model --moe-experts 8 --moe-expert-threshold 2

# the inverse (dynamic pruning, AgrillaMoE's --moe-reduce):
agrimlxo serve -m model --moe-reduce 0.5 --moe-reduce-layers 25-39

# tune/verify with the stats line:
MOE_EXPERT_STATS_EVERY=64 agrimlxo serve -m model --moe-experts 20 ...
```

## Model coverage

| implementation | model_type | family |
|---|---|---|
| `models/qwen3_5_moe.py` | `qwen3_5_moe` | Qwen3.5/3.6 MoE — **Qwen3.6-35B-A3B** (hybrid gated-delta linear attention + gated full attention with partial RoPE, sparse MoE + shared expert) |
| `models/qwen3_moe.py` | `qwen3_moe` | classic dense-attention Qwen3 MoE (Qwen3-30B-A3B and friends) |

Anything else is rejected at load with a clear message — AgrimlxO is
dedicated to the Qwen3.x MoE family.

## Verification

* `tests/test_moe_expansion.py` (23 checks): the post-pass matches the
  normative semantics on hand-derived golden cases — T=0 fixed N (decay
  values, renorm sum-to-1), T=0.8 power-law cut, T=10 → floor, flat
  probabilities → N, single extra → 0.745, D=0.30 endpoints, no-decay,
  strict step semantics at the threshold, pruning below K, per-token
  independence, 32-configuration sweep (kept weights sum to 1, kept set is
  a contiguous prefix), renorm-off raw scale, zero-guard, `--moe-reduce`,
  layer-range resolution, N==K→stock, `--moe-experts-add`, fail-fast
  validation, banner format, stats window, default profiles.
* `tests/test_engine_toy.py` (11 checks) on a real toy `qwen3_moe`
  checkpoint: flags-off determinism, expansion ≠ stock at the tensor level,
  T=0 keeps exactly N (stats), N==K+auto == stock, KV-prefix-reuse
  equivalence with a fresh prefill, stop strings/max tokens/context
  overflow/eos, expert log, ThinkSplit streaming, hybrid `qwen3_5_moe`
  forward (linear+full attention layers in the right positions).
* `tests/test_server.py` (7 checks): /health, /props (MoE state),
  /v1/models, /metrics, chat non-stream, chat SSE stream with usage,
  completions, slots.
* `tests/test_args.py` (8 checks): flag parsing, ds4 aliases, default
  profile injection rules, disable-by-env, profiles.
* When benchmarking quality, keep the paired protocol (same questions,
  greedy, one variable at a time) described in the Zenodo paper.

## macOS build/run notes

Nothing to compile: AgrimlxO is pure Python on top of the MLX wheels
(`pip install mlx` on macOS ships Metal). Metal kernels for the gated delta
net are provided by the vendored `models/gated_delta.py` (pure-MLX fallback
elsewhere). If a segfault happens with a working install, capture a
backtrace:

```bash
lldb -- agrimlxo serve -m <model> --moe-experts 20 --moe-expert-threshold 0.8 ...
(lldb) run
(lldb) bt
```

and open an issue with the backtrace plus the `moe:` startup banner lines.
