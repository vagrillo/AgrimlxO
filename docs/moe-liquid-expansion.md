# Liquid MoE Expansion — AgrimlxO

Perplexity-driven **dynamic** expert budget: instead of a fixed expanded
budget N (the static expansion, `docs/moe-expansion.md`), each MoE layer
grows its expert count **per token**, re-evaluating a router "perplexity"
index after every added expert, up to a per-band maximum. This implements
the variation described in `liquidmoeexpansion.txt` (repo root). Like the
static expansion it is a runtime-only routing modification: no model-file
changes, no MLX changes, and with the feature off the engine behaves
bit-identically to stock.

**Reference model**: Qwen3.6-35B-A3B — 40 layers, 256 routed experts,
native top-8, `norm_topk_prob=true`. Layer indices are 0-based throughout.

## The perplexity index

The router produces, for every layer and every token, a softmax
distribution over the 256 experts. The index condenses it to a value in
`[0, 1]`, `0` = "the top-8 experts are clearly the best ones vs the other
248", higher = more indecision (e.g. the scores of ranks 9..12 close to
rank 8). The default index:

```
tail_rel(n) = clip( tail_beyond(n) / p(rank n), 0, 1 )
```

where `tail_beyond(n)` is the router probability mass past the current cut
and `p(rank n)` the score of the last kept expert. It directly measures
the notes' indecision case — "the scores of experts ranked 9th to 12th
are not far from the 8th" — and it re-evaluates naturally after each
added expert (the cut moves, the tail shrinks).

Alternatives (`--moe-liquid-index`), all in `[0, 1]`:

| name | formula (budget n) | note |
|---|---|---|
| `tail_rel` (default) | `clip(tail(n) / p(rank n), 0, 1)` | scale-free indecision |
| `tail` | `tail(n)` | raw escaped mass |
| `entropy` | `H(full router) / ln(E)` | cut-independent: degenerates to all-or-nothing |
| `kept_entropy` | `H(renorm top-n) / ln(n)` | flatness of the kept set |
| `eff` | `clip((exp(H) − n) / (E − n), 0, 1)` | normalized effective expert count |

Phase 2 of the protocol decides which one to use (see below) — the stats
tables of `moe-analyze` report all of them on the captured runs.

## The ladder

Starting from the native budget K (=8), per token and per layer:

```
pass(n) = idx(n) > T          # the check that decides adding rank n+1
kept    = K + leading-run-length(pass)   # stop at the first failed check
```

with `T = T0` for the first check and `T'` for the subsequent ones (the
notes: *"se è superiore allora aggiungi un esperto (il 9°) e ricalcola la
perplexity se ancora superiore a 0.7 aggiungine un altro, ricalcola e così
via .. fino al topE"*). Fully vectorized on the graph (no per-token
sync); like the static expansion, compute scales with the band budget,
not with the per-token kept count (fixed-N mask: the graph carries topE
slots, dropped ranks get zero weight).

The extra ranks (≥ K) get the same linear influence decay `0.99..D` as
the static expansion, computed against the band's topE so a rank's factor
does not jump when the adaptive cut moves. Kept-weight renormalization
follows `--moe-expert-renorm` (`auto` = the model's stock behavior).

## The static band rules (starting point)

| layers (0-based) | first check T0 | later checks T' | max experts topE |
|---|---|---|---|
| 2–13 | 0.80 | 0.70 | 10 |
| 14–27 | 0.60 | 0.60 | 12 |
| 28–39 | 0.50 | 0.50 | 20 |

Layers 0–1 are not covered by any band → stock routing (top-8). The
thresholds are the starting values of the experiment notes; the protocol's
phase 2 recalibrates them from captured data. Custom rules go in a JSON
file:

```json
{
  "index": "tail_rel",
  "bands": [
    {"start": 2,  "end": 13,  "perp_t": 0.80, "perp_next": 0.70, "top_e": 10},
    {"start": 14, "end": 27,  "perp_t": 0.60, "perp_next": 0.60, "top_e": 12},
    {"start": 28, "end": 39,  "perp_t": 0.50, "perp_next": 0.50, "top_e": 20}
  ]
}
```

(`"layers": [start, end]` is accepted as an alias for `start`/`end`;
bands must not overlap; `top_e` ∈ [K, expert_count]; thresholds ∈ [0, 1].
A band with `top_e == K` is inert — that layer stays stock.)

## The protocol (three phases)

### Phase 1 — collect router distributions

Run generation with **stock routing** and capture the full distribution
(all 256 scores × 40 layers × every processed position) per run:

```bash
agrimlxo capture -m mlx-community/Qwen3.6-35B-A3B-8bit \
    --temp 0.7 --max-tokens 256 --seed 1234 --out ./routercapture
```

* default prompt set: 8 diverse prompts; `--prompts FILE` for your own
  (one per line); `--runs N` repeats.
* one `.npz` per run in `--out`: `scores` `[positions, layers, experts]`
  (float16), `n_prompt` (prefill rows), generated `tokens`, `prompt_tokens`
  and a JSON `meta` (prompt, text, temp, seed, mode).
* the warmup generation is NOT captured (no run is open); the all-time
  expert-usage file is redirected under `--out` so the server's
  `./expertusage` stays untouched.
* sampling params are fixed (`--seed` base, run i uses seed+i) so phase 3
  can replay the identical runs.

### Phase 2 — pick the index, calibrate thresholds

```bash
agrimlxo moe-analyze ./routercapture/*.npz --calibrate 0.9 --rules-out rules.json
```

prints, for every candidate index, the per-layer mean/median/p90 of the
first check (budget = native top-K) over the generated tokens of all the
runs — the statistical basis for choosing the index and its thresholds.
`--calibrate P` turns the notes' bands into rules with each band's
thresholds set to the P-quantile of the default index over that band's
layers/tokens; `--rules-out` writes the JSON skeleton.

### Phase 3 — liquid run and comparison

```bash
agrimlxo capture -m mlx-community/Qwen3.6-35B-A3B-8bit \
    --temp 0.7 --max-tokens 256 --seed 1234 --out ./routercapture-liq \
    --moe-liquid --moe-liquid-rules rules.json

agrimlxo moe-compare ./routercapture/run000.npz ./routercapture-liq/run000.npz
```

`moe-compare` prints the per-layer perplexity table (base vs liquid,
mean/p90/Δ with the run's index) and the output agreement (common token
prefix, token agreement %); `--show-text` prints both generated texts.

## Serving with the liquid expansion

```bash
agrimlxo serve -m mlx-community/Qwen3.6-35B-A3B-8bit --moe-liquid
# custom rules/index:
agrimlxo serve -m ... --moe-liquid --moe-liquid-rules rules.json --moe-liquid-index tail
```

`--moe-liquid` is mutually exclusive with `--moe-experts`,
`--moe-experts-add`, `--moe-expert-threshold` and `--moe-reduce`
(fail fast). The decay/renorm knobs of the static expansion
(`--moe-expert-decay-end`, `--moe-no-expert-decay`, `--moe-expert-renorm`)
apply to the liquid path too. The per-layer experts/token stats line
(`--moe-stats-every`) and the kept-expert log (`--moe-expert-log`) work
unchanged (the kept count is now the ladder's per-token decision).

## Caveats

* Looping (re-running a layer) is explicitly **out of scope** for this
  first phase — the ladder only grows the expert budget.
* Compute scales with the band topE (fixed-N mask), not with the average
  kept count: the 28–39 band carries 20 slots/token (2.5× the routed
  compute of stock) even when most tokens keep 8–9 experts.
* The index/threshold calibration is model-dependent: always re-run
  phases 1–2 per model before trusting the numbers.

## Tests

`tests/test_moe_liquid.py`: band rules and validation, index bounds and
the notes' indecision case, golden ladder cases (hand-derived), the
graph/numpy index twin cross-check, capture round-trip, toy-model stock
equivalence outside the bands (bit-identical) and deterministic liquid
expansion.
