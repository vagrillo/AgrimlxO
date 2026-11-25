# SPDX-License-Identifier: Apache-2.0
"""Liquid MoE expansion — perplexity-driven dynamic expert budget.

Second variation of the routing modification described in
``liquidmoeexpansion.txt`` (repo root): instead of a fixed expanded budget
N, each MoE layer grows its expert count per token while a router
"perplexity" index says the top-8 (native) selection is undecided, with
per-band thresholds and budgets:

  * layers 2..13   — first check T0=0.80, later checks 0.70, max topE=10
  * layers 14..27  — threshold 0.60, max topE=12
  * layers 28..39  — threshold 0.50, max topE=20

(0-based layer indices; layers 0-1 stay stock. The thresholds are the
starting point from the experiment notes — phase 2 of the protocol
calibrates them from captured router distributions.)

The perplexity index is in [0, 1] with 0 = "the selected experts are
clearly the best ones, no indecision". The default index ``tail_rel``
measures exactly the indecision case of the notes — scores of the ranks
just past the cut comparable to the last kept rank:

    idx(n) = clip( tail_beyond(n) / p(rank n), 0, 1 )

The ladder semantics ("aggiungi un esperto e ricalcola la perplexity se
ancora superiore aggiungine un altro, e cosi' via, fino al topE") is the
leading run of passes over the per-budget checks:

    pass(n) = idx(n) > T            (T = T0 for the first check, T' later)
    kept    = K + sum( pass(K) & ... & pass(n) )   up to topE

which is fully vectorizable on the graph (no per-token sync). Like the
static expansion, the routing graph carries the band's topE selection
slots per token (fixed-N mask) and dropped ranks get zero weight, so the
downstream expert kernels stay stock; extra ranks get the same linear
influence decay 0.99..D as the static expansion, computed against topE so
a rank's factor does not jump when the adaptive cut moves.

The module also hosts the router-distribution capture used by phase 1 of
the protocol (``agrimlxo capture``): with a capture sink installed, every
MoE call appends the FULL post-softmax router distribution (all E scores,
all layers, every processed position) to an in-memory buffer that the
engine drains at its sync points and that is saved as one ``.npz`` per
run. ``moe-analyze`` / ``moe-compare`` then compute the candidate indices
off-line (numpy twin of the graph formulas, cross-checked in tests).
"""

from __future__ import annotations

import json
import math
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import mlx.core as mx
import numpy as np

from .moe_expansion import _RENORM_MIN

# near-zero guard for divisions/log on router probabilities
_EPS = 1e-12

INDEX_NAMES = ("tail_rel", "tail", "entropy", "kept_entropy", "eff")
DEFAULT_INDEX = "tail_rel"


# -- band rules ----------------------------------------------------------------


@dataclass
class LiquidBand:
    """One layer band of the liquid expansion rules (inclusive ranges)."""

    start: int  # first layer, 0-based, inclusive
    end: int  # last layer, 0-based, inclusive
    perp_t: float  # perplexity threshold of the FIRST check (budget = K)
    perp_next: float  # threshold of the subsequent checks (default: perp_t)
    top_e: int  # max routed experts for the band

    def covers(self, layer_idx: int) -> bool:
        return self.start <= layer_idx <= self.end

    def to_dict(self) -> Dict:
        return {
            "start": self.start,
            "end": self.end,
            "perp_t": self.perp_t,
            "perp_next": self.perp_next,
            "top_e": self.top_e,
        }


# The static rules of the experiment notes (liquidmoeexpansion.txt), on the
# 0-based layer indices of the reference model (Qwen3.6-35B-A3B, 40 layers):
# "per il layer da 2 a 13 se la perplexity e' inferiore a 0.8 procedi con i
#  router top 8 / se e' superiore allora aggiungi un esperto (il 9o) e
#  ricalcola la perplexity se ancora superiore a 0.7 aggiungine un altro ...
#  fino al topE (10) / per il layer da 14 a 27 il threshold diventa 0.6 e il
#  topE diventa 12 / per i restanti il PerpT diventa 0.5 e il topE diventa 20".
# Layers 0-1 are not covered by any band -> stock routing.
DEFAULT_BANDS: List[LiquidBand] = [
    LiquidBand(2, 13, 0.80, 0.70, 10),
    LiquidBand(14, 27, 0.60, 0.60, 12),
    LiquidBand(28, 39, 0.50, 0.50, 20),
]


@dataclass
class LiquidRules:
    """Liquid expansion rules: the index definition plus the layer bands."""

    bands: List[LiquidBand] = field(default_factory=lambda: list(DEFAULT_BANDS))
    index: str = DEFAULT_INDEX

    # -- construction --------------------------------------------------------

    @classmethod
    def from_dict(cls, data: Dict) -> "LiquidRules":
        if not isinstance(data, dict):
            raise ValueError("liquid rules: expected a JSON object")
        bands = []
        for i, b in enumerate(data.get("bands", [])):
            if isinstance(b.get("layers"), (list, tuple)) and len(b["layers"]) == 2:
                b = {**b, "start": b["layers"][0], "end": b["layers"][1]}
            try:
                bands.append(
                    LiquidBand(
                        start=int(b["start"]),
                        end=int(b["end"]),
                        perp_t=float(b.get("perp_t", b.get("threshold", 0.8))),
                        perp_next=float(b.get("perp_next", b.get("perp_t", b.get("threshold", 0.8)))),
                        top_e=int(b["top_e"]),
                    )
                )
            except (KeyError, TypeError, ValueError) as e:
                raise ValueError(f"liquid rules: bad band {i}: {e}") from e
        index = str(data.get("index", DEFAULT_INDEX))
        if index not in INDEX_NAMES:
            raise ValueError(
                f"liquid rules: unknown index '{index}'"
                f" (expected one of {', '.join(INDEX_NAMES)})"
            )
        return cls(bands=bands, index=index)

    @classmethod
    def load(cls, path: Union[str, Path]) -> "LiquidRules":
        try:
            data = json.loads(Path(path).read_text(encoding="utf-8"))
        except Exception as e:
            raise ValueError(f"liquid rules: cannot read {path}: {e}") from e
        return cls.from_dict(data)

    def to_dict(self) -> Dict:
        return {
            "index": self.index,
            "bands": [b.to_dict() for b in self.bands],
        }

    def save(self, path: Union[str, Path]):
        Path(path).write_text(
            json.dumps(self.to_dict(), indent=2) + "\n", encoding="utf-8"
        )

    # -- queries -------------------------------------------------------------

    def band_for(self, layer_idx: int) -> Optional[LiquidBand]:
        for b in self.bands:
            if b.covers(layer_idx):
                return b
        return None

    def validate(self, n_layer: int, k_native: int, n_expert: int) -> None:
        """Fail fast on invalid rules (mirrors validate_expansion)."""
        seen: set = set()
        for b in self.bands:
            if b.start < 0 or b.end < b.start or b.end >= n_layer:
                raise ValueError(
                    f"liquid rules: band {b.start}-{b.end} outside the layer"
                    f" range [0, {n_layer - 1}]"
                )
            for l in range(b.start, b.end + 1):
                if l in seen:
                    raise ValueError(
                        f"liquid rules: layer {l} is covered by more than one band"
                    )
                seen.add(l)
            for t in (b.perp_t, b.perp_next):
                if not (0.0 <= t <= 1.0):
                    raise ValueError(
                        f"liquid rules: threshold {t} outside [0, 1]"
                        f" (band {b.start}-{b.end})"
                    )
            if b.top_e < k_native or b.top_e > n_expert:
                raise ValueError(
                    f"liquid rules: top_e {b.top_e} outside"
                    f" [{k_native}, {n_expert}] (band {b.start}-{b.end})"
                )


def default_rules() -> LiquidRules:
    """The static rules of the experiment notes."""
    return LiquidRules(bands=list(DEFAULT_BANDS), index=DEFAULT_INDEX)


# -- perplexity index ------------------------------------------------------------
#
# All indices are functions of the current budget n: one value per token per
# candidate cut, so the ladder can re-evaluate after each added expert.


def index_series(
    name: str,
    w: mx.array,
    gates: Optional[mx.array],
    k_native: int,
    n_max: int,
) -> List[mx.array]:
    """Candidate-cut perplexity index on the graph.

    Args:
      name: one of INDEX_NAMES.
      w: [..., n_max] rank-ordered (descending) router probabilities of the
        selection prefix (softmax output, sums to 1 over the FULL set).
      gates: [..., n_expert] the full distribution (only ``entropy``/``eff``
        need it; pass None otherwise).
      k_native: native top-K (the starting budget; the first check runs at
        budget K).
      n_max: the largest budget (band topE). One array is returned per
        candidate budget n = K..n_max-1 (the check that decides whether rank
        n+1 is added).

    Returns: list of [...] arrays, i-th element = index at budget K+i.
    """
    if name not in INDEX_NAMES:
        raise ValueError(
            f"liquid index: unknown index '{name}'"
            f" (expected one of {', '.join(INDEX_NAMES)})"
        )
    cuts = list(range(k_native, n_max))
    if not cuts:
        return []
    if name in ("tail_rel", "tail"):
        cum = mx.cumsum(w, axis=-1)  # cum[..., j] = mass of ranks 0..j
        out = []
        for n in cuts:
            tail = 1.0 - cum[..., n - 1]  # mass beyond the cut
            if name == "tail":
                out.append(tail)
            else:
                p_last = w[..., n - 1]
                out.append(mx.clip(tail / mx.maximum(p_last, _EPS), 0.0, 1.0))
        return out
    if name == "entropy":
        if gates is None:
            raise ValueError("liquid index 'entropy' needs the full distribution")
        n_expert = gates.shape[-1]
        p = mx.clip(gates, _EPS, 1.0)
        h = -(p * mx.log(p)).sum(axis=-1) / math.log(n_expert)
        return [h for _ in cuts]  # cut-independent: same value re-checked
    if name == "kept_entropy":
        # entropy of the renormalized kept prefix, divided by ln(n): the
        # flatness of what is kept so far
        cum = mx.cumsum(w, axis=-1)
        cl = mx.cumsum(mx.clip(w, _EPS, 1.0) * mx.log(mx.clip(w, _EPS, 1.0)), axis=-1)
        out = []
        for n in cuts:
            s = cum[..., n - 1]
            h = mx.log(mx.maximum(s, _EPS)) - cl[..., n - 1] / mx.maximum(s, _EPS)
            out.append(h / math.log(n))
        return out
    # eff: normalized effective expert count exp(H) anchored on the budget
    if gates is None:
        raise ValueError("liquid index 'eff' needs the full distribution")
    n_expert = gates.shape[-1]
    p = mx.clip(gates, _EPS, 1.0)
    perp = mx.exp(-(p * mx.log(p)).sum(axis=-1))
    out = []
    for n in cuts:
        out.append(mx.clip((perp - n) / max(1, n_expert - n), 0.0, 1.0))
    return out


def liquid_ladder(
    idx_list: List[mx.array],
    perp_t: float,
    perp_next: float,
    k_native: int,
) -> mx.array:
    """Per-token kept-expert count from the per-budget index values.

    pass(i) = idx(i) > (perp_t if i == 0 else perp_next); the kept count is
    K plus the length of the LEADING run of passes ("e cosi' via fino al
    topE": the first failed check stops the additions).
    """
    total = None
    active = None
    for i, idx in enumerate(idx_list):
        thr = perp_t if i == 0 else perp_next
        passed = (idx > thr).astype(mx.float32)
        active = passed if active is None else active * passed
        total = active if total is None else total + active
    return k_native + (total if total is not None else 0.0)


def liquid_post_pass(
    w: mx.array,
    gates: Optional[mx.array],
    band: LiquidBand,
    index: str,
    k_native: int,
    decay_end: float,
    no_decay: bool,
    renormalize: bool,
) -> Tuple[mx.array, mx.array]:
    """Shape the rank-ordered top-topE weights with the liquid ladder.

    Args mirror ``expansion_post_pass``: ``w`` [..., top_e] descending router
    probabilities (fixed-N mask slots); returns (weights, kept_count) with
    dropped ranks at zero weight, the extra ranks (>= K) decayed 0.99..D
    against the band topE, and the kept weights optionally renormalized.
    """
    n_max = w.shape[-1]
    idx = index_series(index, w, gates, k_native, n_max)
    kept = liquid_ladder(idx, band.perp_t, band.perp_next, k_native)

    ranks = mx.arange(n_max, dtype=w.dtype)  # 0-based rank index j
    keep = (ranks < kept[..., None]).astype(w.dtype)
    w = w * keep

    # expansion decay on the extra ranks, computed against the maximum
    # budget n_max so a rank's factor does not jump when the cut moves
    # (same ramp as expansion_post_pass)
    if n_max > k_native and not no_decay:
        sel_k = (ranks >= k_native).astype(w.dtype)
        if n_max - k_native > 1:
            t = mx.clip((ranks - k_native) / (n_max - k_native - 1), 0.0, 1.0)
        else:
            t = sel_k * 0.5
        fac = sel_k * (0.99 + t * (decay_end - 0.99)) + (1.0 - sel_k)
        w = w * fac

    if renormalize:
        wsum = mx.clip(w.sum(axis=-1, keepdims=True), _RENORM_MIN, None)
        w = w / wsum
    return w, kept


# -- router distribution capture (phase 1 of the protocol) -----------------------


class RouterCapture:
    """In-memory accumulator of the full router distributions.

    With a sink installed (``set_capture_sink``), every MoE call appends the
    post-softmax distribution of its layer (all expert scores, one row per
    processed position). The arrays are lazy graph branches: the engine
    drains them at its sync points (same pattern as ExpertUsage). A "run"
    is one prompt generation: start_run() opens it (captures outside a run
    — e.g. the warmup — are ignored), save_run() materializes everything
    into one .npz and closes it.
    """

    def __init__(self, n_layer: int, n_expert: int):
        self.n_layer = n_layer
        self.n_expert = n_expert
        self._lazy: deque = deque()
        self.rows: List[List[np.ndarray]] = [[] for _ in range(n_layer)]
        self.run: Optional[Dict] = None
        self.dropped = 0  # rows given up on (never materialized)

    # -- engine-side ---------------------------------------------------------

    def observe_lazy(self, layer_idx: int, gates: mx.array):
        """Append (lazy, no eval) the full distribution of one MoE call."""
        if self.run is None:
            return  # outside a run (warmup, other slots' traffic)
        self._lazy.append((layer_idx, gates))

    def drain(self):
        """Convert pending lazy distributions (call after sync points on the
        engine thread)."""
        while self._lazy:
            layer_idx, gates = self._lazy[0]
            try:
                vals = np.asarray(gates, dtype=np.float32).reshape(-1, self.n_expert)
            except Exception:
                break  # not materialized yet; retry on the next drain
            self._lazy.popleft()
            self.rows[layer_idx].append(vals)

    def pending(self) -> int:
        return len(self._lazy)

    # -- run boundaries ------------------------------------------------------

    def start_run(self, meta: Dict):
        """Open a run: buffers are cleared and captures start flowing."""
        self._lazy.clear()
        self.rows = [[] for _ in range(self.n_layer)]
        self.dropped = 0
        self.run = dict(meta)

    def drain_all(self):
        """Drain until empty (arrays become materialized after the engine
        syncs; safe to call repeatedly)."""
        for _ in range(64):
            before = len(self._lazy)
            self.drain()
            if len(self._lazy) == 0 or len(self._lazy) == before:
                break

    def save_run(self, path: Union[str, Path]) -> Dict:
        """Close the run into one .npz (and reset). Returns the run meta."""
        if self.run is None:
            raise ValueError("router capture: no run is open")
        self.drain_all()
        counts = [sum(r.shape[0] for r in layer_rows) for layer_rows in self.rows]
        n_rows = min(counts) if counts else 0
        self.dropped = sum(counts) - n_rows
        if n_rows == 0:
            raise ValueError(
                "router capture: no distribution captured (was the model"
                " forwarded with the sink installed?)"
            )
        # [T, L, E] — positions in engine order (prefill chunks first, then
        # the decode steps), layers stacked on axis 1
        scores = np.stack(
            [
                np.concatenate(self.rows[l], axis=0)[:n_rows]
                for l in range(self.n_layer)
            ],
            axis=1,
        ).astype(np.float16)
        meta = dict(self.run)
        meta.update(
            {
                "n_rows": int(n_rows),
                "n_layer": self.n_layer,
                "n_expert": self.n_expert,
                "saved_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
                "dropped_rows": int(self.dropped),
            }
        )
        tokens = np.asarray(meta.pop("tokens", []), dtype=np.int32)
        prompt_tokens = np.asarray(meta.pop("prompt_tokens", []), dtype=np.int32)
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            p,
            scores=scores,
            n_prompt=np.int64(meta.get("n_prompt", 0)),
            tokens=tokens,
            prompt_tokens=prompt_tokens,
            meta=np.array(json.dumps(meta)),
        )
        self.run = None
        self.rows = [[] for _ in range(self.n_layer)]
        return meta

    def abort_run(self):
        self.run = None
        self._lazy.clear()
        self.rows = [[] for _ in range(self.n_layer)]


# module-level sink (server single-model): wired by `agrimlxo capture`,
# observed by routed_forward at every MoE call
CAPTURE_SINK: Optional[RouterCapture] = None


def set_capture_sink(sink: Optional[RouterCapture]) -> None:
    global CAPTURE_SINK
    CAPTURE_SINK = sink


# -- off-line analysis (numpy twin of the graph formulas) ------------------------


@dataclass
class CaptureRun:
    """One saved capture .npz (``agrimlxo capture`` output)."""

    path: str
    scores: np.ndarray  # [T, L, E] float32 (post-softmax, unsorted)
    n_prompt: int  # rows belonging to the prompt/prefill
    tokens: List[int]  # generated token ids
    prompt_tokens: List[int]
    meta: Dict

    @property
    def n_decode(self) -> int:
        return max(0, self.scores.shape[0] - self.n_prompt)

    def decode_scores(self) -> np.ndarray:
        """Rows generated after the prompt (the protocol's tokens)."""
        return self.scores[self.n_prompt:]


def load_capture(path: Union[str, Path]) -> CaptureRun:
    path = str(path)
    with np.load(path, allow_pickle=False) as d:
        scores = d["scores"].astype(np.float32)
        n_prompt = int(d["n_prompt"]) if "n_prompt" in d else 0
        tokens = d["tokens"].tolist() if "tokens" in d else []
        prompt_tokens = d["prompt_tokens"].tolist() if "prompt_tokens" in d else []
        meta = {}
        if "meta" in d:
            try:
                meta = json.loads(str(d["meta"]))
            except Exception:
                meta = {}
    return CaptureRun(
        path=path,
        scores=scores,
        n_prompt=n_prompt,
        tokens=tokens,
        prompt_tokens=prompt_tokens,
        meta=meta,
    )


def index_series_np(
    name: str,
    scores: np.ndarray,
    k_native: int,
    n_max: int,
) -> np.ndarray:
    """Numpy twin of ``index_series`` for the off-line tools.

    Args: scores [..., n_expert] post-softmax distributions (any order).
    Returns: [..., n_checks] with check i = index at budget K+i.
    """
    if name not in INDEX_NAMES:
        raise ValueError(f"liquid index: unknown index '{name}'")
    w = np.sort(scores, axis=-1)[..., ::-1][..., :n_max]
    cuts = list(range(k_native, n_max))
    if not cuts:
        return np.zeros(scores.shape[:-1] + (0,), dtype=np.float32)
    if name in ("tail_rel", "tail"):
        cum = np.cumsum(w, axis=-1)
        out = []
        for n in cuts:
            tail = 1.0 - cum[..., n - 1]
            if name == "tail":
                out.append(tail)
            else:
                out.append(np.clip(tail / np.maximum(w[..., n - 1], _EPS), 0.0, 1.0))
        return np.stack(out, axis=-1)
    if name == "entropy":
        n_expert = scores.shape[-1]
        p = np.clip(scores, _EPS, 1.0)
        h = -(p * np.log(p)).sum(axis=-1) / math.log(n_expert)
        return np.stack([h for _ in cuts], axis=-1)
    if name == "kept_entropy":
        cum = np.cumsum(w, axis=-1)
        cw = np.clip(w, _EPS, 1.0)
        cl = np.cumsum(cw * np.log(cw), axis=-1)
        out = []
        for n in cuts:
            s = np.maximum(cum[..., n - 1], _EPS)
            h = np.log(s) - cl[..., n - 1] / s
            out.append(h / math.log(n))
        return np.stack(out, axis=-1)
    # eff
    n_expert = scores.shape[-1]
    p = np.clip(scores, _EPS, 1.0)
    perp = np.exp(-(p * np.log(p)).sum(axis=-1))
    out = []
    for n in cuts:
        out.append(np.clip((perp - n) / max(1, n_expert - n), 0.0, 1.0))
    return np.stack(out, axis=-1)


def layer_stats(idx: np.ndarray) -> Dict[str, np.ndarray]:
    """Per-layer stats of one index series.

    Args: idx [..., L, n_checks]. Returns mean/p50/p90 arrays [L] computed
    on the FIRST check (budget = native K: the decision that starts the
    ladder) — the other checks are available for threshold-step tuning.
    """
    first = idx[..., 0].reshape(-1, idx.shape[-2])
    return {
        "mean": first.mean(axis=0),
        "p50": np.median(first, axis=0),
        "p90": np.quantile(first, 0.9, axis=0),
    }


def summarize_runs(
    runs: List[CaptureRun],
    index: str,
    k_native: int,
    n_max: int,
    decode_only: bool = True,
) -> Dict[str, Dict[str, np.ndarray]]:
    """Aggregate the index stats per layer across runs (phase-2 analysis)."""
    per_index: Dict[str, Dict[str, np.ndarray]] = {}
    chunks = []
    for run in runs:
        scores = run.decode_scores() if decode_only else run.scores
        if scores.shape[0] == 0:
            continue
        chunks.append(index_series_np(index, scores, k_native, n_max))
    if not chunks:
        raise ValueError("no capture rows to analyze")
    idx = np.concatenate(chunks, axis=0)
    per_index[index] = layer_stats(idx)
    return per_index


def compare_runs(
    base: CaptureRun,
    other: CaptureRun,
    index: str,
    k_native: int,
    n_max: int,
) -> Dict:
    """Phase-3 comparison: per-layer index stats + output agreement."""
    def stats_of(run: CaptureRun) -> Dict[str, np.ndarray]:
        scores = run.decode_scores()
        if scores.shape[0] == 0:
            scores = run.scores
        return layer_stats(index_series_np(index, scores, k_native, n_max))

    sb, so = stats_of(base), stats_of(other)
    n_common = min(len(base.tokens), len(other.tokens))
    prefix = 0
    for a, b in zip(base.tokens, other.tokens):
        if a != b:
            break
        prefix += 1
    agree = (
        sum(1 for a, b in zip(base.tokens, other.tokens) if a == b) / n_common
        if n_common
        else 0.0
    )
    return {
        "index": index,
        "base": sb,
        "other": so,
        "delta_mean": so["mean"] - sb["mean"],
        "tokens_base": len(base.tokens),
        "tokens_other": len(other.tokens),
        "common_prefix": prefix,
        "token_agreement": agree,
    }
