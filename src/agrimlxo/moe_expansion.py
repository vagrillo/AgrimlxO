# SPDX-License-Identifier: Apache-2.0
"""MoE expert expansion — runtime-only routing modification for sparse MoE models.

Raise the routed-expert budget above the model's native top-K with a linear
influence decay on the extra ranks, optionally with an adaptive
(threshold-based) expert count and a layer range. This is the MLX port of the
"layer-scoped expert-budget expansion" feature described in

    Vincenzo Agrillo, "Succinct convergence ..." (Zenodo, 2026)
    https://zenodo.org/records/22255483

first shipped in the llama.cpp fork `vagrillo/llama.cpp` (branch
`moe-expansion`, `src/llama-moe-expansion.h`) and as AgrillaMoE.

Implementation strategy is the fixed-N mask ("Strategy B"): the routing graph
always carries N selection slots per token and dropped ranks simply get zero
weight, so the downstream expert kernels stay stock. With the feature off the
engine behaves bit-identically to stock: every flag off (or a layer outside
the configured range) takes the untouched stock routing code path.

The three knobs (llama.cpp-compatible semantics):
  1. Max experts N: raise the routed expert budget above the native top-K
     (or `--moe-experts-add` on top of it; values below K = fixed pruning).
  2. Dynamic threshold T: keep ranks while their probability stays above
     T x p_ref, where p_ref is the probability of rank N/2 (1-indexed, i.e.
     0-based index N/2 - 1). The per-token expert count becomes adaptive with
     the floor max(1, N/4).
  3. Expansion decay D: ranks above the native K get a linearly decaying
     influence factor from 0.99 down to D applied to the raw probability
     *before* renormalization, so the native experts keep the output scale and
     the extras fade out.
"""

from __future__ import annotations

import math
import os
import sys
from collections import deque
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import time
from pathlib import Path

import mlx.core as mx
import numpy as np

# sink globale dell'uso degli esperti (server single-model): osservato da
# routed_forward a ogni chiamata MoE, svuotato dal motore dopo i sync
USAGE_SINK: Optional["ExpertUsage"] = None


def set_usage_sink(sink: Optional["ExpertUsage"]) -> None:
    global USAGE_SINK
    USAGE_SINK = sink


def observe_usage(layer_idx: int, ids: mx.array, k_native: int):
    """Append (lazy, no eval) the native top-K expert ids of one MoE call to
    the sink; the engine drains it after its sync points."""
    if USAGE_SINK is None:
        return
    USAGE_SINK.observe_lazy(layer_idx, ids, k_native)


class ExpertUsage:
    """Accumulator of native top-K expert usage per layer.

    counts_all: since server start; counts_session: resettable (chat session).
    Only the first k_native rank-ordered ids count, even when the expansion
    raises the routed budget above the native top-K.
    """

    def __init__(
        self,
        n_layer: int,
        n_expert: int,
        persist_path: Optional[str] = None,
        model_name: str = "",
    ):
        self.n_layer = n_layer
        self.n_expert = n_expert
        self.counts_all = np.zeros((n_layer, n_expert), dtype=np.int64)
        self.counts_session = np.zeros((n_layer, n_expert), dtype=np.int64)
        self.tokens_all = 0
        self.tokens_session = 0
        self._lazy: deque = deque()
        # persistenza all-time per modello (./expertusage/<model>.json):
        # caricata all'avvio, salvata throttled durante il drain e flushata
        # allo shutdown
        self.persist_path = Path(persist_path) if persist_path else None
        self.model_name = model_name
        self._last_save = 0.0
        if self.persist_path:
            self._load()

    def _load(self):
        import json

        try:
            if self.persist_path.exists():
                data = json.loads(self.persist_path.read_text())
                counts = np.asarray(data.get("counts_all"), dtype=np.int64)
                if counts.shape == self.counts_all.shape:
                    self.counts_all = counts
                    self.tokens_all = int(data.get("tokens_all", 0))
        except Exception as e:
            import logging

            logging.getLogger(__name__).warning(
                "expert usage load %s: %s", self.persist_path, e
            )

    def _persist(self):
        import json

        if not self.persist_path:
            return
        try:
            self.persist_path.parent.mkdir(parents=True, exist_ok=True)
            self.persist_path.write_text(json.dumps({
                "model": self.model_name,
                "layers": self.n_layer,
                "experts": self.n_expert,
                "tokens_all": self.tokens_all,
                "counts_all": self.counts_all.tolist(),
            }))
            self._last_save = time.time()
        except Exception as e:
            import logging

            logging.getLogger(__name__).warning(
                "expert usage save %s: %s", self.persist_path, e
            )

    def flush(self):
        """Save the all-time counts now (shutdown / explicit)."""
        if self.persist_path:
            self._persist()

    def observe_lazy(self, layer_idx: int, ids: mx.array, k_native: int):
        self._lazy.append((layer_idx, ids, min(k_native, ids.shape[-1])))

    def drain(self):
        """Convert pending lazy ids to counts (call after sync points on the
        engine thread: the ids are materialized by the forward)."""
        while self._lazy:
            layer_idx, ids, k = self._lazy[0]
            try:
                vals = np.asarray(ids[..., :k], dtype=np.int64).reshape(-1)
            except Exception:
                break  # non materialized yet
            self._lazy.popleft()
            np.add.at(self.counts_all[layer_idx], vals, 1)
            np.add.at(self.counts_session[layer_idx], vals, 1)
            self.tokens_all += 1
            self.tokens_session += 1
            # salvataggio throttled (30s) durante il drain
            if self.persist_path and time.time() - self._last_save > 30:
                self._persist()

    def reset_session(self):
        # svuota anche la coda lazy: gli osservati in volo (schedulati con
        # async_eval prima del reset) non devono riaccendersi dopo
        self._lazy.clear()
        self.counts_session[:] = 0
        self.tokens_session = 0

    def snapshot(self, session: bool = False) -> Dict:
        counts = self.counts_session if session else self.counts_all
        return {
            "layers": self.n_layer,
            "experts": self.n_expert,
            "tokens": self.tokens_session if session else self.tokens_all,
            "counts": counts.tolist(),
        }

# Minimum kept-weight sum guard used by the llama.cpp implementation (2^-14).
_RENORM_MIN = 6.103515625e-5


@dataclass
class MoEExpansionConfig:
    """User-facing MoE expansion configuration (see docs/moe-expansion.md).

    All defaults off → the engine behaves exactly like stock.
    """

    # --moe-experts N: absolute max routed experts per token (0 = off)
    max_experts: int = 0
    # --moe-experts-add N: experts added on top of the model's native top-K
    add_experts: int = 0
    # --moe-expert-threshold T: adaptive count (0 = off; T in (0, 10])
    threshold: float = 0.0
    # --moe-expert-decay-end D: influence of the last extra rank (0.50 default)
    decay_end: float = 0.50
    # --moe-no-expert-decay: extra experts at full influence
    no_decay: bool = False
    # --moe-expert-renorm {auto,always,never}
    renorm: str = "auto"
    # --moe-expert-layer-start I: < 1 fraction of n_layer, >= 1 layer index
    layer_start: float = 0.0
    # --moe-expert-layer-end I: < 0 last layer, < 1 fraction, >= 1 index
    layer_end: float = -1.0
    # --moe-stats-every N (or MOE_EXPERT_STATS_EVERY): observability cadence
    stats_every: int = 0
    # --moe-expert-log FILE: per-token kept-expert log (moe_predict.py format)
    expert_log: str = ""

    @property
    def requested(self) -> bool:
        """True when any expansion flag was explicitly requested."""
        return (
            self.max_experts > 0
            or self.add_experts > 0
            or self.threshold > 0.0
        )

    def resolve_n(self, k_native: int, n_expert: int) -> int:
        """Resolve the expanded budget N from the flags and the model geometry."""
        n = self.max_experts
        if n <= 0 and self.add_experts > 0:
            n = k_native + self.add_experts
        return n


@dataclass
class MoEReduceConfig:
    """--moe-reduce: inverse of the expansion (dynamic routing pruning).

    Keep only experts whose weight >= reduce_factor x weight(top-1), on the
    (optional) layer range --moe-reduce-layers S-E (inclusive, 0-based; -1 =
    last layer).
    """

    factor: float = 0.0  # 0 = off
    layers: Tuple[int, int] = (0, -1)

    def active_for(self, layer_idx: int) -> bool:
        start, end = self.layers
        return self.factor > 0.0 and layer_idx >= start and (
            end < 0 or layer_idx <= end
        )


def validate_expansion(
    cfg: MoEExpansionConfig,
    is_moe: bool,
    k_native: int,
    n_expert: int,
    n_layer: int,
    norm_topk_prob: bool,
) -> Optional[Tuple[int, int, int]]:
    """Validate an expansion config against the model geometry.

    Returns the resolved (N, layer_start_idx, layer_end_idx) or ``None`` when
    the feature is off. Raises ``ValueError`` with a clear message (fail fast,
    non-zero exit) for invalid combinations — mirroring the llama.cpp build.
    """
    if cfg.renorm not in ("auto", "always", "never"):
        raise ValueError(
            f"--moe-expert-renorm: invalid mode '{cfg.renorm}'"
            " (expected auto, always or never)"
        )
    if cfg.threshold > 0.0 and not (0.0 < cfg.threshold <= 10.0):
        raise ValueError(
            f"--moe-expert-threshold: {cfg.threshold} outside the valid range (0, 10]"
        )
    if cfg.decay_end <= 0.0 or cfg.decay_end >= 0.99:
        raise ValueError(
            f"--moe-expert-decay-end: {cfg.decay_end} outside the valid range (0, 0.99)"
        )
    if cfg.max_experts > 0 and (cfg.max_experts < 2 or cfg.max_experts > n_expert):
        raise ValueError(
            f"--moe-experts: {cfg.max_experts} outside the valid range [2, {n_expert}]"
        )
    if cfg.max_experts == 0 and cfg.add_experts > 0:
        n = k_native + cfg.add_experts
        if n < 2 or n > n_expert:
            raise ValueError(
                f"--moe-experts-add: {cfg.add_experts} gives N = {n} outside the"
                f" valid range [2, {n_expert}] (native top-K = {k_native})"
            )

    if not cfg.requested:
        return None
    if not is_moe:
        raise ValueError(
            "MoE expert expansion requested but the model is not a sparse MoE"
            " model (no routed experts in its configuration)"
        )
    if cfg.threshold > 0.0 and cfg.max_experts == 0 and cfg.add_experts == 0:
        raise ValueError(
            "--moe-expert-threshold requires --moe-experts N (or --moe-experts-add N)"
        )

    n = cfg.resolve_n(k_native, n_expert)
    assert 2 <= n <= n_expert

    # Layer range: < 1 means fraction of n_layer, >= 1 means a layer index.
    start = cfg.layer_start
    end = cfg.layer_end
    start_idx = int(start * n_layer) if 0.0 <= start < 1.0 else int(start)
    if end < 0:
        end_idx = n_layer - 1
    elif end < 1.0:
        end_idx = int(end * n_layer)
    else:
        end_idx = int(end)
    start_idx = max(0, min(start_idx, n_layer - 1))
    end_idx = max(0, min(end_idx, n_layer - 1))
    if start_idx > end_idx:
        raise ValueError(
            f"--moe-expert-layer-start/end: start {cfg.layer_start} (layer"
            f" {start_idx}) is after end {cfg.layer_end} (layer {end_idx})"
        )
    return n, start_idx, end_idx


def validate_reduce(
    cfg: MoEReduceConfig, n_layer: int
) -> None:
    """Validate --moe-reduce parameters (fail fast)."""
    if cfg.factor <= 0.0:
        return
    if cfg.factor > 10.0:
        raise ValueError(
            f"--moe-reduce: {cfg.factor} outside the valid range (0, 10]"
        )
    start, end = cfg.layers
    if end >= 0 and start > end:
        raise ValueError(
            f"--moe-reduce-layers: start {start} is after end {end}"
        )


def expansion_post_pass(
    weights: mx.array,
    n_used: int,
    k_native: int,
    threshold: float,
    decay_end: float,
    no_decay: bool,
    renormalize: bool,
):
    """MLX port of ``build_moe_expansion_weights`` (llama-moe-expansion.h).

    Args:
      weights: [.., n_tokens, n_used] the top-N router probabilities in rank
        order, non-increasing per token (raw softmax probabilities, *before*
        any renormalization).
      n_used: N, the expanded routed-expert budget (N >= 2).
      k_native: K, the model's native top-k (decay ramp reference).
      threshold: T, keep ranks while weight > T x weight(rank N/2); 0 = keep N.
      decay_end: D, influence of the last extra rank; linear 0.99..D.
      no_decay: extra experts at full influence.
      renormalize: true -> kept weights renormalized to sum 1 (softmax-style
        routers, e.g. Qwen); false -> kept weights keep their raw decayed
        score scale (raw-score routers).

    Returns:
      (weights, sel_count): the post-processed weights (same shape) and the
      per-token count of kept experts [n_tokens].
    """
    assert n_used >= 2
    ranks = mx.arange(n_used, dtype=weights.dtype)  # 0-based rank index j

    # selection: keep = min(step(F - j) + step(w - T * p_ref), 1), with
    #   F     = max(1, N/4)              (floor: core ranks always kept)
    #   p_ref = weight of rank R - 1 (0-based), R = N/2 (1-indexed)
    # the kept set is a contiguous prefix since the weights are non-increasing
    if threshold > 0.0:
        n_floor = max(1, n_used // 4)
        keep_floor = ranks < n_floor  # step(F - j)
        p_ref = weights[..., n_used // 2 - 1]  # [n_tokens]
        above = weights > threshold * p_ref[..., None]  # step(w - T*p_ref), strict
        keep = mx.minimum(keep_floor.astype(weights.dtype) + above.astype(weights.dtype), 1.0)
    else:
        keep = mx.ones_like(weights)

    sel_count = keep.sum(axis=-1)

    w = weights * keep

    # expansion decay: when N exceeds the model's native top-K, every kept
    # rank above K gets a linearly decaying influence factor, 0.99 for the
    # first extra rank down to D for rank N (midpoint with a single extra).
    # Computed against the maximum N, so a rank's factor does not jump when
    # the adaptive cut moves.
    if n_used > k_native and not no_decay:
        sel_k = ranks >= k_native  # step(j - K + 0.5)
        sel_k = sel_k.astype(w.dtype)
        if n_used - k_native > 1:
            t = mx.clip((ranks - k_native) / (n_used - k_native - 1), 0.0, 1.0)
        else:
            t = sel_k * 0.5  # single extra rank -> midpoint factor
        fac = sel_k * (0.99 + t * (decay_end - 0.99)) + (1.0 - sel_k)
        w = w * fac

    # renormalize the kept weights to sum 1 per token. With renormalize=False
    # the kept weights keep their raw decayed score scale — renormalizing
    # would change the routed-branch output magnitude vs the stock behavior.
    if renormalize:
        wsum = mx.clip(w.sum(axis=-1, keepdims=True), _RENORM_MIN, None)
        w = w / wsum

    return w, sel_count


def reduce_post_pass(
    weights: mx.array,
    factor: float,
    renormalize: bool,
):
    """--moe-reduce: keep only experts with weight >= factor x weight(top-1).

    The top-1 expert is always kept. Returns (weights, sel_count).
    """
    keep = weights >= factor * weights[..., :1]
    keep = keep.astype(weights.dtype)
    sel_count = keep.sum(axis=-1)
    w = weights * keep
    if renormalize:
        wsum = mx.clip(w.sum(axis=-1, keepdims=True), _RENORM_MIN, None)
        w = w / wsum
    return w, sel_count


@dataclass
class _LayerStats:
    """Per-expanded-layer accumulator of kept-expert counts (mx arrays)."""

    counts: List[mx.array] = field(default_factory=list)


class MoERuntime:
    """Runtime state shared by the MoE layers of one loaded model.

    Holds the resolved expansion parameters, routes the per-layer activation
    decision and accumulates the observability counters. A model layer asks
    ``runtime.active_for(layer_idx, k_native, n_expert)``: when False the
    layer runs the untouched stock routing path (bit-identical guarantee).
    """

    def __init__(
        self,
        n_layer: int,
        norm_topk_prob: bool,
        config: Optional[MoEExpansionConfig] = None,
        reduce: Optional[MoEReduceConfig] = None,
    ):
        self.n_layer = n_layer
        self.norm_topk_prob = norm_topk_prob
        self.config = config or MoEExpansionConfig()
        self.reduce = reduce or MoEReduceConfig()
        self.n: int = 0  # resolved expanded budget (0 = inactive everywhere)
        self.k_native: int = 0
        self.n_expert: int = 0
        self.layer_start: int = 0
        self.layer_end: int = -1
        self._stats: Dict[int, _LayerStats] = {}
        self._tokens_since_stats: int = 0
        self._pending_tokens: int = 0
        self._log_fh = None
        self._log_pos: List[int] = []
        # warmup runs generation through the real model before the server
        # listens: suspend observability so the stats window stays clean
        self.suspended: bool = False
        # lazy kept-count branches awaiting materialization (scheduled with
        # async_eval by record()); drained by token_stats() once the decode
        # sync guarantees their compute ran
        self._lazy: deque = deque()
        self._live_sum: float = 0.0  # kept-expert total over all layers/tokens
        self._live_n: int = 0  # layer-call count backing _live_sum
        self._live_last: Optional[float] = None  # mean over the last token's layers

    # -- setup ---------------------------------------------------------------

    def setup(self, is_moe: bool, k_native: int, n_expert: int) -> "MoERuntime":
        """Validate against the model geometry and resolve the layer range."""
        self.k_native = k_native
        self.n_expert = n_expert
        resolved = validate_expansion(
            self.config,
            is_moe,
            k_native,
            n_expert,
            self.n_layer,
            self.norm_topk_prob,
        )
        if resolved is not None:
            self.n, self.layer_start, self.layer_end = resolved
        else:
            self.n = 0
        validate_reduce(self.reduce, self.n_layer)
        if self.reduce.factor > 0.0 and self.n > 0:
            raise ValueError(
                "--moe-reduce and --moe-experts/--moe-expert-threshold are"
                " mutually exclusive routing modifications"
            )
        every = self.config.stats_every
        if every == 0:
            every = int(os.environ.get("MOE_EXPERT_STATS_EVERY", "128"))
        self.stats_every = max(0, every)
        if self.config.expert_log:
            self._log_fh = open(self.config.expert_log, "w", encoding="utf-8")
        return self

    @property
    def active(self) -> bool:
        return self.n > 0

    @property
    def reduce_active(self) -> bool:
        return self.reduce.factor > 0.0

    def active_for(self, layer_idx: int, k_native: int, n_expert: int) -> bool:
        """Whether the expansion graph applies to this layer call."""
        if self.n <= 0:
            return False
        if not (self.layer_start <= layer_idx <= self.layer_end):
            return False
        # N == K with no threshold and auto renorm is exactly stock: take the
        # untouched stock routing path (bit-identical guarantee). A forced
        # renorm mode (always/never) changes semantics, so it keeps the
        # expansion path.
        if (
            self.n == k_native
            and self.config.threshold <= 0.0
            and self.config.renorm == "auto"
        ):
            return False
        return True

    def reduce_active_for(self, layer_idx: int) -> bool:
        return self.reduce_active and self.reduce.active_for(layer_idx)

    def renormalize(self) -> bool:
        """Whether the kept weights are renormalized (renorm mode `auto`
        follows the model's stock normalization)."""
        mode = self.config.renorm
        if mode == "always":
            return True
        if mode == "never":
            return False
        return bool(self.norm_topk_prob)

    # -- per-layer hooks -----------------------------------------------------

    def record(self, layer_idx: int, sel_count: mx.array, kept_ids: mx.array = None):
        """Accumulate observability data for one layer call.

        ``sel_count``: [n_tokens] kept-expert counts; ``kept_ids``: [n_tokens,
        n] rank-ordered expert ids (only when the expert log is enabled).

        The arrays are lazy branches of the model graph; scheduling them with
        ``mx.async_eval`` folds their evaluation into the running pipeline
        WITHOUT syncing it. Evaluating them later from another thread (stats
        printing) would otherwise hit "There is no Stream(gpu, N) in current
        thread" (they stayed bound to the async stream that never ran).
        """
        if self.suspended:
            return
        mx.async_eval(sel_count)
        self._lazy.append((layer_idx, sel_count))
        st = self._stats.setdefault(layer_idx, _LayerStats())
        st.counts.append(sel_count)
        self._pending_tokens += sel_count.shape[0]
        if self._log_fh is not None and kept_ids is not None:
            mx.async_eval(kept_ids)
            st.last_ids = kept_ids

    def token_stats(self) -> Tuple[Optional[float], Optional[float]]:
        """(experts of the last token, running mean) from materialized counts.

        Called right after the decode sync on the engine thread: the branch
        compute for previous steps has run (async_eval), so converting is
        safe and cheap; anything still lazy is materialized here (same
        thread that created it — no cross-stream hazard).
        """
        if not self.active:
            return None, None
        last_sum = 0.0
        last_n = 0
        while self._lazy:
            _, cnt = self._lazy[0]
            try:
                vals = np.asarray(cnt, dtype=np.float32).reshape(-1)
            except Exception as e:
                # non silenzioso: log una tantum — su Metal questo è il
                # segnale di un problema di stream da correggere
                if not getattr(self, "_drain_err_logged", False):
                    self._drain_err_logged = True
                    print(
                        f"moe: live experts unavailable (drain):"
                        f" {type(e).__name__}: {e}",
                        file=sys.stderr,
                        flush=True,
                    )
                break  # not materialized yet; retry on the next token
            self._lazy.popleft()
            last_sum += float(vals.sum())
            last_n += len(vals)
            self._live_sum += float(vals.sum())
            self._live_n += len(vals)
        if last_n:
            self._live_last = last_sum / last_n
        return self._live_last, (
            self._live_sum / self._live_n if self._live_n else None
        )

    def apply_config(
        self, config: MoEExpansionConfig, reduce: Optional[MoEReduceConfig] = None
    ) -> List[str]:
        """Reconfigure at runtime (webui MOE panel). Validates against the
        geometry captured at setup and swaps the per-call parameters; returns
        the new banner lines. Raises ValueError on invalid input."""
        if self.k_native <= 0:
            raise ValueError("runtime not initialized")
        cfg = config or MoEExpansionConfig()
        red = reduce if reduce is not None else self.reduce
        if red.factor > 0.0 and cfg.requested:
            raise ValueError(
                "--moe-reduce and the expansion are mutually exclusive"
            )
        n = cfg.resolve_n(self.k_native, self.n_expert) if cfg.requested else 0
        resolved = validate_expansion(
            {**cfg.__dict__},  # validated again below with a real instance
            True,
            self.k_native,
            self.n_expert,
            self.n_layer,
            self.norm_topk_prob,
        ) if False else None
        # validate_expansion expects a config object; reuse the instance path
        new_rt = MoERuntime(
            self.n_layer, self.norm_topk_prob, config=cfg, reduce=red
        )
        new_rt.setup(True, self.k_native, self.n_expert)
        # swap in place: le chiamate attive leggono i campi a ogni passaggio
        self.config = cfg
        self.reduce = red
        self.n = new_rt.n
        self.layer_start = new_rt.layer_start
        self.layer_end = new_rt.layer_end
        self.reset_stats()
        self._lazy.clear()
        self._live_sum = 0.0
        self._live_n = 0
        self._live_last = None
        return self.banner_lines()

    def reset_stats(self):
        """Clear the observability window (e.g. after warmup)."""
        self._stats.clear()
        self._tokens_since_stats = 0
        self._pending_tokens = 0
        self._lazy.clear()
        self._live_sum = 0.0
        self._live_n = 0
        self._live_last = None

    def flush_log(self, positions: List[int]):
        """Write the buffered per-token kept-expert lists (JSONL record).

        Format compatible with AgrillaMoE's LLAMA_MOE_EXPERT_LOG /
        moe_predict.py: {"pos": [...], "layer": {"25": [[id, ...], ...]}}.
        """
        if self._log_fh is None:
            return
        layers = {}
        for idx, st in self._stats.items():
            if getattr(st, "last_ids", None) is None:
                continue
            ids_arr = st.last_ids
            if ids_arr.ndim > 2:
                ids_arr = ids_arr.reshape(-1, ids_arr.shape[-1])
            elif ids_arr.ndim < 2:
                ids_arr = ids_arr.reshape(-1, self.n if self.n else ids_arr.shape[-1])
            ids = [[int(x) for x in row] for row in ids_arr]
            layers[str(idx)] = ids
            del st.last_ids
        if layers:
            self._log_fh.write(
                '{"pos": ' + str(list(map(int, positions))) + ', "layer": '
                + str(layers).replace("'", '"') + "}\n"
            )
            self._log_fh.flush()

    # -- observability -------------------------------------------------------

    def banner_lines(self) -> List[str]:
        """Startup banner lines, in the AgrillaMoE / llama.cpp format."""
        lines: List[str] = []
        if not self.active:
            if self.reduce_active:
                layers = self.reduce.layers
                rng = (
                    "all layers"
                    if layers[1] < 0 or layers[1] >= self.n_layer - 1
                    else f"layers {layers[0]}..{layers[1]}"
                )
                lines.append(
                    f"moe: expert reduce: keep experts with weight >= "
                    f"{self.reduce.factor:g} x weight(top-1) ({rng})"
                )
                return lines
            return lines
        k = self.k_native
        lines.append(
            f"moe: routed experts per token: {self.n} (model default {k}; "
            + (
                f"ranks {k + 1}..{self.n} get linear influence decay "
                f"0.99..{self.config.decay_end:g}"
                if not self.config.no_decay and self.n > k
                else "extra experts at full influence"
                if self.n > k
                else f"pruned below the native top-{k}"
            )
            + ")"
        )
        if self.config.threshold > 0.0:
            ref_rank = self.n // 2
            lines.append(
                f"moe: expert threshold: keep experts while p >= "
                f"{self.config.threshold:g} x p(rank {ref_rank}) "
                f"(experts {max(1, self.n // 4)}..{self.n} per token)"
            )
        if self.layer_start > 0 or self.layer_end < self.n_layer - 1:
            lines.append(
                f"moe: expansion active on layers {self.layer_start}.."
                f"{self.layer_end} of {self.n_layer}"
            )
        if self.config.renorm != "auto":
            lines.append(f"moe: kept-weight renormalization: {self.config.renorm}")
        return lines

    def stats_line(self) -> Optional[str]:
        """Aggregate accumulated counts into the periodic stats line.

        Returns the formatted line (and resets the window), or None when the
        window is not full yet / stats are disabled.
        """
        if self.stats_every <= 0 or not self._stats:
            return None
        if self._tokens_since_stats < self.stats_every:
            return None
        parts = []
        total = 0.0
        total_tokens = 0
        for idx in sorted(self._stats):
            counts = mx.concatenate([c.reshape(-1) for c in self._stats[idx].counts])
            mean = float(counts.mean())
            parts.append(f"L{idx}:{mean:.1f}")
            total += float(counts.sum())
            total_tokens += counts.shape[0]
            self._stats[idx].counts = []
        self._tokens_since_stats = 0
        mean_all = total / max(1, total_tokens)
        return " | ".join(parts) + f" | mean {mean_all:.1f}"

    def maybe_print_stats(self, force: bool = False):
        line = self.stats_line()
        if line:
            print(f"moe: experts/token avg: {line}", file=sys.stderr, flush=True)

    def note_tokens(self, n: int = 1):
        """Advance the processed-token counter for the stats window."""
        if self.suspended:
            return
        self._tokens_since_stats += n

    def close(self):
        if self._log_fh is not None:
            self._log_fh.close()
            self._log_fh = None


# -- AgrillaMoE default profiles -----------------------------------------------

# (block_count, expert_count, expert_count_used) -> profile, matched on the
# GGUF/HF metadata like AgrillaMoE's launcher does.
_PROFILES: Dict[Tuple[int, int, int], Dict] = {
    # Qwen3.5/3.6-35B-A3B: 40 layers, 256 experts, top-8 — user default
    # (paper benchmark recipe was 20/0.80; 12/0.75 is the lighter daily-
    # driving profile: ~1.5x routed compute instead of 2.5x).
    (40, 256, 8): {
        "max_experts": 12,
        "threshold": 0.75,
        "layer_start": 25,
        "layer_end": 39,
        "name": "Qwen3.5/3.6-35B-A3B",
    },
}


def default_profile_for(n_layer: int, n_expert: int, top_k: int) -> Optional[Dict]:
    """Signature-matched default expansion profile (AgrillaMoE behavior).

    Only injected when the user passes no --moe-* / --q35-* flag.
    """
    sig = (n_layer, n_expert, top_k)
    profile = _PROFILES.get(sig)
    if profile is not None:
        return profile
    # Generic MoE fallback: expansion without a layer range.
    if n_expert > 0 and top_k > 0 and n_expert >= 2 * top_k:
        return {
            "max_experts": 12,
            "threshold": 0.75,
            "name": "generic MoE",
        }
    return None
