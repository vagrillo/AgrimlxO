# SPDX-License-Identifier: Apache-2.0
"""Generation engine for AgrimlxO.

Single-model, single-stream MLX generation with:
  * chunked prefill (bounded activation memory),
  * per-slot KV-cache prefix reuse (a cached sequence that is a strict prefix
    of the new prompt is reused; the recurrent linear-attention state is only
    valid for exact continuations, so any other prompt re-prefills),
  * llama.cpp-style stop handling (eos ids, stop strings, max tokens, context
    limit),
  * the reasoning budget (--reasoning-budget): when the <think> phase exceeds
    the budget, a wrap-up message is appended to the context and streamed,
    forcing the end-of-thinking before a chain drifts (as in AgrillaMoE),
  * MoE expansion observability (stats window driven by processed tokens).
"""

import time
from dataclasses import dataclass, field
from threading import Lock
from typing import Iterator, List, Optional, Tuple

import mlx.core as mx

from .moe_expansion import MoERuntime
from .sampler import make_sampler, make_logits_processors
from .tokenizer import Tokenizer


@dataclass
class GenParams:
    """Per-request sampling and stopping parameters."""

    temp: float = 0.8
    top_k: int = 40
    top_p: float = 0.95
    min_p: float = 0.05
    xtc_probability: float = 0.0
    xtc_threshold: float = 0.1
    repeat_penalty: float = 1.0
    repeat_last_n: int = 64
    presence_penalty: float = 0.0
    frequency_penalty: float = 0.0
    logit_bias: Optional[dict] = None
    seed: Optional[int] = None
    max_tokens: int = -1  # -1 = until ctx/eos
    stop: List[str] = field(default_factory=list)
    ignore_eos: bool = False
    reasoning_budget: int = 0  # 0 = unlimited

    @property
    def greedy(self) -> bool:
        return self.temp == 0.0


@dataclass
class Step:
    token: int
    text: str  # detokenized delta for this step
    prompt_tokens: int
    generated: int  # tokens generated so far (incl. wrap-up)
    finish_reason: Optional[str]  # "stop" | "length" | None
    # timings (only set on the final Step): prefill/decode wall time in ms
    prefill_ms: Optional[float] = None
    decode_ms: Optional[float] = None


# Margin before the reasoning budget at which the wrap-up is injected: leaves
# room for the closing </think> and the start of the answer.
_WRAPUP_MARGIN = 24
_DEFAULT_WRAPUP = (
    "\n\nI have used up my reasoning budget. I will stop thinking and give"
    " my final answer now.\n</think>\n\n"
)


class ThinkSplit:
    """Split streamed text into reasoning_content (<think> phase) and content.

    Feeds raw deltas in, emits (kind, delta) pairs; keeps a small tail buffer
    so tags split across detokenization boundaries are still recognized.
    """

    _OPEN = "<think>"
    _CLOSE = "</think>"

    def __init__(self):
        self.state = "outside"  # outside | inside | closed
        self._tail = ""

    def push(self, text: str) -> List[Tuple[str, str]]:
        """Consume a raw delta; return the list of (kind, delta) fragments
        with kind in {"reasoning", "content"}."""
        self._tail += text
        out: List[Tuple[str, str]] = []
        keep = len(self._CLOSE)  # hold back enough chars for a split tag
        while True:
            if self.state == "closed":
                if self._tail:
                    out.append(("content", self._tail))
                    self._tail = ""
                return out
            if self.state == "outside":
                idx = self._tail.find(self._OPEN)
                if idx >= 0:
                    before = self._tail[:idx]
                    self._tail = self._tail[idx + len(self._OPEN):]
                    self.state = "inside"
                    if before:
                        out.append(("content", before))
                    continue
                if len(self._tail) > keep:
                    emit = self._tail[:-keep]
                    self._tail = self._tail[-keep:]
                    if emit:
                        out.append(("content", emit))
                return out
            # inside <think>
            idx = self._tail.find(self._CLOSE)
            if idx >= 0:
                body = self._tail[:idx]
                self._tail = self._tail[idx + len(self._CLOSE):]
                self.state = "closed"
                if body:
                    out.append(("reasoning", body))
                continue
            if len(self._tail) > keep:
                emit = self._tail[:-keep]
                self._tail = self._tail[-keep:]
                if emit:
                    out.append(("reasoning", emit))
            return out

    def flush(self) -> List[Tuple[str, str]]:
        """Emit whatever remains buffered (call at the end of the stream)."""
        out: List[Tuple[str, str]] = []
        if self._tail:
            kind = "content" if self.state != "inside" else "reasoning"
            out.append((kind, self._tail))
            self._tail = ""
        return out


class _Slot:
    """Per-slot cached state for prefix reuse (tokens stay in sync with the
    cache occupancy so a new request can continue from the cached prefix)."""

    __slots__ = ("tokens", "caches")

    def __init__(self):
        self.tokens: List[int] = []
        self.caches = None


class Engine:
    def __init__(
        self,
        loaded,  # loader.LoadedModel
        tokenizer: Tokenizer,
        ctx_size: int,
        prefill_step_size: int = 2048,
        parallel: int = 1,
        cache_type: str = "f16",
        wrapup_text: str = _DEFAULT_WRAPUP,
    ):
        self.loaded = loaded
        self.model = loaded.model
        self.tokenizer = tokenizer
        self.runtime: Optional[MoERuntime] = loaded.expansion
        self.ctx_size = ctx_size
        self.prefill_step_size = max(64, prefill_step_size)
        self.cache_type = cache_type
        self.wrapup_tokens = tokenizer.encode(wrapup_text)
        self.slots = [_Slot() for _ in range(max(1, parallel))]
        # MLX decode serializes on the GPU stream: one generation at a time,
        # further requests wait here (arrival order not strictly FIFO).
        self._lock = Lock()

    # -- generation ----------------------------------------------------------

    def generate(
        self, prompt: List[int], params: GenParams, slot_id: int = 0
    ) -> Iterator[Step]:
        """Serialize generations on the engine lock (MLX decode runs on a
        single GPU stream), then yield one Step per generated token. The
        final Step carries the finish_reason and no text."""
        with self._lock:
            yield from self._generate_locked(prompt, params, slot_id)

    def _generate_locked(
        self, prompt: List[int], params: GenParams, slot_id: int
    ) -> Iterator[Step]:
        """Yield one Step per generated token (blocking; run in a worker
        thread)."""
        if params.max_tokens > 0 and len(prompt) + params.max_tokens > self.ctx_size:
            raise ValueError(
                f"requested {params.max_tokens} tokens plus a prompt of"
                f" {len(prompt)} exceeds the context size of {self.ctx_size};"
                " reduce the prompt or raise --ctx-size"
            )
        if len(prompt) >= self.ctx_size:
            raise ValueError(
                f"prompt of {len(prompt)} tokens exceeds the context size of"
                f" {self.ctx_size}"
            )

        slot = self.slots[slot_id % len(self.slots)]
        reuse = self._reuse_len(slot, prompt)
        if reuse > 0:
            caches = slot.caches
            prompt_left = prompt[reuse:]
        else:
            caches = self.loaded.make_cache(self.cache_type)
            slot.caches = caches
            slot.tokens = []
            prompt_left = prompt

        max_gen = (
            params.max_tokens if params.max_tokens > 0 else self.ctx_size - len(prompt)
        )

        if params.seed is not None:
            mx.random.seed(params.seed)
        sampler = make_sampler(
            temp=params.temp,
            top_p=params.top_p,
            min_p=params.min_p,
            top_k=params.top_k,
            xtc_probability=params.xtc_probability,
            xtc_threshold=params.xtc_threshold,
        )
        logits_processors = make_logits_processors(
            logit_bias=params.logit_bias,
            repetition_penalty=params.repeat_penalty or None,
            repetition_context_size=max(1, params.repeat_last_n),
            presence_penalty=params.presence_penalty or None,
            frequency_penalty=params.frequency_penalty or None,
        )

        detok = self.tokenizer.detokenizer()
        tokens: List[int] = []
        finish_reason: Optional[str] = None
        stop_hit = False
        t_start = time.perf_counter()  # ref for the prefill timing
        total = len(prompt)  # tokens fed to the model so far

        # --- reasoning-budget state ---
        think_ids = (self.tokenizer.think_start_id, self.tokenizer.think_end_id)
        in_think = False
        think_count = 0
        wrapup_pending = False

        def forward_chunked(segment: List[int]):
            nonlocal total, caches
            out = None
            for cs in range(0, len(segment), self.prefill_step_size):
                chunk = segment[cs : cs + self.prefill_step_size]
                out = self.model(mx.array([chunk]), cache=caches)
                pos0 = total
                total += len(chunk)
                slot.tokens.extend(chunk)
                if self.runtime is not None:
                    self.runtime.note_tokens(len(chunk))
                    self._moe_observability(pos0, total)
                if cs + self.prefill_step_size < len(segment):
                    mx.eval(out)
            return out

        def step_one(tok: int):
            nonlocal total
            out = self.model(mx.array([[tok]]), cache=caches)
            pos0 = total
            total += 1
            slot.tokens.append(tok)
            if self.runtime is not None:
                self.runtime.note_tokens(1)
                self._moe_observability(pos0, total)
            return out

        logits = forward_chunked(prompt_left) if prompt_left else None
        if logits is None:
            # prompt exactly equals the cached prefix: nothing to continue
            # from — re-prefill from scratch to get valid logits
            caches = self.loaded.make_cache(self.cache_type)
            slot.caches = caches
            slot.tokens = []
            logits = forward_chunked(prompt)
        t_prefill_done = time.perf_counter()
        prefill_ms = (t_prefill_done - t_start) * 1000.0

        try:
            while True:
                logits = logits[:, -1, :]
                for proc in logits_processors:
                    logits = proc(tokens, logits)
                logprobs = logits - mx.logsumexp(logits, axis=-1, keepdims=True)
                tok = int(sampler(logprobs).item())

                # reasoning budget: watch the think phase and inject the
                # wrap-up once it exceeds the budget
                if (
                    params.reasoning_budget > 0
                    and not wrapup_pending
                    and think_ids[0] is not None
                ):
                    if tok == think_ids[0]:
                        in_think = True
                        think_count = 0
                    elif tok == think_ids[1]:
                        in_think = False
                    elif in_think:
                        think_count += 1
                        if think_count >= params.reasoning_budget - _WRAPUP_MARGIN:
                            wrapup_pending = True

                if wrapup_pending and self.wrapup_tokens:
                    wrapup_pending = False
                    in_think = False
                    for wtok in self.wrapup_tokens:
                        logits = step_one(wtok)
                        tokens.append(wtok)
                        text = detok.push(wtok)
                        yield Step(
                            token=wtok,
                            text=text,
                            prompt_tokens=len(prompt),
                            generated=len(tokens),
                            finish_reason=None,
                        )
                    continue

                if tok in self.tokenizer.eos_token_ids and not params.ignore_eos:
                    finish_reason = "stop"
                    break

                tokens.append(tok)
                text = detok.push(tok)

                if params.stop:
                    for s in params.stop:
                        if s and s in detok.text:
                            stop_hit = True
                            finish_reason = "stop"
                            break

                yield Step(
                    token=tok,
                    text=text,
                    prompt_tokens=len(prompt),
                    generated=len(tokens),
                    finish_reason=None,
                )

                if stop_hit or len(tokens) >= max_gen or total >= self.ctx_size:
                    finish_reason = finish_reason or (
                        "length" if not stop_hit else "stop"
                    )
                    break

                logits = step_one(tok)
        except GeneratorExit:
            # client aborted: the slot keeps its consistent prefix state
            if self.runtime is not None:
                self.runtime.maybe_print_stats()
            raise
        finally:
            detok.finalize()
            if self.runtime is not None:
                self.runtime.maybe_print_stats()

        if finish_reason is None:
            finish_reason = "length"
        now = time.perf_counter()
        yield Step(
            token=-1,
            text="",
            prompt_tokens=len(prompt),
            generated=len(tokens),
            finish_reason=finish_reason,
            prefill_ms=round(prefill_ms, 1),
            decode_ms=round((now - t_prefill_done) * 1000.0, 1),
        )

    # -- MoE expansion observability ------------------------------------------

    def _moe_observability(self, pos_start: int, pos_end: int):
        """Periodic stats line (every stats-every processed tokens, like
        LLAMA_MOE_EXPERT_STATS_EVERY) and the per-token kept-expert log when
        --moe-expert-log is set. Cheap no-op unless the window is full."""
        rt = self.runtime
        if rt is None:
            return
        if rt.config.expert_log:
            rt.flush_log(list(range(pos_start, pos_end)))
        if rt.stats_every > 0:
            rt.maybe_print_stats()

    # -- warmup ---------------------------------------------------------------

    def warmup(self, prompt_len: int = 16, gen_tokens: int = 4):
        """Short dummy generation before serving: pays the Metal kernel
        JIT / autotune / first-trace costs up front instead of on the first
        user request. Expansion observability is suspended."""
        rt = self.runtime
        if rt is not None:
            rt.suspended = True
        try:
            text_args = getattr(self.loaded.model, "text_args", None)
            vocab = getattr(text_args, "vocab_size", None) or 64
            prompt = [i % (vocab - 1) + 1 for i in range(prompt_len)]
            slot = self.slots[0]
            slot.caches = self.loaded.make_cache(self.cache_type)
            slot.tokens = []
            out = None
            for cs in range(0, len(prompt), self.prefill_step_size):
                chunk = prompt[cs : cs + self.prefill_step_size]
                out = self.model(mx.array([chunk]), cache=slot.caches)
                slot.tokens.extend(chunk)
                if cs + self.prefill_step_size < len(prompt):
                    mx.eval(out)
            for _ in range(gen_tokens):
                out = self.model(mx.array([[prompt[-1]]]), cache=slot.caches)
                mx.eval(out)
            slot.caches = None
            slot.tokens = []
        finally:
            if rt is not None:
                rt.suspended = False
                rt.reset_stats()

    # -- cache management ----------------------------------------------------

    def _reuse_len(self, slot: _Slot, prompt: List[int]) -> int:
        """Longest cached prefix that is exactly a prefix of the new prompt.
        The recurrent linear-attention state is only valid for exact
        continuations, so anything else re-prefills."""
        cached = slot.tokens
        if not cached or slot.caches is None:
            return 0
        if len(cached) >= len(prompt):
            return 0
        if prompt[: len(cached)] != cached:
            return 0
        if len(cached) < 2:
            return 0
        return len(cached)

    # -- info ----------------------------------------------------------------

    def moe_stats_summary(self) -> Optional[dict]:
        if self.runtime is None:
            return None
        return {
            "expansion_active": self.runtime.active or self.runtime.reduce_active,
            "experts_per_token_max": self.runtime.n if self.runtime.active else None,
            "native_top_k": self.runtime.k_native,
            "layer_range": (
                [self.runtime.layer_start, self.runtime.layer_end]
                if self.runtime.active
                else None
            ),
        }
