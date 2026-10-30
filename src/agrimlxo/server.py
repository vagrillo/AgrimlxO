# SPDX-License-Identifier: Apache-2.0
"""AgrimlxO HTTP server (FastAPI, uvicorn).

OpenAI-compatible endpoints in the llama.cpp-server style:
  POST /v1/chat/completions   chat (stream via SSE; reasoning_content split)
  POST /v1/completions        raw completion (stream via SSE)
  GET  /v1/models             model list (the served model)
  GET  /health                health check
  GET  /props                 llama.cpp-style server properties (incl. MoE state)
  GET  /slots                 slot occupancy (llama.cpp style)
  GET  /metrics               minimal Prometheus metrics
  GET  /                      chat interface (llama.cpp-webui style)

The generation engine is single-stream (MLX serializes decode on the GPU
stream): requests beyond the free slots queue in arrival order. Slot
bookkeeping mirrors llama.cpp's /slots.
"""

import asyncio
import json
import queue
import sys
import threading
import time
import uuid
from typing import Any, Dict, List, Optional

from fastapi import Depends, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, Response, StreamingResponse
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import BaseModel

from ._version import __version__
from .engine import ContextOverflowError, Engine, GenParams, Step, ThinkSplit
from .tools_registry import ToolRegistry, parse_tool_calls


# -- request schemas (OpenAI-compatible subset) --------------------------------


class ChatMessage(BaseModel):
    role: str
    content: Any = None
    name: Optional[str] = None


class ChatCompletionRequest(BaseModel):
    model: Optional[str] = None
    messages: List[ChatMessage]
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    top_k: Optional[int] = None
    min_p: Optional[float] = None
    max_tokens: Optional[int] = None
    max_completion_tokens: Optional[int] = None
    stop: Optional[Any] = None
    stream: bool = False
    stream_options: Optional[dict] = None
    presence_penalty: Optional[float] = None
    frequency_penalty: Optional[float] = None
    repetition_penalty: Optional[float] = None
    seed: Optional[int] = None
    logit_bias: Optional[dict] = None
    ignore_eos: Optional[bool] = False
    reasoning_budget: Optional[int] = None
    chat_template_kwargs: Optional[dict] = None
    enable_thinking: Optional[bool] = None
    tools: Optional[bool] = None
    n: Optional[int] = 1


class CompletionRequest(BaseModel):
    model: Optional[str] = None
    prompt: Any
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    top_k: Optional[int] = None
    min_p: Optional[float] = None
    max_tokens: Optional[int] = None
    stop: Optional[Any] = None
    stream: bool = False
    stream_options: Optional[dict] = None
    presence_penalty: Optional[float] = None
    frequency_penalty: Optional[float] = None
    repetition_penalty: Optional[float] = None
    seed: Optional[int] = None
    logit_bias: Optional[dict] = None
    ignore_eos: Optional[bool] = False
    echo: bool = False


# -- server state ---------------------------------------------------------------


class ServerState:
    def __init__(self, args, engine: Engine, tokenizer):
        import os
        from pathlib import Path

        self.args = args
        self.engine = engine
        self.tokenizer = tokenizer
        model = str(args.model or "qwen3.x-35b-moe")
        # local dir -> its name; HF repo id stays intact
        self.model_name = Path(model).name if os.path.isdir(model) else model
        self.tools = ToolRegistry(getattr(args, "tools_dir", "./tools"))
        self.vision_processor = None
        if getattr(engine.loaded.model, "vision_tower", None) is not None:
            from .processor import VisionProcessor

            m = engine.loaded.model
            vcfg = getattr(m.args, "vision_config", None) or {}
            try:
                self.vision_processor = VisionProcessor(
                    engine.loaded.path,
                    image_token_id=getattr(m.args, "image_token_id", None),
                    spatial_merge_size=int(vcfg.get("spatial_merge_size", 2)),
                )
            except Exception as e:
                # mai fatale: il server parte text-only con un messaggio chiaro
                import logging

                self.vision_processor_error = f"{type(e).__name__}: {e}"
                logging.getLogger(__name__).warning(
                    "vision processor non disponibile (%s); il server parte"
                    " text-only e le immagini risponderanno con un errore"
                    " esplicito",
                    self.vision_processor_error,
                )
                self.vision_processor = None
        self.active_slots: List[Optional[float]] = [None] * max(1, args.parallel)
        self.counters = {"requests": 0, "prompt_tokens": 0, "generated_tokens": 0}

    def gen_params(self, req, default_max_tokens: int = -1) -> GenParams:
        a = self.args
        stop = req.stop if req.stop is not None else []
        if isinstance(stop, str):
            stop = [stop]
        max_tokens = req.max_tokens
        if getattr(req, "max_completion_tokens", None) is not None:
            max_tokens = req.max_completion_tokens
        if max_tokens is None:
            max_tokens = a.n_predict
        return GenParams(
            temp=req.temperature if req.temperature is not None else a.temp,
            top_k=req.top_k if req.top_k is not None else a.top_k,
            top_p=req.top_p if req.top_p is not None else a.top_p,
            min_p=req.min_p if req.min_p is not None else a.min_p,
            xtc_probability=a.xtc_probability,
            xtc_threshold=a.xtc_threshold,
            repeat_penalty=(
                req.repetition_penalty
                if req.repetition_penalty is not None
                else a.repeat_penalty
            ),
            repeat_last_n=a.repeat_last_n,
            presence_penalty=(
                req.presence_penalty
                if req.presence_penalty is not None
                else a.presence_penalty
            ),
            frequency_penalty=(
                req.frequency_penalty
                if req.frequency_penalty is not None
                else a.frequency_penalty
            ),
            logit_bias=getattr(req, "logit_bias", None),
            seed=req.seed if req.seed is not None else a.seed,
            max_tokens=max_tokens,
            stop=list(stop),
            ignore_eos=bool(getattr(req, "ignore_eos", False)),
            reasoning_budget=(
                req.reasoning_budget
                if getattr(req, "reasoning_budget", None) is not None
                else a.reasoning_budget
            ),
            emit_prefill_stats=bool(getattr(req, "stream", False)),
        )

    def prompt_tokens_for_chat(self, messages: List[dict], template_kwargs: dict,
                               tools_schema: Optional[List[dict]] = None):
        """Token ids for the request, or (with images) the full processor
        inputs dict. Raises HTTPException(400) when images arrive but the
        checkpoint has no vision tower."""
        kwargs = dict(template_kwargs or {})
        if kwargs.get("enable_thinking") is None:
            kwargs.pop("enable_thinking", None)
        flat = []
        image_urls: List[str] = []
        for m in messages:
            c = m.get("content")
            if isinstance(c, list):
                texts = []
                for part in c:
                    if isinstance(part, dict) and part.get("type") == "text":
                        texts.append(str(part.get("text", "")))
                    elif isinstance(part, dict) and part.get("type") == "image_url":
                        url = part.get("image_url", {})
                        url = url.get("url", "") if isinstance(url, dict) else str(url)
                        if url:
                            image_urls.append(url)
                m = dict(m)
                m["content"] = "\n".join(t for t in texts if t)
            flat.append(m)

        has_vision = getattr(self.engine.loaded.model, "vision_tower", None) is not None
        if image_urls and not has_vision:
            raise HTTPException(
                400,
                f"{len(image_urls)} image(s) received but this checkpoint has"
                " no vision tower (text-only model). Text attachments work.",
            )
        if image_urls and has_vision and self.vision_processor is None:
            raise HTTPException(
                400,
                "processor vision non caricato su questo ambiente — motivo: "
                + str(getattr(self, "vision_processor_error", "sconosciuto"))
                + " | rimedi: pip install torchvision (abilita il processor"
                " completo) oppure incolla questo messaggio",
            )
        if image_urls and has_vision:
            rendered = self.tokenizer.apply_chat_template(flat, tokenize=False, **kwargs)
            prepared = self.vision_processor.prepare(rendered, image_urls)
            return prepared
        if tools_schema:
            kwargs["tools"] = tools_schema
        return self.tokenizer.apply_chat_template(flat, **kwargs)

    def slot_acquire(self) -> "SlotHandle":
        for i, s in enumerate(self.active_slots):
            if s is None:
                self.active_slots[i] = time.time()
                return SlotHandle(self, i)
        return SlotHandle(self, 0)  # queue behind slot 0 (single-stream decode)

    def slot_release(self, i: int):
        if 0 <= i < len(self.active_slots):
            self.active_slots[i] = None


class SlotHandle:
    def __init__(self, state: ServerState, slot_id: int):
        self.state = state
        self.id = slot_id
        self.released = False

    def release(self):
        if not self.released:
            self.state.slot_release(self.id)
            self.released = True


def _sse(data: str) -> str:
    return f"data: {data}\n\n"


async def _run_generation(
    state: ServerState, tokens: List[int], params: GenParams, slot_id: int,
    images: Optional[dict] = None,
):
    """Async generator of Steps; the blocking engine runs in a worker thread
    and forwards steps through a queue. The final Step carries finish_reason.
    Raises on engine errors (e.g. context overflow)."""
    loop = asyncio.get_running_loop()
    q: "queue.Queue" = queue.Queue()

    def run():
        try:
            for step in state.engine.generate(tokens, params, slot_id, images):
                q.put(("step", step))
            q.put(("done", None))
        except ContextOverflowError as e:
            q.put(("overflow", str(e)))
        except Exception as e:
            q.put(("error", e))

    threading.Thread(target=run, daemon=True).start()

    while True:
        kind, payload = await loop.run_in_executor(None, q.get)
        if kind == "step":
            step: Step = payload
            if step.finish_reason is not None:
                yield step, True
                return
            yield step, False
        elif kind == "overflow":
            raise HTTPException(400, payload)
        elif kind == "error":
            raise payload
        else:
            return


def _make_reasoning_feed(split_enabled: bool):
    """raw delta -> (content_delta, reasoning_delta); the returned callable
    also exposes flush() for the end of the stream."""
    if not split_enabled:
        def passthrough(raw):
            return (raw, "")

        passthrough.flush = lambda: ("", "")
        return passthrough
    ts = ThinkSplit()

    def feed(raw: str):
        content, reasoning = "", ""
        for kind, delta in ts.push(raw):
            if kind == "reasoning":
                reasoning += delta
            else:
                content += delta
        return content, reasoning

    def flush():
        content, reasoning = "", ""
        for kind, delta in ts.flush():
            if kind == "reasoning":
                reasoning += delta
            else:
                content += delta
        return content, reasoning

    feed.flush = flush
    return feed


def _tools_enabled(state: "ServerState", req) -> bool:
    if getattr(req, "tools", None) is not None:
        return bool(req.tools)
    return bool(getattr(state.args, "tools", False))


def _log_request(params: GenParams, final: Step):
    """One console line per completed request: engine timings + sampling
    params, so the server's own stderr tells the whole story."""
    if final is None:
        return
    prefill = (
        f"prefill {final.prompt_tokens} tok in {final.prefill_ms:.0f} ms"
        f" ({final.prompt_tokens / max(1e-6, final.prefill_ms / 1000):.0f} tok/s)"
        if final.prefill_ms
        else "prefill n/a"
    )
    decode = (
        f"decode {final.generated} tok in {final.decode_ms / 1000:.1f} s"
        f" ({final.generated / max(1e-6, final.decode_ms / 1000):.1f} tok/s)"
        if final.decode_ms
        else "decode n/a"
    )
    print(
        f"req: {prefill} | {decode} | {final.finish_reason} |"
        f" temp={params.temp} top_p={params.top_p} top_k={params.top_k}"
        f" min_p={params.min_p} seed={params.seed}",
        file=sys.stderr,
        flush=True,
    )


def _usage(
    prompt_tokens: int,
    generated: int,
    prefill_ms: Optional[float] = None,
    decode_ms: Optional[float] = None,
) -> dict:
    usage = {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": generated,
        "total_tokens": prompt_tokens + generated,
    }
    if prefill_ms:
        usage["prefill_ms"] = prefill_ms
        usage["prompt_tps"] = round(prompt_tokens / (prefill_ms / 1000.0), 1)
    if decode_ms and generated:
        usage["decode_ms"] = decode_ms
        usage["completion_tps"] = round(generated / (decode_ms / 1000.0), 1)
    return usage


def _chunk(rid, created, model, delta, finish=None, experts=None, experts_mean=None):
    chunk = {
        "id": rid,
        "object": "chat.completion.chunk",
        "created": created,
        "model": model,
        "choices": [{"index": 0, "delta": delta, "finish_reason": finish}],
    }
    if experts is not None:
        chunk["experts"] = {
            "last": round(experts, 2),
            "mean": round(experts_mean, 2) if experts_mean is not None else None,
        }
    return chunk


# -- app factory ----------------------------------------------------------------


def create_app(state: ServerState) -> FastAPI:
    app = FastAPI(title="AgrimlxO", version=__version__, docs_url=None, redoc_url=None)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=False,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    bearer = HTTPBearer(auto_error=False)

    def auth(credentials: Optional[HTTPAuthorizationCredentials] = Depends(bearer)):
        key = state.args.api_key
        if not key:
            return
        if credentials is None or credentials.credentials != key:
            raise HTTPException(status_code=401, detail="invalid API key")

    # ---- chat interface ----

    @app.get("/", include_in_schema=False)
    def index():
        from pathlib import Path

        return FileResponse(
            Path(__file__).parent / "webui" / "index.html",
            headers={"Cache-Control": "no-store"},
        )

    # ---- health / info ----

    @app.get("/health")
    def health():
        return {"status": "ok"}

    @app.get("/version")
    def version():
        return {"version": __version__}

    @app.get("/props")
    def props(_: None = Depends(auth)):
        a = state.args
        moe = state.engine.moe_stats_summary() or {}
        return {
            "default_generation_settings": {
                "n_ctx": max(1, a.ctx_size // max(1, a.parallel)),
                "n_predict": a.n_predict,
                "temperature": a.temp,
                "top_k": a.top_k,
                "top_p": a.top_p,
                "min_p": a.min_p,
            },
            "total_slots": max(1, a.parallel),
            "model_path": a.model or "",
            "model_name": state.model_name,
            "has_chat_template": state.tokenizer.has_chat_template,
            "reasoning_format": a.reasoning_format,
            "moe_expansion": moe,
            "tools": state.tools.names() if state.tools else [],
            "agrimlxo_version": __version__,
            "server": "AgrimlxO",
        }

    @app.get("/v1/models")
    def v1_models(_: None = Depends(auth)):
        return {
            "object": "list",
            "data": [{"id": state.model_name, "object": "model", "owned_by": "agrimlxo"}],
        }

    @app.get("/slots")
    def slots(_: None = Depends(auth)):
        out = [
            {
                "id": i,
                "busy": s is not None,
                "ctx": max(1, state.args.ctx_size // max(1, state.args.parallel)),
            }
            for i, s in enumerate(state.active_slots)
        ]
        return {"slots": out}

    @app.get("/metrics")
    def metrics(_: None = Depends(auth)):
        c = state.counters
        lines = [
            "# TYPE agrimlxo_requests_total counter",
            f"agrimlxo_requests_total {c['requests']}",
            "# TYPE agrimlxo_prompt_tokens_total counter",
            f"agrimlxo_prompt_tokens_total {c['prompt_tokens']}",
            "# TYPE agrimlxo_generated_tokens_total counter",
            f"agrimlxo_generated_tokens_total {c['generated_tokens']}",
        ]
        moe = state.engine.moe_stats_summary()
        if moe:
            lines += [
                "# TYPE agrimlxo_moe_expansion info",
                "agrimlxo_moe_expansion{active=\"%s\",native_top_k=\"%s\",budget=\"%s\"} 1"
                % (
                    str(bool(moe.get("expansion_active"))).lower(),
                    moe.get("native_top_k"),
                    moe.get("experts_per_token_max"),
                ),
            ]
        return Response(content="\n".join(lines) + "\n", media_type="text/plain")

    # ---- chat completions ----

    @app.post("/v1/chat/completions")
    async def chat_completions(req: ChatCompletionRequest, _: None = Depends(auth)):
        if not state.tokenizer.has_chat_template:
            raise HTTPException(400, "the tokenizer has no chat template")
        messages = [m.model_dump(exclude_none=True) for m in req.messages]
        template_kwargs = dict(req.chat_template_kwargs or {})
        if req.enable_thinking is not None:
            template_kwargs["enable_thinking"] = req.enable_thinking
        try:
            prepared = state.prompt_tokens_for_chat(messages, template_kwargs)
        except Exception as e:
            raise HTTPException(400, f"chat template failed: {e}")

        params = state.gen_params(req)
        slot = state.slot_acquire()
        images = None
        if isinstance(prepared, dict):
            images = {
                "pixel_values": prepared["pixel_values"],
                "image_grid_thw": prepared["image_grid_thw"],
            }
            tokens = prepared["input_ids"]
        else:
            tokens = prepared
        state.counters["requests"] += 1
        state.counters["prompt_tokens"] += len(tokens)
        split_enabled = (
            state.args.reasoning_format == "auto"
            and template_kwargs.get("enable_thinking") is not False
        )

        try:
            if req.stream:
                return StreamingResponse(
                    _chat_stream(state, req, tokens, params, slot, split_enabled, images),
                    media_type="text/event-stream",
                    headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
                )
            return await _chat_once(
                state, req, tokens, params, slot, split_enabled, images
            )
        except HTTPException:
            slot.release()
            raise

    async def _chat_once(state, req, tokens, params, slot, split_enabled, images=None, messages=None, template_kwargs=None):
        rid = f"chatcmpl-{uuid.uuid4().hex[:24]}"
        content = ""
        reasoning = ""
        final: Optional[Step] = None
        msgs = list(messages or [])
        try:
            for _tool_iter in range(3):
                feed = _make_reasoning_feed(split_enabled)
                phase_content = ""
                async for step, is_final in _run_generation(
                    state, tokens, params, slot.id, images
                ):
                    if is_final:
                        final = step
                        break
                    c, r = feed(step.text)
                    phase_content += c
                    reasoning += r
                c, r = feed.flush()
                phase_content += c
                content += phase_content

                calls = (
                    parse_tool_calls(phase_content)
                    if state.tools and _tools_enabled(state, req)
                    else []
                )
                if not calls:
                    break
                tool_text = "\n\n"
                for call in calls:
                    result = state.tools.execute(call["name"], call["arguments"])
                    tool_text += (
                        f"<tool_call>{json.dumps({'name': call['name'], 'arguments': call['arguments']})}</tool_call>\n"
                        f"<tool_result>{result}</tool_result>\n\n"
                    )
                content += tool_text
                msgs.append({"role": "assistant", "content": phase_content})
                msgs.append({"role": "user", "content": tool_text})
                tools_schema = (
                    state.tools.openai_schema()
                    if state.tools and _tools_enabled(state, req) else None
                )
                prepared = state.prompt_tokens_for_chat(
                    msgs, template_kwargs or {}, tools_schema=tools_schema
                )
                if isinstance(prepared, dict):
                    tokens = prepared["input_ids"]
                    images = {"pixel_values": prepared["pixel_values"],
                              "image_grid_thw": prepared["image_grid_thw"]}
                else:
                    tokens, images = prepared, None
                final = None
        finally:
            _log_request(params, final)
            slot.release()

        usage = _usage(
            final.prompt_tokens if final else len(tokens),
            final.generated if final else 0,
            prefill_ms=final.prefill_ms if final else None,
            decode_ms=final.decode_ms if final else None,
        )
        state.counters["generated_tokens"] += usage["completion_tokens"]
        msg: Dict[str, Any] = {"role": "assistant", "content": content}
        if reasoning:
            msg["reasoning_content"] = reasoning
        return {
            "id": rid,
            "object": "chat.completion",
            "created": int(time.time()),
            "model": state.model_name,
            "choices": [
                {
                    "index": 0,
                    "message": msg,
                    "finish_reason": final.finish_reason if final else "stop",
                }
            ],
            "usage": usage,
        }

    async def _chat_stream(state, req, tokens, params, slot, split_enabled,
                           images=None, messages=None, template_kwargs=None):
        rid = f"chatcmpl-{uuid.uuid4().hex[:24]}"
        created = int(time.time())
        include_usage = bool((req.stream_options or {}).get("include_usage"))
        final: Optional[Step] = None
        msgs = list(messages or [])
        try:
            for _tool_iter in range(3):
                feed = _make_reasoning_feed(split_enabled)
                phase_content = ""
                async for step, is_final in _run_generation(
                    state, tokens, params, slot.id, images
                ):
                    if is_final:
                        final = step
                        break
                    if step.token == -2:
                        # prefill stats, before the first token
                        yield _sse(
                            json.dumps(
                                {
                                    "id": rid,
                                    "object": "chat.completion.chunk",
                                    "created": created,
                                    "model": state.model_name,
                                    "choices": [],
                                    "usage_partial": {
                                        "prompt_tokens": step.prompt_tokens,
                                        "prefill_ms": step.prefill_ms,
                                        "prompt_tps": round(
                                            step.prompt_tokens
                                            / max(1e-6, (step.prefill_ms or 1) / 1000.0),
                                            1,
                                        ),
                                    },
                                }
                            )
                        )
                        continue
                    c, r = feed(step.text)
                    phase_content += (c or "") + (r or "")
                    if r:
                        yield _sse(
                            json.dumps(
                                _chunk(
                                    rid, created, state.model_name,
                                    {"content": None, "reasoning_content": r},
                                    experts=step.experts,
                                    experts_mean=step.experts_mean,
                                )
                            )
                        )
                    if c:
                        yield _sse(
                            json.dumps(
                                _chunk(
                                    rid, created, state.model_name, {"content": c},
                                    experts=step.experts,
                                    experts_mean=step.experts_mean,
                                )
                            )
                        )
                    if not c and not r and step.experts is not None:
                        yield _sse(
                            json.dumps(
                                _chunk(
                                    rid, created, state.model_name, {},
                                    experts=step.experts,
                                    experts_mean=step.experts_mean,
                                )
                            )
                        )
                c, r = feed.flush()
                phase_content += c
                if r:
                    yield _sse(
                        json.dumps(
                            _chunk(
                                rid, created, state.model_name,
                                {"content": None, "reasoning_content": r},
                            )
                        )
                    )
                if c:
                    yield _sse(
                        json.dumps(_chunk(rid, created, state.model_name, {"content": c}))
                    )

                # ---- tool phase-end: execute, feed back, loop ----
                calls = (
                    parse_tool_calls(phase_content)
                    if state.tools and _tools_enabled(state, req)
                    else []
                )
                if not calls:
                    break
                tool_text = "\n\n"
                for call in calls:
                    result = state.tools.execute(call["name"], call["arguments"])
                    tool_text += (
                        f"<tool_call>{json.dumps({'name': call['name'], 'arguments': call['arguments']})}</tool_call>\n"
                        f"<tool_result>{result}</tool_result>\n\n"
                    )
                yield _sse(
                    json.dumps(_chunk(rid, created, state.model_name, {"content": tool_text}))
                )
                msgs.append({"role": "assistant", "content": phase_content})
                msgs.append({"role": "user", "content": tool_text})
                tools_schema = (
                    state.tools.openai_schema()
                    if state.tools and _tools_enabled(state, req) else None
                )
                prepared = state.prompt_tokens_for_chat(
                    msgs, template_kwargs or {}, tools_schema=tools_schema
                )
                if isinstance(prepared, dict):
                    tokens = prepared["input_ids"]
                    images = {"pixel_values": prepared["pixel_values"],
                              "image_grid_thw": prepared["image_grid_thw"]}
                else:
                    tokens, images = prepared, None
                final = None

            yield _sse(
                json.dumps(
                    _chunk(
                        rid,
                        created,
                        state.model_name,
                        {},
                        finish=final.finish_reason if final else "stop",
                    )
                )
            )
            if include_usage and final is not None:
                state.counters["generated_tokens"] += final.generated
                yield _sse(
                    json.dumps(
                        {
                            "id": rid,
                            "object": "chat.completion.chunk",
                            "created": created,
                            "model": state.model_name,
                            "choices": [],
                            "usage": _usage(
                                final.prompt_tokens,
                                final.generated,
                                prefill_ms=final.prefill_ms,
                                decode_ms=final.decode_ms,
                            ),
                        }
                    )
                )
        except ContextOverflowError as e:
            yield _sse(
                json.dumps(_chunk(rid, created, state.model_name,
                                  {"content": f"⚠ {e}"}))
            )
            yield _sse(
                json.dumps(_chunk(rid, created, state.model_name, {}, finish="error"))
            )
        except Exception as e:
            yield _sse(
                json.dumps(_chunk(rid, created, state.model_name,
                                  {"content": f"⚠ errore motore: {e}"}))
            )
            yield _sse(
                json.dumps(_chunk(rid, created, state.model_name, {}, finish="error"))
            )
        except GeneratorExit:
            raise
        finally:
            _log_request(params, final)
            slot.release()

    # ---- raw completions ----

    @app.post("/v1/completions")
    async def completions(req: CompletionRequest, _: None = Depends(auth)):
        texts = [req.prompt] if isinstance(req.prompt, str) else list(req.prompt)
        params = state.gen_params(req)
        state.counters["requests"] += len(texts)
        rid = f"cmpl-{uuid.uuid4().hex[:24]}"

        async def run_one(text: str) -> tuple:
            tokens = state.tokenizer.encode(text)
            state.counters["prompt_tokens"] += len(tokens)
            slot = state.slot_acquire()
            content = ""
            final: Optional[Step] = None
            try:
                async for step, is_final in _run_generation(
                state, tokens, params, slot.id, None
            ):
                    if is_final:
                        final = step
                        break
                    content += step.text
            finally:
                slot.release()
            state.counters["generated_tokens"] += final.generated if final else 0
            return content, final

        if not req.stream:
            choices = []
            total_prompt = total_gen = 0
            for i, text in enumerate(texts):
                content, final = await run_one(text)
                total_prompt += final.prompt_tokens if final else 0
                total_gen += final.generated if final else 0
                choices.append(
                    {
                        "index": i,
                        "text": content,
                        "finish_reason": final.finish_reason if final else "stop",
                    }
                )
            return {
                "id": rid,
                "object": "text_completion",
                "created": int(time.time()),
                "model": state.model_name,
                "choices": choices,
                "usage": _usage(total_prompt, total_gen),
            }

        async def stream_all():
            created = int(time.time())
            for i, text in enumerate(texts):
                tokens = state.tokenizer.encode(text)
                state.counters["prompt_tokens"] += len(tokens)
                slot = state.slot_acquire()
                final: Optional[Step] = None
                try:
                    async for step, is_final in _run_generation(
                state, tokens, params, slot.id, None
            ):
                        if is_final:
                            final = step
                            break
                        yield _sse(
                            json.dumps(
                                {
                                    "id": rid,
                                    "object": "text_completion.chunk",
                                    "created": created,
                                    "model": state.model_name,
                                    "choices": [
                                        {"index": i, "text": step.text, "finish_reason": None}
                                    ],
                                }
                            )
                        )
                finally:
                    slot.release()
                state.counters["generated_tokens"] += final.generated if final else 0
                yield _sse(
                    json.dumps(
                        {
                            "id": rid,
                            "object": "text_completion.chunk",
                            "created": created,
                            "model": state.model_name,
                            "choices": [
                                {
                                    "index": i,
                                    "text": "",
                                    "finish_reason": final.finish_reason if final else "stop",
                                }
                            ],
                        }
                    )
                )
            yield "data: [DONE]\n\n"

        return StreamingResponse(
            stream_all(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    return app


def run_server(args, engine: Engine, tokenizer):
    state = ServerState(args, engine, tokenizer)
    app = create_app(state)

    import uvicorn

    print(
        f"listening on http://{args.host}:{args.port}  (slots:"
        f" {max(1, args.parallel)}, ctx: {args.ctx_size})",
        flush=True,
    )
    uvicorn.run(app, host=args.host, port=args.port, log_level="info" if args.verbose else "warning")
