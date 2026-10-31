# SPDX-License-Identifier: Apache-2.0
"""Tokenizer wrapper and incremental detokenizer for AgrimlxO.

Thin layer over transformers AutoTokenizer with:
  * chat-template application (enable_thinking passthrough for Qwen3.x),
  * eos id set assembled from tokenizer/generation_config,
  * think-token detection (<think> / </think>) for the reasoning budget,
  * a monotonic incremental detokenizer (prefix-diff over the pending token
    buffer, holding back incomplete UTF-8 sequences).
"""

import json
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Set, Union

from transformers import AutoTokenizer


class Detokenizer:
    """Incremental detokenizer: decode the pending token buffer, emit the
    stable delta, hold back anything that still looks like a split UTF-8
    tail.

    Invariant: ``_pending`` holds only the tokens whose text has not been
    emitted yet; ``_pending_decode`` is the last decode OF THAT BUFFER. The
    emitted delta is the longest-common-prefix difference between the two
    decodes of the same buffer — never a slice of the global text.
    """

    def __init__(self, tokenizer):
        self._tokenizer = tokenizer
        self._pending: List[int] = []
        self._pending_decode = ""  # decode of the current pending buffer
        self.text = ""  # full stable text emitted so far

    def reset(self):
        self._pending = []
        self._pending_decode = ""
        self.text = ""

    def _decode_pending(self) -> str:
        try:
            return self._tokenizer.decode(
                self._pending, skip_special_tokens=True
            )
        except TypeError:
            return self._tokenizer.decode(self._pending)

    def push(self, token: int) -> str:
        """Add one token, return the newly emitted text (may be empty)."""
        self._pending.append(int(token))
        new_text = self._decode_pending()

        # incomplete utf-8 tail: hold the bytes in the pending buffer
        if new_text.endswith("\ufffd"):
            return ""

        prev = self._pending_decode
        # longest common prefix between the previous and new buffer decodes
        common = 0
        limit = min(len(prev), len(new_text))
        while common < limit and prev[common] == new_text[common]:
            common += 1
        emitted = new_text[common:]

        self._pending_decode = new_text
        if emitted:
            self.text += emitted
            self._pending = []  # fully flushed
            self._pending_decode = ""
        return emitted

    def finalize(self) -> str:
        """Flush any held-back text (call at the end of generation); returns
        the remainder emitted now ("" when everything was already flushed)."""
        rest = ""
        if self._pending:
            decoded = self._decode_pending()
            if decoded and not decoded.endswith("\ufffd"):
                rest = decoded
                self.text += rest
            self._pending = []
            self._pending_decode = ""
        return rest


def _read_template_file(model_path: Path) -> Optional[str]:
    """Standalone template serializations some checkpoints ship instead of
    an inline tokenizer_config entry: ``chat_template.jinja`` (raw, the
    recent transformers format) or ``chat_template.json`` ({"chat_template":
    ...} or the multi-template list). AutoTokenizer does not read these on
    older transformers, leaving the model without a chat template."""
    jinja = model_path / "chat_template.jinja"
    if jinja.exists():
        try:
            text = jinja.read_text()
            if text.strip():
                return text
        except Exception:
            pass
    js = model_path / "chat_template.json"
    if js.exists():
        try:
            data = json.loads(js.read_text())
        except Exception:
            return None
        if isinstance(data, dict) and isinstance(data.get("chat_template"), str):
            return data["chat_template"]
        if isinstance(data, list):
            for t in data:
                if isinstance(t, dict) and t.get("name") in (None, "default"):
                    return t.get("template")
            for t in data:
                if isinstance(t, dict) and isinstance(t.get("template"), str):
                    return t["template"]
    return None


class Tokenizer:
    """AgrimlxO tokenizer wrapper."""

    def __init__(self, model_path: Union[str, Path], chat_template_override: str = ""):
        self._tokenizer = AutoTokenizer.from_pretrained(
            str(model_path), trust_remote_code=False
        )
        self._chat_template_override = chat_template_override or None
        if self._tokenizer.chat_template is None:
            tmpl = _read_template_file(Path(model_path))
            if tmpl:
                self._tokenizer.chat_template = tmpl

        self.eos_token_ids: Set[int] = set()
        if self._tokenizer.eos_token_id is not None:
            self.eos_token_ids.add(int(self._tokenizer.eos_token_id))
        # generation_config eos ids (lists included)
        gen_cfg = Path(model_path) / "generation_config.json"
        if gen_cfg.exists():
            try:
                with open(gen_cfg) as f:
                    gc = json.load(f)
                v = gc.get("eos_token_id")
                if isinstance(v, list):
                    self.eos_token_ids.update(int(x) for x in v)
                elif v is not None:
                    self.eos_token_ids.add(int(v))
            except Exception:
                pass
        for special in ("<|im_end|>", "<|endoftext|>"):
            tid = self._tokenizer.convert_tokens_to_ids(special)
            if tid is not None and tid >= 0:
                self.eos_token_ids.add(int(tid))

        # think tokens (Qwen3.x hybrid thinking)
        self.think_start_id = self._single_token_id("<think>")
        self.think_end_id = self._single_token_id("</think>")

    def _single_token_id(self, text: str) -> Optional[int]:
        try:
            ids = self._tokenizer.encode(text, add_special_tokens=False)
        except Exception:
            return None
        if len(ids) == 1:
            return int(ids[0])
        return None

    @property
    def has_chat_template(self) -> bool:
        return (
            self._tokenizer.chat_template is not None
            or self._chat_template_override is not None
        )

    def apply_chat_template(
        self, messages: Sequence[dict], tokenize: bool = True, **kwargs
    ):
        kwargs.setdefault("return_dict", False)
        kwargs.setdefault("add_generation_prompt", True)
        if self._chat_template_override is not None:
            import jinja2  # noqa: F401  (transformers pulls it in)

            tmpl = self._chat_template_override
            rendered = self._tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                chat_template=tmpl,
                **{k: v for k, v in kwargs.items() if k != "chat_template"},
            )
            if tokenize:
                return self._tokenizer.encode(rendered, add_special_tokens=False)
            return rendered
        return self._tokenizer.apply_chat_template(
            messages, tokenize=tokenize, **kwargs
        )

    def encode(self, text: str) -> List[int]:
        return self._tokenizer.encode(text, add_special_tokens=False)

    def detokenizer(self) -> Detokenizer:
        return Detokenizer(self._tokenizer)
