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
    """Incremental detokenizer: decode the pending buffer, emit the stable
    prefix, hold back anything that still looks like a split UTF-8 tail."""

    def __init__(self, tokenizer):
        self._tokenizer = tokenizer
        self._pending: List[int] = []
        self._pending_text = ""
        self.text = ""  # full stable text emitted so far

    def reset(self):
        self._pending = []
        self._pending_text = ""
        self.text = ""

    def push(self, token: int) -> str:
        """Add one token, return the newly emitted text (may be empty)."""
        self._pending.append(int(token))
        try:
            new_text = self._tokenizer.decode(
                self._pending, skip_special_tokens=True
            )
        except Exception:
            new_text = self._tokenizer.decode(
                self._pending, skip_special_tokens=True, errors="ignore"
            )
        # incomplete utf-8 tail: hold the bytes in the pending buffer
        if new_text.endswith("\ufffd") and not self._pending_text.endswith("\ufffd"):
            return ""
        if len(new_text) < len(self._pending_text):
            # decode shrank (rare): keep buffering
            return ""
        emitted = new_text[len(self._pending_text):]
        if not emitted:
            return ""
        self._pending_text = new_text
        self.text += emitted
        self._pending = []
        return emitted

    def finalize(self) -> str:
        """Flush any held-back text (call at the end of generation)."""
        if self._pending:
            try:
                rest = self._tokenizer.decode(
                    self._pending, skip_special_tokens=True
                )
            except Exception:
                rest = ""
            if rest and not rest.endswith("\ufffd"):
                self.text += rest
                self._pending_text += rest
            self._pending = []
        return self.text


class Tokenizer:
    """AgrimlxO tokenizer wrapper."""

    def __init__(self, model_path: Union[str, Path], chat_template_override: str = ""):
        self._tokenizer = AutoTokenizer.from_pretrained(
            str(model_path), trust_remote_code=False
        )
        self._chat_template_override = chat_template_override or None

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
