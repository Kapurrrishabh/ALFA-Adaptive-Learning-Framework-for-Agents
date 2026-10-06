"""An open-source language model as the writer, never the source of a number.

It turns what our models measured into a few plain sentences. Every figure it writes is checked by
`llm.verify_numbers` against the facts it was given; a note with an unsupported figure is discarded
and the deterministic note shown instead, with the reason. Greedy decoding, so one set of facts
always gives one note.

Same interface as `llm.LLMClient` (`create`, `text`), so the chat narrator can use it too.
"""
from __future__ import annotations

import os
import threading
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from backend.config import TORCH_DEVICE
from backend.models.external.llm import LLMUnavailable

# chosen by the faithfulness comparison in models/external/compare_writers.py
MODEL_ID = os.environ.get("STOCKINTEL_OPEN_LM", "Qwen/Qwen2.5-1.5B-Instruct")
MAX_NEW_TOKENS = 220


@dataclass
class Reply:
    model: str
    content: str


class OpenLM:
    """A local or Space-hosted instruction model behind the LLMClient interface."""

    def __init__(self, model_id: str = MODEL_ID, max_new_tokens: int = MAX_NEW_TOKENS):
        self.model, self.max_new_tokens = model_id, max_new_tokens
        self._lock = threading.Lock()
        self._loaded = None

    def _load(self):
        if self._loaded is None:
            try:
                import torch
                from transformers import AutoModelForCausalLM, AutoTokenizer
            except ImportError as exc:
                raise LLMUnavailable("the open model needs torch and transformers (pip install -e '.[ml]')") from exc
            dtype = torch.bfloat16      # half the memory of float32 at the same speed, measured on CPU
            tok = AutoTokenizer.from_pretrained(self.model)
            lm = AutoModelForCausalLM.from_pretrained(self.model, dtype=dtype).to(TORCH_DEVICE).eval()
            self._loaded = tok, lm
        return self._loaded

    def create(self, system: str, messages: List[Dict[str, Any]], tools: Optional[List[Dict[str, Any]]] = None) -> Reply:
        if tools:
            raise LLMUnavailable(f"{self.model} is used as a writer only; tool use needs the Claude provider")
        import torch
        with self._lock:
            tok, lm = self._load()
            chat = [{"role": "system", "content": system}] + [{"role": m["role"], "content": m["content"]} for m in messages]
            # Qwen3 thinks aloud unless told not to; other templates ignore the flag
            prompt = tok.apply_chat_template(chat, tokenize=False, add_generation_prompt=True, enable_thinking=False)
            ids = tok(prompt, return_tensors="pt").to(lm.device)
            with torch.no_grad():
                out = lm.generate(**ids, max_new_tokens=self.max_new_tokens, do_sample=False,
                                  pad_token_id=tok.eos_token_id)
            text = tok.decode(out[0, ids["input_ids"].shape[1]:], skip_special_tokens=True).strip()
        return Reply(self.model, text)

    @staticmethod
    def text(resp: Reply) -> str:
        return resp.content
