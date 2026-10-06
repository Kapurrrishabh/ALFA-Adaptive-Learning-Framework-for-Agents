"""Generative reasoning layer (Claude).

The model only phrases, compares and explains evidence the engines produced;
it never supplies a number. Every draft passes `verify_numbers`: each figure
in the text must match a figure in the evidence payload (within rounding) or
the draft is rejected and the deterministic answer is returned, with the
rejection stated in the reply. Secrets never enter prompts; the API key is
read by the SDK from the environment.

Refusals: requests opt into server-side fallbacks (`fallbacks="default"`) on
the Claude API; a response that still ends in `refusal` is reported as such.
"""
from __future__ import annotations

import json
import logging
import os
import re
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

from .data.provider import DataUnavailable
from .data.quality import DataQualityError

log = logging.getLogger("stockintel.llm")

DEFAULT_MODEL = os.environ.get("STOCKINTEL_LLM_MODEL", "claude-opus-5")
PROVIDER = os.environ.get("STOCKINTEL_LLM_PROVIDER", "anthropic")   # anthropic | bedrock
FALLBACK_BETA = "server-side-fallback-2026-07-01"

SYSTEM_PROMPT = """You are the explanation layer of a personal stock research system for an \
Indian retail investor. Specialized engines have already computed every fact, score and \
classification; they are in the EVIDENCE you receive. Your job is to explain that evidence \
clearly and answer the user's question.

Rules that keep the system trustworthy:
- Every number you write must appear in the EVIDENCE (you may round it). Never estimate, \
recall or compute a new figure; a verifier rejects drafts containing unsupported numbers.
- Keep the decision-support label exactly as given. Do not upgrade, downgrade or rephrase it \
into a different recommendation.
- Distinguish facts (observed or calculated) from model outputs (forecasts, sentiment scores).
- When domains disagree, say so and name both sides. When data is unavailable, say \
"Data unavailable" rather than filling the gap.
- Write for someone who is smart but not a finance professional: plain sentences, short \
paragraphs, no hype. Lead with the answer."""


class LLMUnavailable(Exception):
    pass


NUM = re.compile(r"(?<![\w.])([-+−]?)(\d[\d,]*(?:\.\d+)?)")
DATE = re.compile(r"\b\d{4}-\d{2}-\d{2}(?:[T ][\d:]+(?:[+-]\d{2}:\d{2}|Z)?)?\b")
CURRENCY = re.compile(r"(₹|\bRs\.?|\bINR|\bUSD|\$)\s*", re.I)
UNIT_AFTER = re.compile(r"\s*(%|x\b|times\b|crore|cr\b|lakh|bps|percent)", re.I)
COUNT_AFTER = re.compile(r"[\s-]*(domains?|stocks?|days?|sessions?|shares?|years?|months?|weeks?|"
                         r"events?|holdings?|analogues?|setups?|articles?|horizons?|models?)\b", re.I)


@dataclass(frozen=True)
class Figure:
    value: float
    decimals: int
    sign: str           # "", "+" or "-" as written
    has_unit: bool      # followed by %, x, crore... or preceded by a currency
    is_count: bool      # "3 domains", "5-day", or a list marker "1."


def figures(text: str) -> List[Figure]:
    text = CURRENCY.sub(" ₹", DATE.sub(" ", text))
    out = []
    for m in NUM.finditer(text):
        sign = m.group(1).replace("−", "-")
        raw = m.group(2).replace(",", "")
        try:
            value = float(raw)
        except ValueError:
            continue
        after = text[m.end():m.end() + 16]
        line_start = text[:m.start()].rstrip(" ").endswith("\n") or not text[:m.start()].strip()
        out.append(Figure(value=-value if sign == "-" else value,
                          decimals=len(raw.split(".")[1]) if "." in raw else 0, sign=sign,
                          has_unit=bool(UNIT_AFTER.match(after)) or text[max(0, m.start() - 2):m.start()].strip() == "₹",
                          is_count=bool(COUNT_AFTER.match(after)) or (line_start and after[:1] in ".)")))
    return out


def _source_numbers(obj: Any) -> List[Tuple[float, bool]]:
    """(value, is_raw_field). Figures already written in text are in display form."""
    found: List[Tuple[float, bool]] = []

    def walk(x: Any) -> None:
        if isinstance(x, bool) or x is None:
            return
        if isinstance(x, (int, float)):
            found.append((float(x), True))
        elif isinstance(x, str):
            found.extend((f.value, False) for f in figures(x))
        elif isinstance(x, dict):
            for v in x.values():
                walk(v)
        elif isinstance(x, (list, tuple)):
            for v in x:
                walk(v)
    walk(obj)
    return found


def evidence_surface(payload: Any) -> List[Any]:
    """The figures an explanation may cite from an analysis: the decision's
    numbers and every evidence claim. Deeper detail fields are excluded on
    purpose: the larger the pool, the likelier an invented figure matches."""
    if not (isinstance(payload, dict) and "results" in payload):
        return []
    d = payload.get("decision") or {}
    surface: List[Any] = [d.get("score"), d.get("confidence"), d.get("uncertainty"), payload.get("price")]
    for r in payload["results"].values():
        surface += [e.get("claim", "") for e in r.get("evidence", [])]
    return surface


def _supported(f: Figure, candidates: set) -> bool:
    tol = 0.5 * 10 ** (-f.decimals) + 1e-9
    for c in candidates:
        if f.sign == "-":
            ok = abs(f.value - c) <= tol
        elif f.sign == "+":
            ok = c >= 0 and abs(f.value - c) <= tol
        else:
            ok = abs(abs(f.value) - abs(c)) <= tol
        if ok:
            return True
    return False


def verify_numbers(text: str, sources: Any, small_int_ok: int = 12) -> List[str]:
    """Figures in `text` that no source figure supports. Empty list = faithful.

    A figure matches a source figure rounded to the precision written; a
    written sign must match; fractions may appear as percentages (0.183 ->
    18.3%). Small integers pass only when they count something or number a list.
    """
    candidates = set()
    for v, raw in _source_numbers(sources):
        candidates.add(v)
        if raw and abs(v) <= 1:
            candidates.add(v * 100)
    bad = []
    for f in figures(text):
        if f.decimals == 0 and abs(f.value) <= small_int_ok and f.is_count and not f.has_unit:
            continue
        if not _supported(f, candidates):
            bad.append(f"{f.value:g}")
    return bad


def _tool_surface(out: Any) -> List[Any]:
    if isinstance(out, dict) and "text" in out:
        return [out["text"]] + evidence_surface(out.get("data"))
    return [out]


@dataclass
class LLMClient:
    model: str = DEFAULT_MODEL
    effort: str = "medium"
    max_tokens: int = 4000
    _client: Any = field(default=None, repr=False)

    def __post_init__(self):
        try:
            import anthropic
        except ImportError as exc:
            raise LLMUnavailable("the anthropic package is not installed") from exc
        self._anthropic = anthropic
        if PROVIDER == "bedrock":
            region = os.environ.get("AWS_REGION")
            if not region:
                raise LLMUnavailable("STOCKINTEL_LLM_PROVIDER=bedrock needs AWS_REGION set")
            self._client = anthropic.AnthropicBedrockMantle(aws_region=region)
            if not self.model.startswith("anthropic."):
                self.model = "anthropic." + self.model
        else:
            self._client = anthropic.Anthropic()

    def create(self, system: str, messages: List[Dict[str, Any]],
               tools: Optional[List[Dict[str, Any]]] = None) -> Any:
        kwargs: Dict[str, Any] = dict(model=self.model, max_tokens=self.max_tokens, system=system,
                                      messages=messages, thinking={"type": "adaptive"},
                                      output_config={"effort": self.effort})
        if tools:
            kwargs["tools"] = tools
        a = self._anthropic
        try:
            if PROVIDER == "bedrock":
                resp = self._client.messages.create(**kwargs)
            else:
                resp = self._client.beta.messages.create(betas=[FALLBACK_BETA], fallbacks="default",
                                                         **kwargs)
        except a.AuthenticationError as exc:
            fix = ("set AWS_PROFILE to a profile with Bedrock access to this model"
                   if PROVIDER == "bedrock" else "set ANTHROPIC_API_KEY or run `ant auth login`")
            raise LLMUnavailable(f"Claude authentication failed ({exc.status_code}); {fix}") from exc
        except a.RateLimitError as exc:
            raise LLMUnavailable("Claude API rate limit reached; retry shortly") from exc
        except a.APIConnectionError as exc:
            raise LLMUnavailable(f"cannot reach the Claude API: {exc}") from exc
        except a.APIStatusError as exc:
            raise LLMUnavailable(f"Claude API error {exc.status_code}: {exc.message}") from exc
        if resp.stop_reason == "refusal":
            cat = getattr(getattr(resp, "stop_details", None), "category", None)
            raise LLMUnavailable(f"the model declined this request (category: {cat})")
        return resp

    @staticmethod
    def text(resp: Any) -> str:
        return "".join(b.text for b in resp.content if b.type == "text").strip()


def compact_payload(analysis_dict: Dict[str, Any], max_events: int = 6) -> Dict[str, Any]:
    """Trim an analysis to what an explanation needs (keeps tokens bounded)."""
    res = {}
    for dom, r in analysis_dict["results"].items():
        details = dict(r.get("details") or {})
        if dom == "news_sentiment" and "events" in details:
            details["events"] = [{k: e[k] for k in ("timestamp", "event_type", "headline", "source",
                                                    "sentiment", "scope", "importance")}
                                 for e in details["events"][:max_events]]
        if dom == "forecast":
            for h in (details.get("horizons") or {}).values():
                for m in (h.get("oos") or {}).values():
                    m.pop("calibration", None)
        if dom == "pattern":
            details.pop("patterns_history", None)
        res[dom] = {"state": r["state"], "score": r["score"], "confidence": r["confidence"],
                    "error": r["error"], "details": details,
                    "evidence": [{"claim": e["claim"], "direction": e["direction"],
                                  "kind": "MODEL OUTPUT" if e["is_model_output"] else "FACT"}
                                 for e in r["evidence"]]}
    return {k: analysis_dict[k] for k in ("symbol", "name", "price", "currency",
                                          "analysis_timestamp", "held", "quality")} | {
        "decision": analysis_dict["decision"], "results": res}


class Narrator:
    """Rewrites a deterministic answer conversationally, under verification."""

    def __init__(self, client: LLMClient):
        self.client = client

    def narrate(self, question: str, payload: Dict[str, Any], draft: str) -> Tuple[str, Dict[str, Any]]:
        prompt = (f"USER QUESTION:\n{question}\n\nEVIDENCE (JSON):\n"
                  f"{json.dumps(payload, default=str)}\n\nDETERMINISTIC DRAFT (complete and correct, "
                  f"but mechanical):\n{draft}\n\nAnswer the user's question from the evidence. Keep "
                  "the classification unchanged and use only figures that appear in the draft.")
        resp = self.client.create(SYSTEM_PROMPT, [{"role": "user", "content": prompt}])
        text = self.client.text(resp)
        bad = verify_numbers(text, [draft, question] + evidence_surface(payload))
        meta = {"model": getattr(resp, "model", self.client.model), "unsupported_numbers": bad}
        if bad:
            log.warning("narration rejected, unsupported numbers: %s", bad)
            return (draft + f"\n\n(The conversational rewrite was discarded because it contained "
                            f"figures not present in the evidence: {', '.join(bad[:6])}.)"), meta
        return text, meta


class ToolAgent:
    """Claude chooses which analytical tools to call; the system executes
    them (read-only, validated, time-bounded) and Claude explains the results.
    The final answer is verified against every tool output it saw."""

    def __init__(self, client: LLMClient, tools: Dict[str, Tuple[Dict[str, Any], Callable[..., Dict[str, Any]]]],
                 max_steps: int = 6):
        self.client = client
        self.tools = tools
        self.max_steps = max_steps

    def run(self, question: str, history: List[Dict[str, Any]]) -> Tuple[str, Dict[str, Any]]:
        schemas = [{"name": n, **spec} for n, (spec, _) in self.tools.items()]
        messages = history + [{"role": "user", "content": question}]
        seen: List[Any] = []
        calls: List[str] = []
        for _ in range(self.max_steps):
            resp = self.client.create(SYSTEM_PROMPT + "\nUse the tools to fetch evidence before "
                                      "answering any question about a specific stock, portfolio or "
                                      "screen. Answer definitional questions with the knowledge tool. "
                                      "Quote figures as they appear in the tools' text output.",
                                      messages, tools=schemas)
            messages.append({"role": "assistant", "content": resp.content})
            if resp.stop_reason != "tool_use":
                text = self.client.text(resp)
                bad = verify_numbers(text, [question] + [x for o in seen for x in _tool_surface(o)])
                return text, {"tool_calls": calls, "unsupported_numbers": bad, "messages": messages}
            results = []
            for block in resp.content:
                if block.type != "tool_use":
                    continue
                calls.append(block.name)
                if block.name not in self.tools:
                    results.append({"type": "tool_result", "tool_use_id": block.id, "is_error": True,
                                    "content": f"unknown tool {block.name}"})
                    continue
                try:
                    out = self.tools[block.name][1](**(block.input or {}))
                    seen.append(out)
                    results.append({"type": "tool_result", "tool_use_id": block.id,
                                    "content": json.dumps(out, default=str)[:60000]})
                except (ValueError, KeyError, TypeError, DataUnavailable, DataQualityError) as exc:
                    results.append({"type": "tool_result", "tool_use_id": block.id, "is_error": True,
                                    "content": f"{type(exc).__name__}: {exc}"})
            messages.append({"role": "user", "content": results})
        raise LLMUnavailable(f"agent did not finish within {self.max_steps} tool steps")
