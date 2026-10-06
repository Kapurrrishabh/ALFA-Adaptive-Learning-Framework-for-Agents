"""The analyst note: our models' numbers, judged by our code, then made readable by a language model.

`facts` gathers what the models measured for one stock and states each judgement in words
("about the same as no change", "within noise"); `computed_note` puts them in sentences. The writer
only rewrites that note. A rewrite is shown only if `check` passes it: no figure, term or claim the
note does not carry, and nothing it leaves out that matters. Otherwise the computed note is shown,
with the reason, because a 1.5B model composing from raw fields misattributed figures in testing.
"""
from __future__ import annotations

import json
import re
from typing import Any, Dict, List

from backend.models.external.llm import verify_numbers

SYSTEM = """Rewrite the NOTE for an Indian retail investor in 4 short, plain sentences. Keep every number \
exactly as written, in rupees (₹). Keep every judgement as it is: "about the same", "within noise", the \
momentum fifth and the verdict. Add nothing: no new facts, no reasons, no tests or indicators the NOTE \
does not name, no predictions, no advice. Never say the price will rise or fall, and never call a model \
reliable or trustworthy. No lists or headings."""

# a model that claims direction is wrong by the project's own measurements, so the draft is refused
DIRECTION = re.compile(r"\b(will|is likely to|is expected to|should|could)\s+(rise|fall|go up|go down|increase|decrease|"
                       r"climb|drop|rally|decline|gain)|\b(bullish|bearish|upside target|price target|upward trend|"
                       r"downward trend|uptrend|downtrend|positive bias|negative bias)\b", re.I)
# terms a writer may reach for that the note never uses; adding one means it added a claim
NEW_TERMS = re.compile(r"\b(granger|sharpe|sortino|rsi|macd|bollinger|fibonacci|moving average|support level|"
                       r"resistance|p/e|earnings|dividend|beta|alpha|regression|neural|deep learning|sentiment)\b", re.I)
# praise the records do not support
OVERCLAIM = re.compile(r"\b(reliab\w*|trust\w*|accurate|better than random|strong evidence|"
                       r"performing well|confident(ly)?)\b", re.I)
# provisional: within 2% of no-change's own error we call it "about the same"; there is no confidence
# range on that gap yet, so a closer call would claim more than the record shows
SAME_AS_NO_CHANGE = 0.02


def _momentum_words(rank, universe):
    if not rank:
        return "not ranked (outside the Nifty 200)"
    fifth = ("top", "second", "middle", "second-lowest", "bottom")[min(4, int(5 * (rank - 1) / universe))]
    return f"momentum rank {rank} of {universe}, in the {fifth} fifth of the Nifty 200"


def facts(symbol: str, close: float, as_of: str, alfa: Dict[str, Any], gru: Dict[str, Any],
          band: Dict[str, Any], verdict: Dict[str, Any]) -> Dict[str, Any]:
    """One sheet of judged facts; every number the note may use is written here, the way it may be written."""
    days = len(alfa["q50"])
    gt = gru["record"]["test"]
    gd = next(d for d in gt["days"] if d["day"] == days)
    gap = gd["mae_pct"] / gd["no_change_mae_pct"] - 1
    error_words = ("about the same as" if abs(gap) <= SAME_AS_NO_CHANGE else "lower than" if gap < 0 else "higher than")
    noise = gd["direction_edge_lo"] <= 0 <= gd["direction_edge_hi"]
    lo, mid, hi = round(alfa["q10"][-1]), round(alfa["q50"][-1]), round(alfa["q90"][-1])
    return {
        "stock": symbol, "as_of": as_of, "last_close": f"₹{close:,.2f}", "days_ahead": days,
        "return_model_range": f"₹{lo:,} to ₹{hi:,}, middle ₹{mid:,} (80% of our return model's paths)",
        "volatility_band": f"₹{round(band['lo80']):,} to ₹{round(band['hi80']):,}",
        "return_model_record": f"beat the standard GARCH model on next-day returns by {-alfa['record']['vs_garch']:.3f} "
                               f"nats per return; that edge is for one day ahead only",
        "gru_line": f"ends at ₹{round(gru['close'][-1]):,}",
        "gru_moves": ("beat the volatility band at forecasting how big each day's move will be" if gt["nll_gain_lo"] > 0
                      else "did not beat the volatility band at forecasting how big each day's move will be"),
        "gru_error": f"{gd['mae_pct']}% on 2024–26, {error_words} the {gd['no_change_mae_pct']}% of assuming no change",
        "gru_direction": f"right {gd['direction_acc'] * 100:.1f}% of the time against {gd['base_rate_acc'] * 100:.1f}% "
                         f"for the base rate, " + ("which is within noise" if noise else "a real but small edge"),
        "momentum": _momentum_words(verdict["rank"], verdict["universe"]),
        "verdict": verdict["verdict"], "stop_loss": f"₹{round(verdict['stop']):,}",
        # the judgements a rewrite must keep word for word, so it cannot soften or upgrade them
        "_required": [f"₹{lo:,}", f"₹{hi:,}", verdict["verdict"], str(days)]
                     + ([("within noise", "within the noise")] if noise else [])
                     + ([("about the same", "similar", "roughly the same")] if error_words == "about the same as" else []),
    }


def computed_note(f: Dict[str, Any]) -> str:
    return (f"Over the next {f['days_ahead']} trading days our return model puts {f['stock']} at {f['return_model_range']}; "
            f"the volatility band says {f['volatility_band']}. The GRU line {f['gru_line']}; its error was {f['gru_error']}, "
            f"and its direction was {f['gru_direction']}. In testing the GRU {f['gru_moves']}, so its thin lines show "
            f"how far prices could swing day to day, not which way. The return model {f['return_model_record']}. "
            f"Verdict: {f['verdict']}, with a stop-loss at {f['stop_loss']} ({f['momentum']}).")


def check(text: str, f: Dict[str, Any]) -> List[str]:
    """Why a draft cannot be shown; empty when it can."""
    reasons = [f"figure {x} is not in the facts" for x in verify_numbers(text, [f, computed_note(f)])]
    if DIRECTION.search(text):
        reasons.append(f"it claims a direction: “{DIRECTION.search(text).group(0)}”")
    if OVERCLAIM.search(text):
        reasons.append(f"it praises a model beyond its record: “{OVERCLAIM.search(text).group(0)}”")
    if "$" in text:
        reasons.append("it writes dollars, not rupees")
    added = {t.lower() for t in NEW_TERMS.findall(text)} - {t.lower() for t in NEW_TERMS.findall(computed_note(f))}
    if added:
        reasons.append(f"it adds terms the note does not use: {', '.join(sorted(added))}")
    for need in f["_required"]:
        options = need if isinstance(need, tuple) else (need,)
        if not any(o in text.lower() if o.islower() else o in text for o in options):
            reasons.append(f"it leaves out {options[0]}")
    return reasons


def write(f: Dict[str, Any], writer) -> Dict[str, Any]:
    """The writer's note when it passes `check`, else the computed note with the reason."""
    resp = writer.create(SYSTEM, [{"role": "user", "content": "NOTE:\n" + computed_note(f)}])
    draft = writer.text(resp)
    reasons = check(draft, f)
    return {"text": draft if not reasons else computed_note(f), "written_by": resp.model if not reasons else None,
            "draft_rejected": reasons, "draft": draft}
