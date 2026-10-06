"""The language model as a router, never as an answerer.

When our patterns (api/nlu.py) do not recognise a question, the LM is asked which of our own handlers
answers it, choosing from `nlu.ROUTABLE`. It returns a label, not prose: the answer the user reads is
still built by our code from measured data. A reply that is not one of the labels is discarded.
"""
from __future__ import annotations

from typing import Dict, Optional

# measured on 20 questions through the Space's Phi-4-mini: this wording placed 17 correctly, against 15 for
# one that offered NONE on equal terms; both declined all 5 off-topic questions, which is the costly mistake
SYSTEM = """You route questions to one screen of an Indian stock-market app. Reply with exactly one label from \
the list and nothing else. Pick the label whose description comes closest, even if the question is vague or \
indirect. Reply NONE only when the question is not about investing, markets, money or the user's holdings at all."""


def choose(question: str, options: Dict[str, str], writer) -> Optional[str]:
    """The label whose description fits `question`, or None when the LM declines or misbehaves."""
    listed = "\n".join(f"{name}: {what}" for name, what in options.items())
    reply = writer.create(SYSTEM, [{"role": "user", "content": f"{listed}\n\nQuestion: {question}\nLabel:"}])
    answer = writer.text(reply).strip().strip(".:`'\"").lower()
    return answer if answer in options else None
