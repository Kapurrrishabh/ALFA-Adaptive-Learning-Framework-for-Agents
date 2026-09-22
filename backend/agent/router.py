"""Which question is being asked, decided before the decoder reads anything.

**Measured cause.** Of the 95 answers the agent judge called wrong, misrouting is the largest single
group: 14 buy-advice templates answering an overbought question, 13 performance answering an outlook or
volatility one, 6 drawdown answering an outlook one. The figure guard sees none of it, because every
digit in a wrong-intent answer is supported by the evidence it was given. A figure guard is not an
intent guard, and this is the guard it is missing.

**And routing buys more than the intent.** The decoder answers a wording it trained on 90.5% exactly and
an unseen wording 28.5%. Once the nearest *trained* phrasing is known, the question can be rewritten into
it before the decoder reads it — the user's words choose the question, the model only ever reads words it
has seen. That is a large win available with the weights frozen, and it is bought with a risk: a
confident answer to the wrong question. So the router carries a threshold and says nothing when the
margin is thin, fitted from routing outcomes by `scripts/route.py` exactly as B4 fits the answer cut.

The scorer is a seam with two implementations because which one wins is a measurement, not an opinion:
`lexical` needs no weights and nothing loaded, `semantic` runs the frozen encoder, which
`scripts/probe_intents.py` measured placing a novel word 15/15 and a novel sentence shape 28/41.
"""

from collections import namedtuple

from selfagent.backend import xp
from selfagent.models.retriever import BM25, embed

# What the router decided, with the evidence for the decision attached. `margin` is the score of the
# best example minus the best scoring example carrying a different label, so it measures how nearly
# this question was a different question — which is the quantity a cut has to be placed on.
Route = namedtuple("Route", "label key text score margin")


def lexical(texts, tokenize):
    """Word overlap over the known phrasings. No weights, so it works before any model is loaded."""
    index = BM25(texts, tokenize)
    return index.scores


def semantic(texts, model, tokenizer, length):
    """Cosine against the frozen encoder's reading of each known phrasing."""
    known = embed(model, tokenizer, texts, length)
    # einsum rather than @: numpy 2.0.2 on Apple's Accelerate BLAS raises spurious overflow flags for a
    # float64 matmul whose result is correct to 1.3e-15.
    return lambda text: xp.einsum("kd,d->k", known, embed(model, tokenizer, [text], length)[0])


class Router:
    """Matches free text to the nearest labelled example, and reports how near the runner-up was.

    Deliberately holds no decision. Whether the margin is wide enough to act on is an `Abstainer`,
    fitted on measured routing outcomes and owned by the caller, so the router cannot promise a
    precision nobody measured.
    """

    def __init__(self, examples, score):
        if not examples:
            raise ValueError("a router needs labelled examples to match against; none were given")
        self.labels, self.keys, self.texts = (tuple(column) for column in zip(*examples))
        self.score = score

    def route(self, text):
        scores = xp.asarray(self.score(text), dtype=xp.float64)
        best = int(scores.argmax())
        # A question sharing no term with anything known scores zero everywhere, and then the argmax is
        # whichever example came first. Reporting a zero margin is what stops that from being served.
        others = [index for index, label in enumerate(self.labels) if label != self.labels[best]]
        runner = float(scores[others].max()) if others else float("-inf")
        return Route(self.labels[best], self.keys[best], self.texts[best],
                     float(scores[best]), float(scores[best]) - runner)
