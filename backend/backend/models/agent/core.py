"""One free-text question in, one answer or one refusal out, with the reason attached either way.

The four stages are the ones the architecture names, and the order matters: **route, assemble, generate,
screen.** Routing comes first because everything downstream depends on knowing which question this is —
which facts the answer may quote, and which trained wording to hand the decoder in place of the user's.

Six separate things can stop this from speaking, and each one names itself in the turn rather than
returning a generic refusal:

  no subject         no instrument on disk is named, so there is nothing to look up
  unclear question   the routing margin is under the cut, so answering would be a guess at the intent
  missing facts      the routed intent quotes a figure the evidence does not carry, so it would invent
  search failed      the outside sources could not be read, so "nothing found" would be a guess
  unsupported figure the guard found a figure in the answer that the evidence does not state
  low confidence     the answer is below the abstention cut the feedback log solved for

The last two are B's work reused unchanged, and nothing in this file fits a parameter. What adapts is
the cut and the calibrator, both loaded from what the feedback log solved for.

A question naming no instrument is checked against the conversation's subject before it becomes the first
refusal. "How has it been doing" is a whole question to a person and no question at all to a router, and
the carried subject is resolved against the price files like any other, so nothing can be named this way
that a user could not have typed.

The first two of those five are also where a **reference** answer gets its chance, and so is the one
routed intent whose answer is a refusal. A question naming no instrument, one the router cannot place, or
one it placed as asking for a figure we do not hold, is not necessarily a question we hold nothing on -- it
is one the price snapshot cannot answer. So before any of those three refusals is returned, `reference` is
asked to find it in the documents. Those three and nowhere else on purpose: an intent the router placed on
a figure we do have is answered from that figure, because that is the path whose faithfulness is measured.
With no `reference` given, all three refuse exactly as they did before it existed.
"""

from collections import namedtuple

from backend.models.guardrails import guardrails
from backend.models.core.backend import xp
from backend.models.networks.generator import ANSWER_TEMPERATURE, ANSWER_TOP_P

from backend.models.agent import phrase
from backend.models.agent.context import assemble
from backend.models.agent.reference import REFERENCE, UNANSWERED, SearchFailed

# Everything the turn did, so a caller can log it, show it, or judge it without re-running anything.
# `served` is what a user reads; `answer` is what the model wrote, which differ exactly when a guard
# fired, and keeping both is what makes the unsupported-figure rate measurable at serving time.
# `phrased_by` names the language model whose words were served; `unphrased_because` says why they were
# not, when a writer was asked and its rewrite was not served.
Turn = namedtuple("Turn", "question asked ticker intent served answer evidence confidence stated "
                          "spoke margin unsupported because as_of phrased_by unphrased_because",
                  defaults=("", ""))


class Agent:
    """Router, context assembler, generator and guards, with the weights loaded once.

    The model is given to the constructor rather than a path: a served agent and a measured one have to
    be the same object, and loading inside would make "which checkpoint answered this" unanswerable.
    """

    def __init__(self, domain, market, router, gate, tokenizer, model, config, abstainer,
                 calibrator=None, rng=None, temperature=ANSWER_TEMPERATURE, top_p=ANSWER_TOP_P,
                 reference=None, writer=None):
        self.domain = domain
        self.market = market
        self.router = router
        # Optional, and the default is the behaviour that was measured without it: the advisory path is
        # what C6's numbers are on, and this only takes over questions that path was going to refuse.
        self.reference = reference
        # Optional too: the language model that rewords a spoken answer. Whether to speak is still decided
        # on the generator's own draft, which is what the cut and the faithfulness record were measured on.
        self.writer = writer
        # Two cuts, on two different quantities, fitted the same way: `gate` on the routing margin and
        # `abstainer` on the answer's own confidence. A question can be understood and the answer still
        # not worth saying, and the reverse, so one cut could not stand for both.
        self.gate = gate
        self.abstainer = abstainer
        self.calibrator = calibrator
        self.tokenizer = tokenizer
        self.model = model
        self.config = config
        self.rng = rng
        self.temperature = temperature
        self.top_p = top_p

    def answer(self, question, as_of=None, subject=""):
        """One turn. Never raises on a question it cannot handle — it says which stage stopped it.

        `subject` is the instrument the conversation was already about, and it is used only when this
        question names none: "how has it been doing" is a question about whatever the last one was about,
        and resolved against the price files like any other, so a caller cannot name an instrument this
        way that a user could not.
        """
        ticker = self.market.resolve(question) or self.market.resolve(subject)
        if ticker is None:
            return (self._looked_up(question, as_of)
                    or self._quiet(question, self.domain.NO_SUBJECT, "no subject"))

        route = self.router.route(self.domain.without_subject(question, ticker))
        if not bool(self.gate.answers([route.margin])[0]):
            return (self._looked_up(question, as_of, ticker=ticker, margin=route.margin)
                    or self._quiet(question, self.domain.UNKNOWN_QUESTION, "unclear question",
                                   ticker=ticker, margin=route.margin))

        # The third place a reference answer gets its chance, and the plainest: this intent's whole answer
        # is "the figures do not carry that", so the documents are where it could be carried. Nothing
        # changes when they hold nothing -- the trained refusal below is still what a user hears.
        if route.label == self.domain.UNANSWERABLE:
            looked = self._looked_up(question, as_of, ticker=ticker, margin=route.margin)
            if looked is not None:
                return looked

        evidence, shown, taken_at = self.market.snapshot(ticker, as_of)
        missing = [fact for fact in self.domain.needs(route.label) if fact not in shown]
        if missing:
            return self._quiet(question, self.domain.REFUSAL, f"missing {', '.join(missing)}",
                               ticker=ticker, intent=route.label, evidence=evidence,
                               margin=route.margin, as_of=taken_at)

        asked = self.domain.ask(route.label, route.key, ticker)
        context = assemble(self.tokenizer, asked, [evidence], self.config.max_text_length,
                           self.domain.QUESTION_TOKENS)
        produced = self.model.generate(context.source, context.keep, self.temperature, self.top_p,
                                       self.rng)
        confidence = float(self.model.confidence(context.source, produced, context.keep)[0])
        written = self.tokenizer.decode(produced[0])

        served, unsupported = guardrails.screen(written, evidence, self.domain.REFUSAL)
        speaks = bool(self.abstainer.answers([confidence])[0]) and not unsupported
        because = "unsupported figure" if unsupported else ("" if speaks else "low confidence")
        phrased = (phrase.rewrite(question, written, evidence, self.writer, phrase.currency(ticker))
                   if speaks and self.writer is not None else phrase.Phrased("", "", ""))
        return Turn(question, asked, ticker, route.label,
                    (phrased.text or served) if speaks else self.domain.REFUSAL,
                    written, evidence, confidence, self._stated(confidence), speaks, route.margin,
                    unsupported, because, taken_at, phrased.by, phrased.because)

    def _looked_up(self, question, as_of, ticker="", margin=float("nan")):
        """A turn read out of the documents, or None when there is nothing to read one from.

        None, not a refusal, so a caller that gets nothing back still returns the refusal it already had.
        A search that could not be run is the exception: that is a refusal of its own, because the documents
        never got their answer and the trained refusal would be describing a question rather than a network.
        The question is passed to the model as the user typed it: this path has no canonical phrasings to
        rewrite it into, which is the point of it. The ticker goes with it as the subject, which both widens
        the search and is what the passages then have to be about, so a question that says "it" is looked up
        under the name of the thing it means and cannot be answered out of a passage about something else.

        `stated` stays None even though a confidence was reported. The calibrator maps a confidence to a
        chance of being right using the advisory generator's record, and these are other weights on
        another task, so running it here would quote a likelihood nothing has measured. `as_of` carries the
        passage's own publication date, which is the date of the evidence actually served.
        """
        if self.reference is None:
            return None
        try:
            looked = self.reference.look_up(question, as_of,
                                            self.market.subject(ticker) if ticker else None)
        except SearchFailed as error:
            # A refusal rather than nothing, and a different refusal from the trained one: the documents
            # would have been asked and were not, so saying "i am not sure what you are asking" would
            # blame the question for a network that was down.
            return self._quiet(question, self.domain.UNREACHABLE, f"search failed: {error}",
                               ticker=ticker, margin=margin)
        if looked is None:
            return None
        # Checked once something was found, so a question nothing matched keeps the refusal it always had:
        # 519,135 general passages answer "write me a poem about the sea" out of The Lusiads otherwise.
        if not ticker and not self.domain.about_money(question):
            return self._quiet(question, self.domain.OFF_TOPIC, "not about money", margin=margin)
        if looked.how == UNANSWERED:
            return self._quiet(question, self.domain.NOT_IN_SOURCES, "sources do not answer it", ticker=ticker,
                               evidence=looked.evidence, margin=margin, as_of=looked.day or None)
        return Turn(question, question, ticker, f"{REFERENCE} {looked.how}", looked.served,
                    looked.answer, looked.evidence, looked.confidence, None, True, margin,
                    looked.unsupported, "", looked.day or None, looked.phrased_by, looked.unphrased_because)

    def _stated(self, confidence):
        """The chance of being right, in the units B3 fitted. None until a calibrator has been fitted."""
        return None if self.calibrator is None else float(self.calibrator(confidence))

    def _quiet(self, question, served, because, ticker="", intent="", evidence="",
               margin=float("nan"), as_of=None):
        """A turn that stopped before the model ran. No confidence, because nothing was generated.

        Empty rather than None for the text a stage never reached: every caller already reads these as
        "nothing routed", and one of them stores them in columns that refuse a null.
        """
        return Turn(question, "", ticker, intent, served, "", evidence, float("nan"), None, False,
                    margin, [], because, as_of)


def answered(turns):
    """(spoke, of how many) over a set of turns, which is the coverage every precision here is on."""
    return sum(turn.spoke for turn in turns), len(turns)


def routing_margins(turns):
    """The margins, for fitting the routing gate on outcomes rather than picking it by hand."""
    return xp.asarray([turn.margin for turn in turns], dtype=xp.float64)
