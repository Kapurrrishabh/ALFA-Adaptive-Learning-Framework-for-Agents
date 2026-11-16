"""One free-text question in, one answer or one refusal out, with the reason attached either way.

The four stages are the ones the architecture names, and the order matters: **route, assemble, generate,
screen.** Routing comes first because everything downstream depends on knowing which question this is —
which facts the answer may quote, and which trained wording to hand the decoder in place of the user's.

Twelve separate things can stop this from speaking, and each one names itself in the turn rather than
returning a generic refusal:

  no subject         no instrument on disk is named, so there is nothing to look up
  unclear question   the routing margin is under the cut, so answering would be a guess at the intent
  missing facts      the routed intent quotes a figure the evidence does not carry, so it would invent
  search failed      the outside sources could not be read, so "nothing found" would be a guess
  no prices          a listed share's prices could not be fetched, or are too short for its figures
  no facts           a facts question's share has no dated analysis to read, here or from Yahoo
  another share      the guard found the answer naming a share other than the evidence's
  mislabelled figure the guard found a facts answer giving a figure as another metric's: ROE as ROIC
  wrong trend        the guard found the answer's "above both", "below both" or "between" against its averages
  own meaning        the guard found a facts answer explaining a term itself, which TERM_MEANINGS does below it
  unsupported figure the guard found a figure in the answer that the evidence does not state
  low confidence     the answer is below the abstention cut the feedback log solved for, or for a facts
                     answer the cut fitted on the facts rows

For a price intent, another share, wrong trend, unsupported figure and low confidence stop ALFA's answer, not the turn:
our code's sentence for the intent speaks instead.

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
asked to find it in the documents. Beyond those three, only a question about what happened lately, or one asking
how the price moved over a window, reads the dated sources first (`Reference.asks_lately`), and that
answer opens with the price files' own figures in a sentence of ours. Any other intent the router placed on a
figure we do have is answered from that figure, because that is the path whose faithfulness is measured.
With no `reference` given, all three refuse exactly as they did before it existed.
"""

from collections import namedtuple

from backend.models.guardrails import guardrails
from backend.models.core.backend import xp
from backend.models.networks.generator import ANSWER_TEMPERATURE, ANSWER_TOP_P

from backend.models.agent import facts_reply, phrase
from backend.models.agent.context import assemble, evidence_room
from backend.models.agent.reference import REFERENCE, UNANSWERED, SearchFailed

# Everything the turn did, so a caller can log it, show it, or judge it without re-running anything.
# `served` is what a user reads; `answer` is what the model wrote, which differ exactly when a guard
# fired, and keeping both is what makes the unsupported-figure rate measurable at serving time.
# `phrased_by` names the language model whose words were served, or BUILT when our code's were: a facts reply, or the
# sentence in place of an answer of ALFA's that a guard or the cut refused; `unphrased_because` says why a writer's
# rewrite was not served, and `asked_writer` whether the turn's request reached that language model.
Turn = namedtuple("Turn", "question asked ticker intent served answer evidence confidence stated "
                          "spoke margin unsupported because as_of phrased_by unphrased_because asked_writer",
                  defaults=("", "", False))
BUILT = phrase.BUILT
# names ALFA's own words among the parts of a turn answering two questions, so "our code" names only our code's
ALFA = "ALFA"
# How a turn answering two questions names their intents ("performance and buy"), and how its texts are laid out
TOGETHER = " and "
PARAGRAPH = "\n\n"
# Off: read by hand, 54 of the 68 facts answers step36000 would speak were wrong in their words, every figure right, so
# our code writes them. Turn on only for a generator whose spoken answers pass that reading.
GENERATOR_WRITES_FACTS = False


class Agent:
    """Router, context assembler, generator and guards, with the weights loaded once.

    The model is given to the constructor rather than a path: a served agent and a measured one have to
    be the same object, and loading inside would make "which checkpoint answered this" unanswerable.
    """

    def __init__(self, domain, market, router, gate, tokenizer, model, config, abstainer,
                 calibrator=None, rng=None, temperature=ANSWER_TEMPERATURE, top_p=ANSWER_TOP_P,
                 reference=None, facts=None, facts_abstainer=None):
        self.domain = domain
        self.market = market
        self.router = router
        # `facts(texts)` is the encoder's similarity, which `facts_scorer` picks a share's facts with as training did
        places_facts = domain.FACTS in router.labels
        generates_facts = places_facts and GENERATOR_WRITES_FACTS
        if places_facts != (facts is not None) or generates_facts != (facts_abstainer is not None):
            raise ValueError("give the agent a facts scorer exactly when its router places facts questions, and a "
                             "facts answer cut exactly when the generator writes their answers too "
                             "(GENERATOR_WRITES_FACTS): without either it cannot pick a passage or judge the answer, "
                             "or is never asked to")
        self.facts = facts
        # Fitted on facts rows: their confidence averaged 0.45, and the advisory cut, near 0.8, refused nearly all.
        # An answer our code writes has no confidence to cut.
        self.facts_abstainer = facts_abstainer
        # Optional, and the default is the behaviour that was measured without it: the advisory path is
        # what C6's numbers are on, and this only takes over questions that path was going to refuse.
        self.reference = reference
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
            return (self._looked_up(question, as_of, stock=subject)
                    or self._quiet(question, self.domain.NO_SUBJECT, "no subject"))
        try:
            self.market.hold(ticker)
        except self.domain.NoPrices as error:
            # No ticker on the turn, and `agent_server` clears the conversation's share on this refusal, so "should i
            # buy it" next is about neither this share nor the one before
            return self._quiet(question, self.domain.NO_PRICES, f"no prices: {error}")

        route = self.router.route(self.domain.without_subject(question, ticker))
        sure = bool(self.gate.answers([route.margin])[0])
        # Before the newswire, which reads a joined question's look-back word and answers its first part alone
        if (together := self._together(question, as_of, ticker, sure)) is not None:
            return together
        # Before the newswire too: its "today" answered "what is the intraday high of hdfc bank today" with a headline
        if self.domain.asks_day_range(question):
            return self._quiet(question, self.domain.NO_DAY_RANGE, "no day's high or low", ticker=ticker,
                               margin=route.margin)
        if self.reference is not None and self.reference.asks_lately(question, route.label == self.domain.MOVES):
            looked = self._looked_up(question, as_of, ticker=ticker, margin=route.margin, lately=True)
            if looked is not None:
                return looked

        if not sure:
            return (self._looked_up(question, as_of, ticker=ticker, margin=route.margin)
                    or self._quiet(question, self.domain.UNKNOWN_QUESTION, "unclear question",
                                   ticker=ticker, margin=route.margin))
        if route.label == self.domain.FACTS:
            return self._from_facts(question, as_of, ticker, route.margin)

        # The third place a reference answer gets its chance, and the plainest: this intent's whole answer
        # is "the figures do not carry that", so the documents are where it could be carried. Nothing
        # changes when they hold nothing -- the trained refusal below is still what a user hears.
        if route.label == self.domain.UNANSWERABLE:
            looked = self._looked_up(question, as_of, ticker=ticker, margin=route.margin)
            if looked is not None:
                return looked

        prices = self.market.snapshot(ticker, as_of)
        evidence, shown, taken_at = prices
        missing = [fact for fact in self.domain.needs(route.label) if fact not in shown]
        if missing:
            return self._quiet(question, self.domain.REFUSAL, f"missing {', '.join(missing)}",
                               ticker=ticker, intent=route.label, evidence=evidence,
                               margin=route.margin, as_of=taken_at)

        built = ("" if route.label == self.domain.UNANSWERABLE
                 else phrase.price_move(self.market.subject(ticker), prices) if route.label == self.domain.MOVES
                 else phrase.answer_form(route.label, route.key, ticker, shown))
        return self._written(question, self.domain.ask(route.label, route.key, ticker), ticker, route.label,
                             evidence, self.domain.QUESTION_TOKENS, route.margin, taken_at, built)

    def _from_facts(self, question, as_of, ticker, margin):
        """A share's dated facts, the ones the question needs picked into one passage, answered in our words from it.

        With GENERATOR_WRITES_FACTS the decoder answers instead, under every guard, given the question as typed: the
        facts route has no trained wording to rewrite it into.
        """
        try:
            lines, taken_at = self.market.facts(ticker, as_of)
        except self.domain.NoFacts as error:
            return self._quiet(question, self.domain.NO_FACTS, f"no facts: {error}", ticker=ticker,
                               intent=self.domain.FACTS, margin=margin)
        passage = self.domain.facts_passage(
            question, lines, self.domain.facts_scorer(lines, self.facts), lambda text: len(self.tokenizer.encode(text)),
            evidence_room(self.config.max_text_length, self.domain.FACTS_QUESTION_TOKENS))
        if GENERATOR_WRITES_FACTS:
            return self._written(question, question, ticker, self.domain.FACTS, passage,
                                 self.domain.FACTS_QUESTION_TOKENS, margin, taken_at)
        told = facts_reply.reply(question, passage, all_lines=lines)
        return Turn(question, question, ticker, self.domain.FACTS, told.text, told.text, told.evidence,
                    float("nan"), None, True, margin, [], "", taken_at, BUILT)

    def _written(self, question, asked, ticker, intent, evidence, question_tokens, margin, taken_at, built=""):
        """The decoder's answer to `asked` from `evidence`, spoken as written only past the figure and share guards and the
        cut. Refused, `built` is served when there is one: our code's sentence for the intent, every figure copied from
        `evidence`. It keeps the refused answer and its confidence, so the feedback log judges ALFA's words, not ours."""
        context = assemble(self.tokenizer, asked, [evidence], self.config.max_text_length, question_tokens)
        produced = self.model.generate(context.source, context.keep, self.temperature, self.top_p,
                                       self.rng)
        confidence = float(self.model.confidence(context.source, produced, context.keep)[0])
        written = self.tokenizer.decode(produced[0])

        served, unsupported = guardrails.screen(written, evidence, self.domain.REFUSAL)
        facts = intent == self.domain.FACTS
        others = guardrails.other_shares(written, evidence, self.market.names, self.domain.FACTS_SLOTS
                                         if facts else self.domain.SUBJECT_SLOTS)
        misworded = self._misworded(written, evidence, facts)
        cut = self.facts_abstainer if facts else self.abstainer
        speaks = bool(cut.answers([confidence])[0]) and not unsupported and not misworded and not others
        because = ("unsupported figure" if unsupported else f"names another share: {', '.join(others)}" if others
                   else misworded or ("" if speaks else "low confidence"))
        if not speaks and built:
            return Turn(question, asked, ticker, intent, built, written, evidence, confidence, None, True, margin,
                        unsupported, because, taken_at, BUILT)
        return Turn(question, asked, ticker, intent, served if speaks else self.domain.REFUSAL,
                    written, evidence, confidence, self._stated(confidence), speaks, margin,
                    unsupported, because, taken_at)

    def _misworded(self, text, evidence, facts):
        """Why the words of `text` misstate `evidence`, or "" when they do not."""
        # an advisory form puts each figure in its measure's own slot; a facts answer names the measure in free prose,
        # and explaining one is TERM_MEANINGS' job, below the answer
        mislabelled = guardrails.mislabelled_figures(text, evidence, self.domain.FACTS_METRICS) if facts else []
        defined = guardrails.own_meanings(text, self.domain.FACTS_METRICS) if facts else []
        trend = guardrails.wrong_trend(text, evidence)
        return ("mislabelled figure: " + ", ".join(f"{figure} as {metric}" for figure, metric in mislabelled)
                if mislabelled else f"its own meaning of {', '.join(defined)}" if defined
                else "wrong trend: " + ", ".join(f"says {said}, the figures say {truth or 'no averages'}"
                                                 for said, truth in trend) if trend else "")

    def _together(self, question, as_of, ticker, sure_of_whole):
        """One turn for a question asking two the router is sure of ("how is TCS doing and should i buy it"), else None.
        When the router is `sure_of_whole`, it splits only parts placed apart: one route answered "what is the p/e of
        tcs and how volatile is it" with the P/E alone. Two facts parts are apart too, as a facts reply gives the first
        area asked: "what is the p/e of RELIANCE.NS and how much debt does it have" lost its debt.

        Each part is a turn of its own with every guard, and this speaks when any part does. Its confidence is the least
        of the parts ALFA's words answered, but `agent_server` never judges it into the answer cut: no one generation
        wrote it.
        """
        parts = self.domain.parts(question)
        found = [self.market.resolve(part) for part in parts]
        # A turn carries one instrument, so each part must name it or point back to it: to an earlier part, or to the
        # conversation's share when no part names one, never ahead. resolve() cannot read "apple", and "is it a good time
        # to buy apple" points back, so a part naming another share by any known name was given this share's figures.
        first = found.index(ticker) if ticker in found else -1
        if not parts or any(named != ticker and (named or index < first or not self.domain.points_back(part)
                                                 or self.market.names_another(part, ticker))
                            for index, (part, named) in enumerate(zip(parts, found))):
            return None
        # Sure of the whole, a part's "this" or "that" is the figure just asked about: "how volatile is TCS and is that
        # high" was split into how far TCS.NS is below its high
        if sure_of_whole and any(named != ticker and self.domain.points_back_only_by_this_or_that(part)
                                 for part, named in zip(parts, found)):
            return None
        parts = [part if named == ticker else self.domain.naming(part, ticker) for part, named in zip(parts, found)]
        routes = [self.router.route(self.domain.without_subject(part, ticker)) for part in parts]
        if not all(self.gate.answers([route.margin for route in routes])) \
                or any(route.label == self.domain.UNANSWERABLE for route in routes) \
                or sure_of_whole and len({route.label for route in routes}) == 1 and routes[0].label != self.domain.FACTS:
            return None
        turns = [self.answer(part, as_of, ticker) for part in parts]
        confidence = min((turn.confidence for turn in turns if alfa_spoke(turn) and turn.confidence == turn.confidence),
                         default=float("nan"))

        def joined(field, between=PARAGRAPH):
            return between.join(dict.fromkeys(getattr(turn, field) for turn in turns if getattr(turn, field)))
        wordings = dict.fromkeys(turn.phrased_by or ALFA for turn in turns if turn.spoke)
        return Turn(question, joined("asked"), ticker, TOGETHER.join(route.label for route in routes),
                    joined("served"), joined("answer"), joined("evidence"), confidence,
                    self._stated(confidence) if confidence == confidence else None, any(turn.spoke for turn in turns),
                    min(route.margin for route in routes), [figure for turn in turns for figure in turn.unsupported],
                    "; ".join(f"{route.label}: {turn.because}" for route, turn in zip(routes, turns) if turn.because),
                    next((turn.as_of for turn in turns if turn.as_of), None),
                    "" if list(wordings) == [ALFA] else ", ".join(wordings), joined("unphrased_because", "; "),
                    any(turn.asked_writer for turn in turns))

    def _looked_up(self, question, as_of, ticker="", margin=float("nan"), lately=False, stock=""):
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
        passage's own publication date, which is the date of the evidence actually served. `stock` is the subject the
        caller gave when we hold no prices for it: a question asked about a stock is about money, whatever its words,
        and is looked up under it like a ticker, so only a passage that names it is served, and the turn carries it.
        """
        if self.reference is None:
            return None
        about = ticker or stock
        on_topic = bool(about) or self.domain.about_money(question)
        try:
            looked = self.reference.look_up(question, as_of, self.market.subject(about) if about else None,
                                            self.market.snapshot(ticker, as_of) if ticker else None, lately, on_topic)
        except SearchFailed as error:
            # A refusal rather than nothing, and a different refusal from the trained one: the documents
            # would have been asked and were not, so saying "i am not sure what you are asking" would
            # blame the question for a network that was down.
            return self._quiet(question, self.domain.UNREACHABLE, f"search failed: {error}",
                               ticker=about, margin=margin)
        if looked is None:
            return None
        # Checked once something was found, so a question nothing matched keeps the refusal it always had:
        # 519,135 general passages answer "write me a poem about the sea" out of The Lusiads otherwise.
        if not about and not self.domain.about_money(question, looked.lesson):
            return self._quiet(question, self.domain.OFF_TOPIC, "not about money", margin=margin,
                               asked_writer=looked.asked_writer)
        if looked.how == UNANSWERED:
            return self._quiet(question, self.domain.NOT_IN_SOURCES, "sources do not answer it", ticker=about,
                               evidence=looked.evidence, margin=margin, as_of=looked.day or None,
                               asked_writer=looked.asked_writer)
        return Turn(question, question, about, f"{REFERENCE} {looked.how}", looked.served,
                    looked.answer, looked.evidence, looked.confidence, None, True, margin,
                    looked.unsupported, looked.unread, looked.day or None, looked.phrased_by, looked.unphrased_because,
                    looked.asked_writer)

    def _stated(self, confidence):
        """The chance of being right, in the units B3 fitted. None until a calibrator has been fitted."""
        return None if self.calibrator is None else float(self.calibrator(confidence))

    def _quiet(self, question, served, because, ticker="", intent="", evidence="",
               margin=float("nan"), as_of=None, asked_writer=False):
        """A turn that stopped before the model ran. No confidence, because nothing was generated.

        Empty rather than None for the text a stage never reached: every caller already reads these as
        "nothing routed", and one of them stores them in columns that refuse a null.
        """
        return Turn(question, "", ticker, intent, served, "", evidence, float("nan"), None, False,
                    margin, [], because, as_of, asked_writer=asked_writer)


def answered(turns):
    """(spoke, of how many) over a set of turns, which is the coverage every precision here is on."""
    return sum(turn.spoke for turn in turns), len(turns)


def alfa_spoke(turn):
    """Whether ALFA's own answer was served, which a turn our code's sentence answered (BUILT) did not."""
    return turn.spoke and turn.phrased_by != BUILT


def routing_margins(turns):
    """The margins, for fitting the routing gate on outcomes rather than picking it by hand."""
    return xp.asarray([turn.margin for turn in turns], dtype=xp.float64)
