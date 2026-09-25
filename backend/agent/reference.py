"""Answers to questions the advisory router did not recognise, read out of retrieved documents.

Why this is a separate path and not more intents. The advisory path answers seven things about one
instrument from price indicators, and its generator is measured faithful on them -- 2.4% of answers
state a figure the evidence does not. This path reads human prose, and the generator trained on that
task is not faithful at all: blanking its evidence costs 0.0191 loss, against 1.0-1.2 on the advisory
task, and 57.1% of the figures it writes are unsupported. The diagnosed cause is the training data,
where 30.7% of answer content words are absent from the retrieved input, so a decoder that ignores its
evidence was the correct thing to learn.

So this path **quotes by default**. It retrieves, cites, and serves the passage's own words, and the
paraphrase is generated and graded beside it but not served. That is a measurement decision rather than a
cautious one: on 200 real questions the guard already had to replace 25.5% of paraphrases, and reading the
149 it passed showed them to be fluent and incoherent -- grounded in their figures and wrong about what the
passage says. A guard on figures cannot catch that, so the figure rate was clearing a bar the answers did
not. Quoting makes the claim true for all of them: what the user reads is what a cited source says.

`paraphrase=True` serves the model's words instead, still behind the figure guard, because the generator's
behaviour has to stay demonstrable next to the decision to stop serving it. Either way the grade is
recorded in the turn, and silence is kept for the case where nothing was retrieved at all.

The abstention cut is not consulted here, and this path does not fit it either. That cut relates a
confidence to a correctness rate measured on the advisory generator, and these are different weights on a
different task, so one number could not stand for both. Grounding is what decides this path instead, and
grounding is checked rather than predicted.
"""

from collections import namedtuple

from selfagent.agent import guardrails

from ..retrieval import Hybrid
from .context import assemble

# How many of the question's rare words a live search is given when there is no subject to search by.
# Two, measured against EDGAR, which ANDs them: "tesla deliveries" returns 403 current reports and adding
# two more of the question's words returns none. Wikipedia takes the query whole, because it ranks a
# sentence rather than intersecting terms.
LIVE_TERMS = 2

REFERENCE = "reference"
PARAPHRASED = "paraphrased"
QUOTED = "quoted"


class SearchFailed(RuntimeError):
    """The outside sources could not be read at all, which is not the same as their having nothing.

    Raised here rather than let through as whatever the searcher threw, so the caller has one narrow thing
    to catch and `backend/agent/` still does not import `backend/live/` -- the offline path must not need
    `requests` to exist.
    """

Looked = namedtuple("Looked", "served evidence how unsupported confidence answer day")


class Reference:
    """The retrieval answer path: search, read, and say only what the passages support.

    The index and the model are handed over already loaded, for the reason `Agent` takes its model that
    way -- a served answer and a measured one have to come from the same objects.

    `passages` is both how many chunks to retrieve and how many slots the decoder reads, because those
    were the same number in training and a served row has to be laid out like a trained one.
    """

    def __init__(self, index, tokenizer, model, config, question_tokens, passages,
                 temperature, top_p, rng=None, paraphrase=False, live=None):
        self.paraphrase = paraphrase
        self.index = index
        # Optional, and left out the path makes no network call, which is how the offline gate is still met
        # by this code. Which of it and the index is asked first is `_arms`, and depends on the subject.
        self.live = live
        self.tokenizer = tokenizer
        self.model = model
        self.config = config
        self.question_tokens = question_tokens
        self.passages = passages
        self.temperature = temperature
        self.top_p = top_p
        self.rng = rng

    def look_up(self, question, as_of=None, subject=None):
        """A `Looked` for one question, or None when nothing we can read was about it.

        None rather than a refusal, so the caller keeps owning what silence sounds like.

        `subject` is the instrument the conversation is about, from `finance.Market.subject`. It widens the
        search and it is what the passages are checked against -- the reader is still given the question as
        typed. Without it "how has it been doing" searches for the word "it", which is how a question about
        one company was answered out of a passage about another.
        """
        query = f"{question} {' '.join(subject.names)}" if subject else question
        for arm in self._arms(subject):
            chunks = arm(query, as_of, subject)
            if chunks:
                return self._read(question, chunks)
        return None

    def _arms(self, subject):
        """The arms in the order they are asked, decided by whether the subject has an identity to ask by.

        A company with an SEC number is searched for before the corpus is read, which is measured. On five
        such questions the stored passages all named the ticker and none was about the company -- Stack
        Exchange threads that merely say "MSFT" -- and all five answers were wrong. Asking EDGAR by that
        number and Wikipedia by the registered name returned the matching document five times out of five,
        including the quarter's own earnings exhibit. The corpus holds no news at all, so for a question
        about a company it is the weaker source even when it has something to say.

        The cost of that order, stated because it is a real one: a company question now refuses when the
        network is down, where before it would have been answered out of the corpus. That is the right
        trade against 0 of 5, and it is not a silent one -- the refusal says the source could not be read.

        Anything else reads the corpus first. 519,135 passages cost nothing and a fetch costs a user's wait.
        """
        if self.live is None:
            return (self._stored,)
        return (self._searched, self._stored) if subject is not None and subject.cik \
            else (self._stored, self._searched)

    def _stored(self, query, as_of, subject):
        """The index's own passages, when they are about the query. None is "ask somewhere else"."""
        found = self.index.search(query, self.passages, as_of)
        return self._about([self.index.cite(where) for where in found], subject)

    def _searched(self, query, as_of, subject):
        """Passages fetched for the query, ranked by the same code that ranks the stored ones.

        Gated differently from the stored arm rather than not at all, and the difference rests on how each
        candidate was obtained. A stored passage was picked out of 519,135 by word overlap and has to prove
        it is about this instrument. One fetched for a company with an SEC number was asked for by identity,
        so its provenance is the evidence and re-checking it against a word list would only ask whether the
        search we just ran returned what we searched for.

        **A search term that was a guess is checked, because a guess can be answered wrongly.** For the 51
        instruments SEC's table does not carry there is no name to ask by, only the symbol's own stem, and
        that was measured going wrong: "infy" returned the article on the Royal Canadian Regiment, which
        this path then served against a question about Infosys. So a source asked a guess has to come back
        with something that says the guess. It is a weak check and it is the right strength -- 2 of those 3
        searches were correct and are kept, the wrong one is dropped, and what is left is the honest
        position that this half of our instruments is searched for by one word and cannot do better.

        The subject goes to the sources whole and each decides what to do with it. Rare words are only for
        a question that names nothing: the rarest word in "hows the news about it" is "hows", which no
        filing is about.
        """
        terms = [] if subject else self.index.rare_terms(query, LIVE_TERMS)
        try:
            fetched = self.live.passages(query, terms, as_of, subject)
        except RuntimeError as error:
            raise SearchFailed(str(error)) from error
        if subject is not None and not subject.cik:
            fetched = [chunk for chunk in fetched if subject.named in chunk.text.lower()]
        if not fetched:
            return None
        ranked = Hybrid(fetched, self.index.lexical.tokenize).search(query, self.passages, as_of)
        return [fetched[where] for where in ranked] or None

    def _about(self, chunks, subject):
        """Only the stored `chunks` that name the question's subject, or None when none of them do.

        Filtered rather than just tested, because the first chunk is the one quoted and the one the citation
        names. A set where the third passage mentions the subject would otherwise serve the first, and the
        answer would be attributed to a document it did not come from.

        A threshold on how much of the question the passages carry was tried here and dropped on the
        measurement: weighted by the index's own idf it scored 101 unanswerable ticker questions *higher*
        than 200 answerable ones (median 0.572 against 0.521) and tracked query length rather than fit
        (corr -0.70), which is the flaw it was brought in to fix. Mentioning the subject needs no threshold
        and no tuning: a question about RELIANCE.NS may only be answered out of a passage that says so.

        Two limits, both real and both measured. A question naming no instrument has no subject to check,
        and there the only thing catching a wrong passage is retrieving nothing at all. And a name can be
        present without the passage being about it: 1,495 corpus chunks say "Apple", so six of eight
        company questions were answered from a Stack Exchange thread that merely mentions the ticker -- all
        six wrong, and none of them reaching the filings that would have answered. Word presence cannot
        tell a mention from a subject. A trained reranker over (question, passage) pairs is what would, and
        it is the one piece of this path not built.
        """
        if subject is None:
            return chunks or None
        return [chunk for chunk in chunks if subject.mentions(chunk.text)] or None

    def _read(self, question, chunks):
        """What to serve from passages already judged to be about the question.

        One reader for both origins on purpose: a live answer that went through different code would be a
        second answering path, and the guard that screens figures runs in this one. What says which origin
        a passage had is the document key in the evidence, which is a URL for a fetched one.
        """
        evidence = _as_evidence(chunks)
        quote = chunks[0].text
        if not self.paraphrase:
            # No decoder pass at all, which is most of the cost of a turn. `answer` is the quote rather than
            # empty so that `answer != served` keeps meaning "a guard changed this", and `how` is what says
            # whose words these are. The confidence is absent because the model chose none of them.
            return Looked(quote, evidence, QUOTED, [], float("nan"), quote, chunks[0].day)

        # Cut to the slot: the index is chunked for retrieval, at a window measured on whether it finds
        # the endorsed answer, and that window is wider than this decoder's. Quoting serves the passage
        # whole; only the measured paraphrase is limited to the opening of it.
        context = assemble(self.tokenizer, question, [chunk.text for chunk in chunks],
                           self.config.max_text_length, self.question_tokens, self.passages, cut=True)
        produced = self.model.generate(context.source, context.keep, self.temperature, self.top_p,
                                       self.rng)
        confidence = float(self.model.confidence(context.source, produced, context.keep)[0])
        written = self.tokenizer.decode(produced[0])

        # Checked against the passage text alone, not the whole evidence line: the citation carries a date
        # and a document key full of digits, and a figure matching those is not one the source stated.
        unsupported = guardrails.unsupported_figures(written, context.evidence)
        return Looked(quote if unsupported else written, evidence,
                      QUOTED if unsupported else PARAPHRASED, unsupported, confidence, written,
                      chunks[0].day)


def _as_evidence(chunks):
    """The passages in the same name-value shape the advisory evidence uses, so one reader serves both.

    The document key and date come first, because a quote the user cannot attribute is worth less than no
    quote. The key is the hash `data/manifest.jsonl` files the source URL and licence under.
    """
    lines = [f"document {chunks[0].document}", f"published {chunks[0].day or 'undated'}"]
    # Numbered with an underscore, which is the separator the existing names use and which the reader
    # turns back into a space: repeating the bare name would collide in a panel keyed by it.
    lines += [f"passage_{index} {chunk.text}" for index, chunk in enumerate(chunks, 1)]
    return " ; ".join(lines)
