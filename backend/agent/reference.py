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

from .context import assemble

# Kept at the value that never fires, which is what the measurement showed it can honestly be. On 200 real
# questions the top BM25 score ran 36.59 at the 10th percentile to 169.41, none under 1.0, and the lexical
# arm already returns nothing when no query term appears at all. Raising it would reject short questions
# rather than irrelevant passages, because the magnitude tracks query length: the worst retrieval in the
# run -- algebra homework served to a question about a spinoff -- scored well above any cut worth setting.
# Separating those needs a relevance judgement, not a threshold on this number.
SCORE_FLOOR = 1.0

REFERENCE = "reference"
PARAPHRASED = "paraphrased"
QUOTED = "quoted"

Looked = namedtuple("Looked", "served evidence how unsupported confidence answer day")


class Reference:
    """The retrieval answer path: search, read, and say only what the passages support.

    The index and the model are handed over already loaded, for the reason `Agent` takes its model that
    way -- a served answer and a measured one have to come from the same objects.

    `passages` is both how many chunks to retrieve and how many slots the decoder reads, because those
    were the same number in training and a served row has to be laid out like a trained one.
    """

    def __init__(self, index, tokenizer, model, config, question_tokens, passages,
                 temperature, top_p, rng=None, floor=SCORE_FLOOR, paraphrase=False):
        self.paraphrase = paraphrase
        self.index = index
        self.tokenizer = tokenizer
        self.model = model
        self.config = config
        self.question_tokens = question_tokens
        self.passages = passages
        self.temperature = temperature
        self.top_p = top_p
        self.rng = rng
        self.floor = floor

    def look_up(self, question, as_of=None):
        """A `Looked` for one question, or None when the corpus had nothing to say about it.

        None rather than a refusal, so the caller keeps owning what silence sounds like.
        """
        found = self.index.search(question, self.passages, as_of)
        if not found or self.index.relevance(question, found[0]) < self.floor:
            return None

        chunks = [self.index.cite(where) for where in found]
        evidence = _as_evidence(chunks)
        quote = chunks[0].text
        if not self.paraphrase:
            # No decoder pass at all, which is most of the cost of a turn. `answer` is the quote rather than
            # empty so that `answer != served` keeps meaning "a guard changed this", and `how` is what says
            # whose words these are. The confidence is absent because the model chose none of them.
            return Looked(quote, evidence, QUOTED, [], float("nan"), quote, chunks[0].day)

        context = assemble(self.tokenizer, question, [chunk.text for chunk in chunks],
                           self.config.max_text_length, self.question_tokens, self.passages)
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
