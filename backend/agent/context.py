"""Everything the decoder reads for one turn, assembled in one place and laid out as training laid it.

There is one rule here and it is the reason this is not inlined into the caller: a served row must be
built by the same code that built a training row. The caps, the separators and the passage-slot count
are all part of the input distribution the model learned, and a serving path that quietly used its own
would lose accuracy for a reason no metric in the repo would name.
"""

from collections import namedtuple

from selfagent.backend import xp
from selfagent.data.encode import build_sources

# `question` is what the decoder was actually given, which after routing is not what the user typed.
# Both are kept because a turn that answered the wrong question is unreadable without them side by side.
Context = namedtuple("Context", "question evidence source keep")


def assemble(tokenizer, question, passages, length, question_tokens, slots=1, cut=False):
    """The model input for one turn, and the text it was built from.

    `cut` is for the caller whose passages are sized for retrieval rather than for this decoder: it feeds
    each one's opening `room` tokens and reports those as the evidence. Cut here rather than left to
    `build_sources`, which drops the same tokens but not from the evidence line, and a figure checked
    against text the model never read is not checked.
    """
    passage_ids = [tokenizer.encode(passage) for passage in passages]
    room = length - question_tokens - 3
    if cut:
        shortened = [ids[:room] for ids in passage_ids]
        # Only what was actually cut is decoded back: encoding and decoding a passage that already fits
        # would round-trip its casing and spacing for nothing.
        passages = [text if len(ids) <= room else tokenizer.decode(short)
                    for text, ids, short in zip(passages, passage_ids, shortened)]
        passage_ids = shortened
    over = [len(ids) for ids in passage_ids if len(ids) > room]
    if over:
        raise ValueError(
            f"evidence of {over} tokens exceeds the {room} this window leaves; a cut passage drops a "
            f"figure the answer quotes, and the guard would then refuse every answer"
        )
    source, keep = build_sources(tokenizer.encode(question)[:question_tokens], passage_ids, slots, length)
    # The model works in batches; one turn is a batch of one.
    return Context(question, " ".join(passages), xp.asarray(source)[None], xp.asarray(keep)[None])
