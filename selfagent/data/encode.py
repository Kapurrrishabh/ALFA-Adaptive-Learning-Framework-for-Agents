"""Turn a question and its passages into the array the encoder reads.

This lives here rather than in the script that first needed it because serving has to lay a row out
exactly as training did. A served question assembled with a different separator, cap or slot count is
a different input distribution, and the model would answer worse for a reason no metric would name.
"""

from ..backend import xp
from ..tokenizer.vocab import CLS_ID, PAD_ID, SEP_ID


def build_sources(question_ids, passage_ids_list, passages_wanted, length):
    """(passages, length) ids and the matching 1/0 mask for one row.

    Every row carries the same number of passage slots so the set is one array. A row that retrieved
    fewer keeps empty slots, fully masked, which the encoder then ignores.
    """
    source = xp.full((passages_wanted, length), PAD_ID, dtype=xp.uint16)
    keep = xp.zeros((passages_wanted, length), dtype=xp.uint8)
    for slot, passage_ids in enumerate(passage_ids_list[:passages_wanted]):
        row = [CLS_ID] + question_ids + [SEP_ID] + passage_ids
        row = row[: length - 1] + [SEP_ID]
        source[slot, : len(row)] = row
        keep[slot, : len(row)] = 1
    return source, keep
