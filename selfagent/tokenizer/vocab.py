"""Special tokens. Declared once; the tokenizer, the MLM masker and batching all read these."""

PAD = "[PAD]"
UNK = "[UNK]"
CLS = "[CLS]"
SEP = "[SEP]"
MASK = "[MASK]"

# Order fixes the ids, so a trained checkpoint stays readable. Do not reorder.
SPECIAL_TOKENS = (PAD, UNK, CLS, SEP, MASK)

PAD_ID = SPECIAL_TOKENS.index(PAD)
UNK_ID = SPECIAL_TOKENS.index(UNK)
CLS_ID = SPECIAL_TOKENS.index(CLS)
SEP_ID = SPECIAL_TOKENS.index(SEP)
MASK_ID = SPECIAL_TOKENS.index(MASK)

# Marks a piece that continues the previous one rather than starting a word. Whole-word masking
# reads it to decide which pieces have to be masked together.
CONTINUATION = "##"

# Every sequence starts with [CLS]; pooling for classification reads position 0.
CLS_POSITION = 0

# Positions cross_entropy should skip, matching its ignore_index default.
UNLABELLED = -100
