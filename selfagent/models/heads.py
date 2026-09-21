"""Output heads. Each maps encoder states to one task's predictions."""

from ..autograd import functional as F
from ..autograd import ops
from ..nn import init
from ..nn.layers import Dropout, LayerNorm, Linear
from ..nn.module import Module
from ..tokenizer.vocab import CLS_POSITION


class MaskedLanguageModelHead(Module):
    """Scores the vocabulary at a position: the masked token in pretraining, the next one when
    the generator decodes.

    The output matrix is tied to the input token embedding, which is passed to forward()
    rather than stored: holding it as an attribute would register the same parameter twice
    and the optimizer would then apply its update twice per step.
    """

    def __init__(self, config, rng):
        super().__init__()
        self.transform = Linear(config.dim, config.dim, rng)
        self.norm = LayerNorm(config.dim)
        self.output_bias = init.zeros(config.vocab_size)

    def forward(self, hidden, token_embedding_weight):
        transformed = self.norm(F.gelu(self.transform(hidden)))
        logits = ops.matmul(transformed, ops.transpose(token_embedding_weight, (1, 0)))
        return ops.add(logits, self.output_bias)


class ClassificationHead(Module):
    """Reads the [CLS] position of a sequence, or a fused vector, and scores the classes."""

    def __init__(self, in_features, num_classes, dropout, rng):
        super().__init__()
        self.dropout = Dropout(dropout, rng)
        self.project = Linear(in_features, num_classes, rng)

    def forward(self, features):
        return self.project(self.dropout(features))


def pool_cls(sequence):
    return ops.getitem(sequence, (slice(None), CLS_POSITION))
