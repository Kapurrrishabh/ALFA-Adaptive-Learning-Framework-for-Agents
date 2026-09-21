"""The assembled models.

Both towers wrap the same TransformerEncoder class. A finance textbook and a price window
are different front-ends onto one architecture, which is what lets one agent read both.
"""

from ..autograd import ops
from ..backend import default_rng
from ..nn.attention import padding_mask
from ..nn.encoder import TransformerEncoder
from ..nn.module import Module
from ..nn.recurrent import GRU
from .embeddings import PricePatchEmbedding, TextEmbedding
from .heads import ClassificationHead, MaskedLanguageModelHead, pool_cls


class TextTower(Module):
    def __init__(self, config, rng):
        super().__init__()
        self.embedding = TextEmbedding(config, rng)
        self.encoder = TransformerEncoder(
            config.num_layers, config.dim, config.num_heads, config.ffn_dim, config.dropout, rng
        )

    def forward(self, token_ids, is_real_token=None):
        mask = None if is_real_token is None else padding_mask(is_real_token)
        return self.encoder(self.embedding(token_ids), mask=mask)


class PriceTower(Module):
    def __init__(self, config, rng):
        super().__init__()
        self.embedding = PricePatchEmbedding(config, rng)
        self.encoder = TransformerEncoder(
            config.price_layers, config.dim, config.num_heads, config.ffn_dim, config.dropout, rng
        )

    def forward(self, window):
        # Every patch is real: windows are fixed length, so there is nothing to pad.
        return self.encoder(self.embedding(window))


class RecurrentPriceTower(Module):
    """The GRU alternative to PriceTower, reading every bar instead of patches.

    Patching exists to keep attention affordable; a recurrent pass is already linear in the window,
    so it can keep the per-bar resolution that patching averages away.
    """

    def __init__(self, config, rng):
        super().__init__()
        self.gru = GRU(config.price_channels, config.dim, rng)

    def forward(self, window):
        return self.gru(window)


class PriceWindowClassifier(Module):
    """A tertile class over a forward horizon from a price window alone, with either tower.

    Which forward quantity is being classified is the caller's choice: direction measured
    unpredictable on this data, so the label that carries signal is volatility. The generator only
    ever phrases what this produces, so this is the part that decides whether the advice is worth
    anything.
    """

    def __init__(self, config, rng=None, recurrent=False):
        super().__init__()
        rng = default_rng(config.seed) if rng is None else rng
        self.config = config
        self.price = RecurrentPriceTower(config, rng) if recurrent else PriceTower(config, rng)
        self.classifier = ClassificationHead(config.dim, config.num_classes, config.dropout, rng)

    def forward(self, window):
        return self.classifier(ops.mean(self.price(window), axis=1))


class MaskedLanguageModel(Module):
    """Pretraining model. Learns finance language from unlabelled text such as textbooks."""

    def __init__(self, config, rng=None):
        super().__init__()
        rng = default_rng(config.seed) if rng is None else rng
        self.config = config
        self.text = TextTower(config, rng)
        self.head = MaskedLanguageModelHead(config, rng)

    def forward(self, token_ids, is_real_token=None):
        hidden = self.text(token_ids, is_real_token)
        return self.head(hidden, self.text.embedding.tokens.weight)


class FinanceAgentModel(Module):
    """Reads news text and a price window, fuses them, and scores the task classes.

    Fusion is late concatenation of the two pooled vectors. Cross-attention is the planned
    upgrade and is kept out until this version's number exists to compare against.
    """

    def __init__(self, config, rng=None):
        super().__init__()
        rng = default_rng(config.seed) if rng is None else rng
        self.config = config
        self.text = TextTower(config, rng)
        self.price = PriceTower(config, rng)
        self.classifier = ClassificationHead(
            2 * config.dim, config.num_classes, config.dropout, rng
        )

    def forward(self, token_ids, price_window, is_real_token=None):
        text_summary = pool_cls(self.text(token_ids, is_real_token))
        price_summary = ops.mean(self.price(price_window), axis=1)
        return self.classifier(ops.concat([text_summary, price_summary], axis=-1))
