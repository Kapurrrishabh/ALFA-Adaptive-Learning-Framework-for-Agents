"""Modality front-ends. Each turns raw input into (batch, length, dim) for the shared encoder."""

from ..autograd import functional as F
from ..autograd import ops
from ..backend import xp
from ..nn.layers import Dropout, Embedding, LayerNorm, Linear
from ..nn.module import Module


class TextEmbedding(Module):
    """Token ids plus a learned position. Absolute positions are enough at 128 tokens."""

    def __init__(self, config, rng):
        super().__init__()
        self.tokens = Embedding(config.vocab_size, config.dim, rng)
        self.positions = Embedding(config.max_text_length, config.dim, rng)
        self.norm = LayerNorm(config.dim)
        self.dropout = Dropout(config.dropout, rng)
        self.max_length = config.max_text_length

    def forward(self, token_ids):
        token_ids = xp.asarray(token_ids)
        if token_ids.ndim != 2:
            raise ValueError(f"expected (batch, length) token ids, got shape {token_ids.shape}")
        length = token_ids.shape[1]
        if length > self.max_length:
            raise ValueError(
                f"sequence of {length} tokens exceeds max_text_length {self.max_length}; "
                "chunk the document in the data layer"
            )
        embedded = ops.add(self.tokens(token_ids), self.positions(xp.arange(length)[None, :]))
        return self.dropout(self.norm(embedded))


class AnswerEmbedding(Module):
    """The answer side of the generator. Positions are its own, but the token matrix belongs to
    the encoder and arrives through forward: registering it here as well would give the optimizer
    two names for one parameter and it would apply every update twice.
    """

    def __init__(self, config, rng):
        super().__init__()
        self.positions = Embedding(config.max_answer_length, config.dim, rng)
        self.norm = LayerNorm(config.dim)
        self.dropout = Dropout(config.dropout, rng)
        self.max_length = config.max_answer_length

    def forward(self, token_ids, token_embedding_weight):
        token_ids = xp.asarray(token_ids)
        if token_ids.ndim != 2:
            raise ValueError(f"expected (batch, length) answer ids, got shape {token_ids.shape}")
        length = token_ids.shape[1]
        if length > self.max_length:
            raise ValueError(
                f"answer of {length} tokens exceeds max_answer_length {self.max_length}; "
                "truncate the target in the data layer"
            )
        embedded = ops.add(
            F.embedding(token_embedding_weight, token_ids),
            self.positions(xp.arange(length)[None, :]),
        )
        return self.dropout(self.norm(embedded))


class PricePatchEmbedding(Module):
    """Groups consecutive timesteps into patches, one token per patch (PatchTST, 2023).

    Channels are mixed inside a patch rather than kept independent: open/high/low/close/volume
    at one timestep describe one event, so mixing them is the natural bias here. Channel
    independence is the alternative to measure once the supervised task is running.
    """

    def __init__(self, config, rng):
        super().__init__()
        self.patch_length = config.patch_length
        self.patch_stride = config.patch_stride
        self.channels = config.price_channels
        self.project = Linear(config.patch_length * config.price_channels, config.dim, rng)
        self.positions = Embedding(config.num_patches, config.dim, rng)
        self.norm = LayerNorm(config.dim)
        self.dropout = Dropout(config.dropout, rng)

    def forward(self, window):
        """`window` is (batch, timesteps, channels) of already-normalised features."""
        window = xp.asarray(window)
        if window.ndim != 3 or window.shape[2] != self.channels:
            raise ValueError(
                f"expected (batch, timesteps, {self.channels}) price window, got {window.shape}"
            )
        patches = self._to_patches(window)
        embedded = ops.add(self.project(patches), self.positions(xp.arange(patches.shape[1])[None, :]))
        return self.dropout(self.norm(embedded))

    def _to_patches(self, window):
        batch, timesteps, channels = window.shape
        starts = range(0, timesteps - self.patch_length + 1, self.patch_stride)
        stacked = xp.stack([window[:, start : start + self.patch_length] for start in starts], axis=1)
        return stacked.reshape(batch, len(starts), self.patch_length * channels)
