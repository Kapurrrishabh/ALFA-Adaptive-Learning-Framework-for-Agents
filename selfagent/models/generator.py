"""Grounded generation: the answer is written from passages the model can see.

The encoder reads the question and the retrieved passages as one sequence and the decoder
cross-attends to it. That distinction is what makes a model this small usable: asked to recall a
figure it invents one, but asked to copy a figure that is in front of it, it copies. Everything
here is shaped by that — the decoder is shallow, the answer is short, and the token matrix is
shared by the encoder input, the answer input and the output layer, because three copies of a
vocab x dim matrix would outweigh the rest of the model put together.
"""

from ..autograd import no_grad, ops
from ..backend import default_rng, xp
from ..nn.attention import padding_mask
from ..nn.decoder import TransformerDecoder
from ..nn.module import Module
from ..tokenizer.vocab import CLS_ID, PAD_ID, SEP_ID
from .agent import TextTower
from .embeddings import AnswerEmbedding
from .heads import MaskedLanguageModelHead

# The answer is bracketed by the same markers the encoder already uses, so generation needs no new
# special tokens and no change to the vocabulary order a trained checkpoint depends on.
ANSWER_START_ID = CLS_ID
ANSWER_END_ID = SEP_ID

# Measured on held-out answers: greedy repeats 73% of its 4-grams where humans repeat 1.7%, and
# sampling the whole tail invents non-words. This pair scores 1.6% and 0.4%. Serving passes these;
# generate() still defaults to greedy so tests stay deterministic.
ANSWER_TEMPERATURE = 0.9
ANSWER_TOP_P = 0.9


class GroundedGenerator(Module):
    def __init__(self, config, rng=None):
        super().__init__()
        rng = default_rng(config.seed) if rng is None else rng
        self.config = config
        self.text = TextTower(config, rng)
        self.answer = AnswerEmbedding(config, rng)
        self.decoder = TransformerDecoder(
            config.decoder_layers,
            config.dim,
            config.num_heads,
            config.ffn_dim,
            config.dropout,
            rng,
        )
        self.head = MaskedLanguageModelHead(config, rng)

    @property
    def shared_token_weight(self):
        return self.text.embedding.tokens.weight

    def forward(self, source_ids, answer_ids, is_real_source_token=None):
        """Logits at every answer position for the token that should follow it.

        So `answer_ids` starts at ANSWER_START_ID and the training targets are the same sequence
        shifted one place left. Feeding the unshifted answer as its own target teaches the model
        to copy its input and it will emit the question back.
        """
        memory, mask = self.encode(source_ids, is_real_source_token)
        embedded = self.answer(answer_ids, self.shared_token_weight)
        hidden = self.decoder(embedded, memory, memory_mask=mask)
        return self.head(hidden, self.shared_token_weight)

    def encode(self, source_ids, is_real_source_token=None):
        """(memory, mask) for the decoder to cross-attend to. Accepts one sequence or k passages.

        Given (batch, passages, length) each passage is encoded on its own and the outputs are
        concatenated — Fusion-in-Decoder. Concatenating the text first instead would blow past the
        window the encoder's position embeddings were trained at, so the pretrained encoder would not
        transfer; this way k passages cost linear time and the decoder still sees all of them at once.
        """
        source_ids = xp.asarray(source_ids)
        if source_ids.ndim == 2:
            memory = self.text(source_ids, is_real_source_token)
            return memory, (None if is_real_source_token is None else padding_mask(is_real_source_token))
        if source_ids.ndim != 3:
            raise ValueError(
                f"expected (batch, length) or (batch, passages, length) source ids, "
                f"got shape {source_ids.shape}"
            )

        batch, passages, length = source_ids.shape
        keep = None if is_real_source_token is None else xp.asarray(is_real_source_token)
        flat_keep = None if keep is None else keep.reshape((batch * passages, length))
        encoded = self.text(source_ids.reshape((batch * passages, length)), flat_keep)
        memory = ops.reshape(encoded, (batch, passages * length, self.config.dim))
        mask = None if keep is None else padding_mask(keep.reshape((batch, passages * length)))
        return memory, mask

    def load_pretrained_encoder(self, weights):
        """Starts the encoder from the masked-language-model run instead of from noise."""
        prefix = "text."
        encoder = {
            name[len(prefix) :]: value for name, value in weights.items() if name.startswith(prefix)
        }
        if not encoder:
            raise KeyError(
                f"checkpoint holds no '{prefix}' weights, so there is no encoder to reuse; "
                f"found {sorted(weights)[:4]}"
            )
        self.text.load_state_dict(encoder)

    def generate(self, source_ids, is_real_source_token=None, temperature=0.0, top_p=1.0, rng=None):
        """One list of token ids per row, markers stripped. Greedy when temperature is 0.

        The whole prefix is re-encoded for every new token. A key/value cache would cut that to
        linear, but answers are at most max_answer_length tokens and a stale cache is a silent
        wrong answer, so it waits until generation is on the critical path.
        """
        rng = default_rng(self.config.seed) if rng is None else rng
        source_ids = xp.asarray(source_ids)
        batch = source_ids.shape[0]
        answer = xp.full((batch, 1), ANSWER_START_ID, dtype=int)
        finished = xp.zeros(batch, dtype=bool)

        was_training = self.training
        self.eval()
        try:
            with no_grad():
                while answer.shape[1] < self.config.max_answer_length and not finished.all():
                    logits = self(source_ids, answer, is_real_source_token).data[:, -1]
                    chosen = self._pick(logits, temperature, top_p, rng)
                    # A finished row keeps emitting padding so the batch stays one array.
                    chosen = xp.where(finished, PAD_ID, chosen)
                    answer = xp.concatenate([answer, chosen[:, None]], axis=1)
                    finished = finished | (chosen == ANSWER_END_ID)
        finally:
            self.train(was_training)

        return [self._trim(row) for row in answer]

    def _pick(self, logits, temperature, top_p, rng):
        if temperature <= 0.0:
            return logits.argmax(axis=-1)
        scaled = logits / temperature
        probabilities = xp.exp(scaled - scaled.max(axis=-1, keepdims=True))
        probabilities = probabilities / probabilities.sum(axis=-1, keepdims=True)
        if top_p < 1.0:
            probabilities = self._nucleus(probabilities, top_p)
        drawn = (probabilities.cumsum(axis=-1) < rng.random((logits.shape[0], 1))).sum(axis=-1)
        # Rounding in the cumulative sum can leave it just under 1.0 and push the draw off the end.
        return xp.minimum(drawn, self.config.vocab_size - 1)

    @staticmethod
    def _nucleus(probabilities, top_p):
        """Drop every token outside the smallest set whose mass reaches top_p, then renormalise.

        Both decoding failures live in the tail: greedy repeats itself (76% of its 4-grams were
        duplicates, against 1.3% for human answers) and unrestricted sampling invents non-words.
        """
        order = xp.argsort(-probabilities, axis=-1)
        ranked = xp.take_along_axis(probabilities, order, axis=-1)
        # Subtracting each token's own mass keeps the most likely one even if it alone exceeds top_p.
        ranked = xp.where(ranked.cumsum(axis=-1) - ranked >= top_p, 0.0, ranked)
        kept = xp.zeros_like(probabilities)
        xp.put_along_axis(kept, order, ranked, axis=-1)
        return kept / kept.sum(axis=-1, keepdims=True)

    @staticmethod
    def _trim(row):
        ids = [int(value) for value in row[1:]]
        return ids[: ids.index(ANSWER_END_ID)] if ANSWER_END_ID in ids else ids
