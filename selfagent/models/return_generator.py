"""A generative model of daily returns: a causal transformer over return tokens, trained on the log score.

Each position reads the returns up to and including its own and predicts a distribution over the next one.
Trained with cross-entropy, that is maximum likelihood on the discretised returns, and cross-entropy is the
log score -- strictly proper, so the only way to lower it in expectation is to put the true distribution's
shape on the next return. The model is never told what a fat tail or a volatility cluster is. If it learns
them, they show up in the paths it writes, and `learn/forecast.py` checks that they do.

It says nothing reliable about direction. The literature and this repository agree on that: time-series
foundation models forecasting 1,000 US equities collapse to flat predictions (arXiv:2608.14106), and the
price head here scored 35.7% on three balanced direction classes against 36.0% for the majority class. What
a return history does carry is its own scale -- trailing volatility explains 22.1% of the next week's --
so the product of this model is a distribution and the paths drawn from it, not a call.

**Measured** from January 2023 to September 2026, with a model trained before 2021 and chosen on 2021-22:
its 875,776 parameters beat GARCH(1,1)-t on the one-step log score by 0.0323 +/- 0.0032 nats per return
over 80,576 returns, and a trailing-volatility Gaussian by 0.154 +/- 0.007. Its sampled paths have a tail
index of 2.98 against 3.08 real, and a leverage effect GARCH cannot produce (-0.025 against -0.017 real,
-0.002 GARCH). Two things it does not do: multi-day path CRPS ties both baselines within one standard
error, and it clusters volatility more than the real test period did (lag 1-5 autocorrelation of |r|
0.076 against 0.047). Direction from P(up) > 0.5 is right 51.0% against 51.2% for always calling the
commoner side.

Sampling uses the full distribution at temperature 1. A nucleus cut, which the text generator needs, would
delete exactly the tail a return model exists to get right.
"""

from ..autograd import no_grad, ops
from ..backend import default_rng, xp
from ..data import returns
from ..nn.attention import causal_mask
from ..nn.encoder import TransformerEncoder
from ..nn.layers import Dropout, Embedding, LayerNorm, Linear
from ..nn.module import Module


class ReturnGenerator(Module):
    """`config.vocab_size` is the number of return bins and `config.price_window` the context length."""

    def __init__(self, config, rng=None):
        super().__init__()
        if config.vocab_size != returns.BINS:
            raise ValueError(f"config.vocab_size is {config.vocab_size} but returns are tokenised into "
                             f"{returns.BINS} bins; build the config with vocab_size=returns.BINS")
        rng = default_rng(config.seed) if rng is None else rng
        self.config = config
        self.tokens = Embedding(returns.BINS, config.dim, rng)
        self.positions = Embedding(config.price_window, config.dim, rng)
        self.norm = LayerNorm(config.dim)
        self.dropout = Dropout(config.dropout, rng)
        self.encoder = TransformerEncoder(config.num_layers, config.dim, config.num_heads, config.ffn_dim,
                                          config.dropout, rng)
        self.head = Linear(config.dim, returns.BINS, rng)

    def forward(self, ids):
        """(batch, length, BINS) logits; position t is the distribution of the return after token t."""
        ids = xp.asarray(ids)
        length = ids.shape[1]
        if length > self.config.price_window:
            raise ValueError(f"{length} return tokens exceed the {self.config.price_window}-token context; "
                             f"pass the most recent {self.config.price_window}")
        embedded = ops.add(self.tokens(ids), self.positions(xp.arange(length)[None, :]))
        hidden = self.encoder(self.dropout(self.norm(embedded)), mask=causal_mask(length))
        return self.head(hidden)

    def next_probabilities(self, ids):
        """(batch, BINS) distribution of the return that follows each row of `ids`."""
        context = xp.asarray(ids)[:, -self.config.price_window :]
        with no_grad():
            logits = self(context).data[:, -1]
        logits = logits - logits.max(axis=-1, keepdims=True)
        weights = xp.exp(logits)
        return weights / weights.sum(axis=-1, keepdims=True)

    def sample_paths(self, contexts, steps, rng=None):
        """(rows, steps) token ids continuing each row of `contexts`, drawn from the model's own distribution.

        The window slides once a path outgrows the context, so a path can run far longer than anything
        trained on, and its tokens stay in the one scale unit its context was measured in.
        """
        rng = default_rng(self.config.seed) if rng is None else rng
        running = xp.asarray(contexts, dtype=xp.int64)
        was_training = self.training
        self.eval()
        try:
            for _ in range(steps):
                probabilities = self.next_probabilities(running)
                drawn = (probabilities.cumsum(axis=-1) < rng.random((len(running), 1))).sum(axis=-1)
                # Rounding can leave the cumulative sum just under 1.0 and push the draw off the end.
                drawn = xp.minimum(drawn, returns.BINS - 1)
                running = xp.concatenate([running, drawn[:, None]], axis=1)
        finally:
            self.train(was_training)
        return running[:, running.shape[1] - steps :]
