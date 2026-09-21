"""The Transformer encoder shared by every modality.

It takes already-embedded tokens, never ids, so the text front-end and the price-patch
front-end feed the same stack. That is what makes one model able to read a finance
textbook and a price series without two separate architectures.
"""

from ..autograd import functional as F
from ..autograd import ops
from .attention import MultiHeadAttention
from .layers import Dropout, LayerNorm, Linear
from .module import Module, ModuleList


class TransformerBlock(Module):
    """Pre-LayerNorm. Post-LN diverges at this scale unless the warmup is tuned carefully."""

    def __init__(self, dim, num_heads, ffn_dim, dropout, rng):
        super().__init__()
        self.attention_norm = LayerNorm(dim)
        self.attention = MultiHeadAttention(dim, num_heads, dropout, rng)
        self.ffn_norm = LayerNorm(dim)
        self.ffn_in = Linear(dim, ffn_dim, rng)
        self.ffn_out = Linear(ffn_dim, dim, rng)
        self.dropout = Dropout(dropout, rng)

    def forward(self, x, mask=None):
        attended = self.attention(self.attention_norm(x), mask=mask)
        x = ops.add(x, self.dropout(attended))
        hidden = self.ffn_out(F.gelu(self.ffn_in(self.ffn_norm(x))))
        return ops.add(x, self.dropout(hidden))


class TransformerEncoder(Module):
    def __init__(self, num_layers, dim, num_heads, ffn_dim, dropout, rng):
        super().__init__()
        self.layers = ModuleList(
            [TransformerBlock(dim, num_heads, ffn_dim, dropout, rng) for _ in range(num_layers)]
        )
        # pre-LN leaves the residual stream unnormalised, so the stack needs a final norm
        self.final_norm = LayerNorm(dim)

    def forward(self, x, mask=None):
        for layer in self.layers:
            x = layer(x, mask=mask)
        return self.final_norm(x)
