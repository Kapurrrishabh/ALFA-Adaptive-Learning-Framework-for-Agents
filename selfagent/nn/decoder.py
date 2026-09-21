"""The decoder that writes the answer.

Every block cross-attends to the encoder's output, so generation is grounded rather than free: the
decoder's job is to stitch and paraphrase text it can see, not to recall facts from its own weights.
At this parameter count recall is what a model fabricates and copying is what it can actually do,
which is the difference between an advisory that cites and one that invents.
"""

from ..autograd import functional as F
from ..autograd import ops
from .attention import MultiHeadAttention, causal_mask
from .layers import Dropout, LayerNorm, Linear
from .module import Module, ModuleList


class TransformerDecoderBlock(Module):
    """Pre-LayerNorm, matching TransformerBlock: causal self-attention, then cross-attention."""

    def __init__(self, dim, num_heads, ffn_dim, dropout, rng):
        super().__init__()
        self.self_norm = LayerNorm(dim)
        self.self_attention = MultiHeadAttention(dim, num_heads, dropout, rng)
        self.cross_norm = LayerNorm(dim)
        self.cross_attention = MultiHeadAttention(dim, num_heads, dropout, rng)
        self.ffn_norm = LayerNorm(dim)
        self.ffn_in = Linear(dim, ffn_dim, rng)
        self.ffn_out = Linear(ffn_dim, dim, rng)
        self.dropout = Dropout(dropout, rng)

    def forward(self, x, memory, causal, memory_mask=None):
        attended = self.self_attention(self.self_norm(x), mask=causal)
        x = ops.add(x, self.dropout(attended))
        crossed = self.cross_attention(self.cross_norm(x), context=memory, mask=memory_mask)
        x = ops.add(x, self.dropout(crossed))
        hidden = self.ffn_out(F.gelu(self.ffn_in(self.ffn_norm(x))))
        return ops.add(x, self.dropout(hidden))


class TransformerDecoder(Module):
    def __init__(self, num_layers, dim, num_heads, ffn_dim, dropout, rng):
        super().__init__()
        self.layers = ModuleList(
            [
                TransformerDecoderBlock(dim, num_heads, ffn_dim, dropout, rng)
                for _ in range(num_layers)
            ]
        )
        # pre-LN leaves the residual stream unnormalised, so the stack needs a final norm
        self.final_norm = LayerNorm(dim)

    def forward(self, x, memory, memory_mask=None):
        causal = causal_mask(x.shape[1])
        for layer in self.layers:
            x = layer(x, memory, causal, memory_mask=memory_mask)
        return self.final_norm(x)
