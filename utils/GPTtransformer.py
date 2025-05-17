import torch.nn as nn, torch.cuda as cuda
from .MultiHeadAttention import MultiHeadAttention
from .LayerNormalization import LayerNormalization
from .FeedForward import FeedForward


class GPTTransformerBlock(nn.Module):
    def __init__(self, context_length, dimension, num_heads=4, dropout=0.1, bias=False):
        super().__init__()

        self.layernorm1 = LayerNormalization(dimension)
        self.attention1 = MultiHeadAttention(dimension, dimension, context_length, dropout=dropout, num_heads=num_heads, bias=bias)
        self.dropout1 = nn.Dropout(dropout)

        self.layernorm2 = LayerNormalization(dimension)
        self.feedforward2 = FeedForward(dimension)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x):
        shortcut = x
        x = self.layernorm1(x)
        x = self.attention1(x)
        x = self.dropout1(x)
        x = x + shortcut
        
        shortcut = x
        x = self.layernorm2(x)
        x = self.feedforward2(x)
        x = self.dropout2(x)
        x = x + shortcut

        return x

