import torch.nn as nn, torch
from .GPTtransformer import GPTTransformerBlock
from .LayerNormalization import LayerNormalization


class GPTLargeLanguageModel(nn.Module):
    def __init__(self, vocab_size=50257, num_layers=12, context_length=1024, dimension=768, num_heads=12, dropout=0.1, bias=False):
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, dimension)
        self.positional_encoding = nn.Embedding(context_length, dimension)
        self.encoding_dropout = nn.Dropout(dropout)

        self.transformers = nn.Sequential(
            *[GPTTransformerBlock(context_length, dimension, num_heads, dropout, bias) for _ in range(num_layers)]
        )

        self.final_normalization = LayerNormalization(dimension)

        self.out = nn.Linear(dimension, vocab_size, bias=bias)

    def forward(self, x):
        b, len = x.shape

        embeddings = self.embedding(x)
        positions = self.positional_encoding(torch.arange(len, device=x.device))

        x = embeddings + positions
        x = self.encoding_dropout(x)
        x = self.transformers(x)
        x = self.final_normalization(x)

        output = self.out(x)

        return output

    