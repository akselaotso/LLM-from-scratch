import torch.nn as nn, torch
from .LayerNormalization import LayerNormalization

class SimpleMultiHeadLatentAttention(nn.Module):
    def __init__(self, dimension, latent_dimension, context_length, dropout, num_heads, bias = False, causal = True):
        super().__init__()

        self.d_out = dimension
        self.num_heads = num_heads
        self.head_dim = dimension // num_heads
        self.causal = causal

        self.dropout = nn.Dropout(dropout)
        self.out_projection = nn.Linear(dimension, dimension)

        self.W_dkv = nn.Linear(dimension, latent_dimension, bias=bias)
        self.W_query = nn.Linear(dimension, dimension, bias=bias)
        self.W_key   = nn.Linear(latent_dimension, dimension, bias=bias)
        self.W_value = nn.Linear(latent_dimension, dimension, bias=bias)

        self.ln = LayerNormalization(latent_dimension)

        self.register_buffer("C_kv", None)

        if causal:
            self.register_buffer("mask", torch.triu(torch.ones(context_length, context_length), diagonal=1))

    def forward(self, x):
        b, n_tokens, dimension = x.shape

        queries = self.W_query(x)

        new_C_kv = self.ln(self.W_dkv(x))
        self.C_kv = torch.cat([self.C_kv, new_C_kv], dim=1) if self.C_kv != None else new_C_kv

        keys = self.W_key(self.C_kv)
        values = self.W_value(self.C_kv)

        keys = keys.view(b, n_tokens, self.num_heads, self.head_dim)
        queries = queries.view(b, n_tokens, self.num_heads, self.head_dim)
        values = values.view(b, n_tokens, self.num_heads, self.head_dim)

        keys = keys.transpose(1, 2)
        queries = queries.transpose(1, 2)
        values = values.transpose(1, 2)

        attention_scores = queries @ keys.transpose(2, 3)

        if self.causal:
            attention_scores.masked_fill(self.mask.bool()[:n_tokens, :n_tokens], -torch.inf) 

        attention_weights = torch.softmax(attention_scores / keys.shape[-1]**0.5, dim=-1)

        attention_weights = self.dropout(attention_weights)

        context_vector = (attention_weights @ values).transpose(1, 2).contiguous().view(b, n_tokens, self.d_out)
        context_vector = self.out_projection(context_vector)

        return context_vector

