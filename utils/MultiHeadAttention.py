import torch.nn as nn, torch

class MultiHeadAttention(nn.Module):
    def __init__(self, d_in, d_out, context_length, dropout, num_heads, bias = False, causal = True):
        super().__init__()

        self.d_out = d_out
        self.num_heads = num_heads
        self.head_dim = d_out // num_heads
        self.causal = causal

        self.W_query = nn.Linear(d_in, d_out, bias=bias)
        self.W_key   = nn.Linear(d_in, d_out, bias=bias)
        self.W_value = nn.Linear(d_in, d_out, bias=bias)
        self.dropout = nn.Dropout(dropout)
        self.out_projection = nn.Linear(d_out, d_out)

        if causal:
            self.register_buffer("mask", torch.triu(torch.ones(context_length, context_length), diagonal=1))

    def forward(self, x):
        b, n_tokens, _ = x.shape

        keys = self.W_key(x)
        queries = self.W_query(x)
        values = self.W_value(x)

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

