import torch.nn as nn, torch


class SelfAttention(nn.Module):
    def __init__(self, d_in, d_out, bias=False, device = "cuda" if torch.cuda.is_available() else "cpu"):
        super().__init__()

        self.W_query = nn.Linear(d_in, d_out, bias=bias).to(device)
        self.W_key   = nn.Linear(d_in, d_out, bias=bias).to(device)
        self.W_value = nn.Linear(d_in, d_out, bias=bias).to(device)
    
    def forward(self, x):
        keys = self.W_key(x)
        queries = self.W_query(x)
        values = self.W_value(x)

        attention_scores = queries @ keys.T

        attention_weights = torch.softmax(attention_scores / keys.shape[-1]**0.5, dim=-1)

        context_vector = attention_weights @ values

        return context_vector
    

class CausalAttention(nn.Module):
    def __init__(self, d_in, d_out, context_length, dropout, bias=False, device = "cuda" if torch.cuda.is_available() else "cpu"):
        super().__init__()

        self.W_query = nn.Linear(d_in, d_out, bias=bias).to(device)
        self.W_key   = nn.Linear(d_in, d_out, bias=bias).to(device)
        self.W_value = nn.Linear(d_in, d_out, bias=bias).to(device)
        self.mask = torch.tril(torch.ones(context_length, context_length)).to(device)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        _, n, _ = x.shape

        keys = self.W_key(x)
        queries = self.W_query(x)
        values = self.W_value(x)

        attention_scores = queries @ keys.T

        masked_attention_scores = attention_scores.masked_fill(self.mask.bool()[:n, :n], -torch.inf) 

        attention_weights = torch.softmax(masked_attention_scores / keys.shape[-1]**0.5, dim=-1)

        attention_weights = self.dropout(attention_weights)

        context_vector = attention_weights @ values

        return context_vector
    

class InefficientMultiHeadAttention(nn.Module):
    def __init__(self, d_in, d_out, context_length, dropout, num_heads, bias, device = "cuda" if torch.cuda.is_available() else "cpu"):
        super().__init__()
        self.heads = nn.ModuleList(
            [CausalAttention(d_in, d_out, context_length, dropout, bias, device=device) for _ in range(num_heads)]
        )
        
    def forward(self, x):
        return torch.cat([head(x) for head in self.heads], dim=-1)