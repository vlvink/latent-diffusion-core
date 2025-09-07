import torch
import torch.nn as nn
import math
import torch.nn.functional as F


class SelfAttention(nn.Module):
    def __init__(self, emded_dim, n_heads, in_proj_bias=True, out_proj_bias=True):
        super(SelfAttention, self).__init__()
        self.n_heads = n_heads
        self.in_proj = nn.Linear(emded_dim, 3 * emded_dim, bias=in_proj_bias)
        self.out_proj = nn.Linear(emded_dim, emded_dim, bias=out_proj_bias)

        self.dim_heads = emded_dim // n_heads

    def forward(self, x, causal_mask):
        # x: [batch_size, seq_len, emb_dim]
        batch_size, seq_len, emb_dim = x.shape

        # [batch_size, seq_len, emb_dim] --> 3 * [batch_size, seq_len, emb_dim]
        Q, K, V = self.in_proj(x).chunk(3, dim=-1)

        # [batch_size, seq_len, emb_dim] --> [batch_size, seq_len, n_heads, dim_heads]
        Q = Q.view(batch_size, seq_len, self.n_heads, self.dim_heads)
        K = K.view(batch_size, seq_len, self.n_heads, self.dim_heads)
        V = V.view(batch_size, seq_len, self.n_heads, self.dim_heads)

        # [batch_size, seq_len, n_heads, dim_heads] --> [batch_size, n_heads, seq_len, dim_heads]
        Q = Q.transpose(1, 2)
        K = K.transpose(1, 2)
        V = V.transpose(1, 2)

        weight = Q @ K.transpose(-1, -2)

        if causal_mask:
            mask = torch.ones_like(weight, dtype=torch.bool).triu(diagonal=1)
            weight.masked_fill_(mask, -torch.inf)

        weight /= math.sqrt(self.dim_heads)
        weight = F.softmax(weight, dim=-1)

        output = weight @ V

        # [batch_size, n_heads, seq_len, dim_heads] --> [batch_size, seq_len, n_heads, dim_heads]
        output = output.transpose(1, 2)
        # [batch_size, seq_len, n_heads, dim_heads] --> [batch_size, seq_len, emb_dim]
        output = output.reshape((batch_size, seq_len, emb_dim))
        # [batch_size, seq_len, emb_dim] --> [batch_size, seq_len, emb_dim]
        output = self.out_proj(output)
        return output
