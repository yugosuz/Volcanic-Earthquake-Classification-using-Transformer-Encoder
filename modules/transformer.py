from __future__ import annotations

import numpy as np
import torch
from torch import nn


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000) -> None:
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(x + self.pe[:, : x.size(1)])


class MultiHeadAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim**-0.5
        self.to_qkv = nn.Linear(dim, dim * 3, bias=False)
        self.to_out = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None):
        b, n, _, h = *x.shape, self.num_heads
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: t.reshape(b, n, h, -1).transpose(1, 2), qkv)
        dots = (q @ k.transpose(-2, -1)) * self.scale

        if mask is not None:
            mask_value = -torch.finfo(dots.dtype).max
            dots = dots.masked_fill(~mask[:, None, None, :].to(x.device), mask_value)

        attn = self.dropout(dots.softmax(dim=-1))
        out = (attn @ v).transpose(1, 2).reshape(b, n, -1)
        return self.to_out(out), attn


class FeedForward(nn.Module):
    def __init__(self, dim: int, hidden_dim: int, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class EncoderBlock(nn.Module):
    def __init__(self, dim: int, num_heads: int, hidden_dim: int, dropout: float = 0.1):
        super().__init__()
        self.attn = MultiHeadAttention(dim, num_heads, dropout)
        self.ff = FeedForward(dim, hidden_dim, dropout)
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, src: torch.Tensor, mask: torch.Tensor | None = None):
        x, attn = self.attn(src, mask=mask)
        x = self.norm1(src + self.dropout1(x))
        x = self.norm2(x + self.dropout2(self.ff(x)))
        return x, attn


class TransformerEncoder(nn.Module):
    def __init__(self, dim: int, num_heads: int, hidden_dim: int, depth: int, dropout: float = 0.1):
        super().__init__()
        self.layers = nn.ModuleList(
            [EncoderBlock(dim, num_heads, hidden_dim, dropout) for _ in range(depth)]
        )

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None):
        attn_list = []
        for layer in self.layers:
            x, attn = layer(x, mask)
            attn_list.append(attn)
        return x, attn_list
