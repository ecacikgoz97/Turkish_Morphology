import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class MultiHead_Masked_SelfAttention(nn.Module):
    """
    A vanilla multi-head masked self-attention layer with a projection at the end.
    It is possible to use torch.nn.MultiheadAttention here but I am including an
    explicit implementation here to show that there is nothing too scary here.
    """

    def __init__(self, embed_dim=512, num_heads=8, block_size=128, attention_dropout_rate=0.1, residual_dropout_rate=0.1):
        super().__init__()
        # key, query, value projections for all heads
        self.key   = nn.Linear(embed_dim, embed_dim)
        self.query = nn.Linear(embed_dim, embed_dim)
        self.value = nn.Linear(embed_dim, embed_dim)
        # regularization
        self.attention_dropout  = nn.Dropout(attention_dropout_rate)
        self.residual_dropout = nn.Dropout(residual_dropout_rate)
        # output projection
        self.fc = nn.Linear(embed_dim, embed_dim)
        # causal mask to ensure that attention is only applied to the left in the input sequence
        self.register_buffer("mask", torch.tril(torch.ones(block_size, block_size)).view(1, 1, block_size, block_size))
        self.num_heads = num_heads

    def forward(self, x, layer_past=None):
        B, T, C = x.size()

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        k = self.key(x).view(B, T, self.num_heads, C // self.num_heads).transpose(1, 2) # (B, nh, T, hs)
        q = self.query(x).view(B, T, self.num_heads, C // self.num_heads).transpose(1, 2) # (B, nh, T, hs)
        v = self.value(x).view(B, T, self.num_heads, C // self.num_heads).transpose(1, 2) # (B, nh, T, hs)

        # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        #att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        attention = torch.mm(q, k.T) * (1.0 / math.sqrt(k.size(-1)))
        attention = attention.masked_fill(self.mask[:,:,:T,:T] == 0, float('-inf'))
        normalized_attention = F.softmax(attention, dim=-1)
        attention = self.attention_dropout(normalized_attention)
        #y = att @ v # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = torch.mm(attention, v)
        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side

        # output projection
        y = self.fc(y)
        y = self.residual_dropout(y)
        return y