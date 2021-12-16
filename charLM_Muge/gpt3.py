import torch
import torch.nn as nn
import torch.nn.functional as F
from multihead_attention import MultiHead_Masked_SelfAttention

class Decoder(nn.Module):
    def __init__(self, embed_dim=512, num_heads=8, block_size=128, attention_dropout_rate=0.1, residual_dropout_rate=0.1, expand_ratio=4):
        super(Decoder, self).__init__()
        self.MH_attention = MultiHead_Masked_SelfAttention(embed_dim=embed_dim, num_heads=num_heads, block_size=block_size, attention_dropout_rate=attention_dropout_rate, residual_dropout_rate=residual_dropout_rate)
        self.layer_norm1 = nn.LayerNorm(embed_dim)
        self.feed_forward = nn.Sequential(
            nn.Linear(embed_dim, embed_dim*expand_ratio),
            nn.GELU(),
            nn.Linear(embed_dim*expand_ratio, embed_dim),
            nn.Dropout(p=residual_dropout_rate)
        )
        self.layer_norm2 = nn.LayerNorm(embed_dim)

    def forward(self, x):
        res1 = x
        x = self.MH_attention(x)
        x = x + res1
        x = self.layer_norm1(x)

        res2 = x
        x = self.feed_forward(x)
        x = x + res2
        x = self.layer_norm2
        return x


class GPT3(nn.Module):
    def __init__(self, vocab_size, num_layers, embed_dim, num_heads=8, block_size=128, embedding_dropout_rate=0.1, attention_dropout_rate=0.1, residual_dropout_rate=0.1, expand_ratio=4):
        super(GPT3, self).__init__()
        self.token_embedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=embed_dim)
        self.position_embedding = nn.Parameter(torch.zeros(1, block_size, embed_dim))
        self.embedding_dropout = nn.Dropout(p=embedding_dropout_rate)
        self.decoders = nn.Sequential(*[Decoder(embed_dim=512, num_heads=8, block_size=128, attention_dropout_rate=0.1, residual_dropout_rate=0.1, expand_ratio=4) for layer in range(num_layers)])
        self.layer_norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, vocab_size, bias=False)

    def forward(self, index, targets=None):
        b, t = index.size()

        # forward the GPT model
        token_embeddings = self.token_embedding(index)  # each index maps to a (learnable) vector
        position_embeddings = self.position_embedding[:, :t, :]  # each position maps to a (learnable) vector
        x = self.embedding_dropout(token_embeddings + position_embeddings)
        x = self.decoders(x)
        x = self.layer_norm(x)
        logits = self.head(x)

        # if we are given some desired targets also calculate the loss
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))

        return logits, loss
