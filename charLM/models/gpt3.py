import torch
import torch.nn as nn
import torch.nn.functional as F
from models.multihead_attention import MultiHead_Masked_SelfAttention

"""
1) x: torch.Size([128, 8])
2) src: torch.Size([128, 7]) and tgt: torch.Size([128, 7])
3) word_embed: torch.Size([128, 7, 64])
4) position_embeddings: torch.Size([1, 7, 64])
5) embedding_dropout: torch.Size([128, 7, 64])
6) decoders: torch.Size([128, 7, 64])
7) layer_norm: torch.Size([128, 7, 64])
8) logits: torch.Size([128, 7, 39])
9) _tgt: torch.Size([896])
10) _output_logits: torch.Size([896, 39])
11) loss: torch.Size([896])


1) x: torch.Size([128, 12])
2) src: torch.Size([128, 11]) and tgt: torch.Size([128, 11])
3) word_embed: torch.Size([128, 11, 64])
4) position_embeddings: torch.Size([1, 11, 64])
5) embedding_dropout: torch.Size([128, 11, 64])
6) decoders: torch.Size([128, 11, 64])
7) layer_norm: torch.Size([128, 11, 64])
8) logits: torch.Size([128, 11, 39])
9) _tgt: torch.Size([1408])
10) _output_logits: torch.Size([1408, 39])
11) loss: torch.Size([1408])
"""

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
        x = self.layer_norm2(x)
        return x


class GPT3(nn.Module):
    def __init__(self, vocab, num_layers, embed_dim, num_heads=8, block_size=128, embedding_dropout_rate=0.1, attention_dropout_rate=0.1, residual_dropout_rate=0.1, expand_ratio=4):
        super(GPT3, self).__init__()
        self.vocab = vocab
        self.token_embedding = nn.Embedding(num_embeddings=len(vocab.word2id), embedding_dim=embed_dim)
        self.position_embedding = nn.Parameter(torch.zeros(1, block_size, embed_dim))
        self.embedding_dropout = nn.Dropout(p=embedding_dropout_rate)
        self.decoders = nn.Sequential(*[Decoder(embed_dim=embed_dim, num_heads=num_heads, block_size=block_size, attention_dropout_rate=attention_dropout_rate, residual_dropout_rate=residual_dropout_rate, expand_ratio=expand_ratio) for layer in range(num_layers)])
        self.layer_norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, len(vocab.word2id), bias=False)
        vocab_mask = torch.ones(len(vocab.word2id))
        self.loss = nn.CrossEntropyLoss(weight=vocab_mask, reduce=False, ignore_index=0)

    def forward(self, x):

        print(f"1) x: {x.shape}")
        src = x[:, :-1]  # remove end symbol
        tgt = x[:, 1:]  # remove start symbol
        b, t = src.size()
        print(f"2) src: {src.shape} and tgt: {tgt.shape}")

        # forward the GPT model

        token_embeddings = self.token_embedding(src)  # each index maps to a (learnable) vector
        print(f"3) word_embed: {token_embeddings.shape}")
        position_embeddings = self.position_embedding[:, :t, :]  # each position maps to a (learnable) vector
        print(f"4) position_embeddings: {position_embeddings.shape}")
        x = self.embedding_dropout(token_embeddings + position_embeddings)
        print(f"5) embedding_dropout: {x.shape}")
        x = self.decoders(x)
        print(f"6) decoders: {x.shape}")
        x = self.layer_norm(x)
        print(f"7) layer_norm: {x.shape}")
        logits = self.head(x)
        print(f"8) logits: {logits.shape}")

        _tgt = tgt.contiguous().view(-1)
        print(f"9) _tgt: {_tgt.shape}")
        _output_logits = logits.view(-1, logits.size(2))
        print(f"10) _output_logits: {_output_logits.shape}")
        # if we are given some desired targets also calculate the loss
        #loss = F.cross_entropy(logits.view(-1, logits.size(-1)), tgt.view(-1))
        loss = self.loss(_output_logits, _tgt)
        print(f"11) loss: {loss.shape}")

        return loss, self.accuracy(logits, tgt), logits



    def accuracy(self, output_logits, targets):
        # output_logits: (B, T, vocab_size), targets: (B,T)
        surface_vocab = self.vocab
        B, T = targets.size()
        sft = nn.Softmax(dim=2)
        # (batchsize, T)
        pred_tokens = torch.argmax(sft(output_logits),2)
        correct_tokens = (pred_tokens == targets)
        wrong_tokens = (pred_tokens != targets)
        wrong_predictions = []
        correct_predictions = []
        for i in range(B):
            target  = ''.join(surface_vocab.decode_sentence(targets[i]))
            pred = ''.join(surface_vocab.decode_sentence(pred_tokens[i]))
            if target != pred:
                wrong_predictions.append('target: %s pred: %s' % (target, pred))
            else:
                correct_predictions.append('target: %s pred: %s' % (target, pred))
        acc = correct_tokens.sum().item(), B*T, wrong_tokens.sum().item(), wrong_predictions, correct_predictions
        return  acc