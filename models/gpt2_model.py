import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from dataclasses import dataclass

import os


@dataclass
class GPTConfig:
    embed_dim: int = 768
    num_heads: int = 12
    num_blocks: int = 12
    dropout: float = 0.0
    bias: bool = False

    def __init__(self, vocab_size, context_length, **kwargs):
        self.vocab_size = vocab_size
        self.context_length = context_length
        for key, value in kwargs.items():
            setattr(self, key, value)


class CausalSelfAttention(nn.Module):

    def __init__(self, config, **kwargs):
        super(CausalSelfAttention, self).__init__(**kwargs)
        embed_dim = config.embed_dim
        self.num_heads = config.num_heads
        assert embed_dim % self.num_heads == 0, "Invalid values for embed_dim/num_heads (embed_dim % num_heads != 0)."
        self.k_W = nn.Linear(embed_dim, embed_dim)
        self.q_W = nn.Linear(embed_dim, embed_dim)
        self.v_W = nn.Linear(embed_dim, embed_dim)
        self.o_W = nn.Linear(embed_dim, embed_dim)
        self.attn_dropout = nn.Dropout(config.dropout)
        self.out_dropout = nn.Dropout(config.dropout)

        self.register_buffer(
            "mask",
            torch.tril(torch.ones(config.context_length,
                                  config.context_length)).unsqueeze(0).unsqueeze(0)
        )

    def forward(self, X):
        batch_size = X.size(0)
        seq_length = X.size(1)

        # Pass input through key, value and query transforms and transpose
        # for computation of multiple attention heads
        k = self.k_W(X).reshape(batch_size, seq_length, self.num_heads, -1).permute(0, 2, 3, 1).transpose(1, 2)
        q = self.q_W(X).reshape(batch_size, seq_length, self.num_heads, -1).transpose(1, 2)
        v = self.v_W(X).reshape(batch_size, seq_length, self.num_heads, -1).transpose(1, 2)

        d = q.size(-1)
        attn = (q @ k.transpose(-2, -1)) / math.sqrt(d)
        mask = self.mask[:, :, :seq_length, :seq_length]
        attn = attn.masked_fill(mask == 0, float("-inf"))
        attn = self.attn_dropout(attn)
        attn = F.softmax(attn, dim=-1)

        # Multipy by v to get output and transpose into original shape
        out = (attn @ v)
        out = out.transpose(1, 2)
        out = out.reshape(batch_size,  seq_length, -1)
        out = self.out_dropout(self.o_W(out))
        return out


class FeedForwardNet(nn.Module):

    def __init__(self, config, **kwargs):
        super(FeedForwardNet, self).__init__(**kwargs)
        embed_dim = config.embed_dim
        self.dense1 = nn.Linear(embed_dim, embed_dim * 4)
        self.gelu = nn.GELU()
        self.dense2 = nn.Linear(embed_dim * 4, embed_dim)
        self.dropout = config.dropout

    def forward(self, X):
        X = self.dense2(self.gelu(self.dense1(X)))
        return X


class AddNorm(nn.Module):

    def __init__(self, config, **kwargs):
        super(AddNorm, self).__init__()
        norm_shape = config.embed_dim   # norm_shape = config.embed_dim
        self.ln = nn.LayerNorm(norm_shape)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, X, Y):
        X = self.ln(Y + X)
        return X


class GPTBlock(nn.Module):

    def __init__(self, config, **kwargs):
        super(GPTBlock, self).__init__(**kwargs)
        self.ln1 = nn.LayerNorm(config.embed_dim)   # norm_shape = embed_dim
        self.ln2 = nn.LayerNorm(config.embed_dim)   # norm_shape = embed_dim
        self.cs_attn = CausalSelfAttention(config)
        self.ffn = FeedForwardNet(config)

    def forward(self, X):
        X = X + self.cs_attn(self.ln1(X))
        X = X + self.ffn(self.ln2(X))
        return X


class GPTModel(nn.Module):

    def __init__(self, config, **kwargs):
        super(GPTModel, self).__init__(**kwargs)
        embed_dim = config.embed_dim
        self.context_length = config.context_length
        self.token_embedding = nn.Embedding(
            config.vocab_size, embed_dim
        )
        self.pos_encoding = nn.Parameter(
            torch.zeros(1, config.context_length, embed_dim)
        )
        self.dropout = nn.Dropout(config.dropout)
        self.blks = nn.Sequential()
        for i in range(config.num_blocks):
            self.blks.add_module(str(i), GPTBlock(config))
        self.ln = nn.LayerNorm(embed_dim)   # norm_shape = embed dim
        self.lm_head = nn.Linear(embed_dim, config.vocab_size)

    def forward(self, X, targets=None):
        # Forward pass on input `X`. Also Compute loss if there are targets to compare to
        context_length = X.size(1)
        assert context_length <= self.context_length, "Input sequence exceeds maximum allowed sequence length."

        token_embedding = self.token_embedding(X)
        pos_encoding = self.pos_encoding[:, :context_length:, :]
        X = self.dropout(token_embedding + pos_encoding)
        for blk in self.blks:
            X = blk(X)
        X = self.ln(X)

        if targets is not None:
            logits = self.lm_head(X)
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
        else:
            logits = self.lm_head(X[:, [-1], :])
            loss = None
        return logits, loss

    @torch.no_grad()
    def generate(self, input_seq, max_gen_tokens, temperature=1.0, top_k=None):

        for _ in range(max_gen_tokens):
            
            cond_in_seq = input_seq if input_seq.size(
                1) <= self.context_length else input_seq[:, -self.context_length:]
            logits, loss = self(cond_in_seq)
            # Obtain logits at final time step and scale by temperature (default = 1.0, no scale).
            logits = logits[:, -1, :] / temperature
            if top_k is not None:
                # Fetch only the top_k logits
                pass
            # Obtain probabilities of tokens by applying softmax
            probs = F.softmax(logits, dim=-1)
            nxt_token_ndx = torch.multinomial(probs, num_samples=1)

            # Append generated token to original sequence
            input_seq = torch.cat((input_seq, nxt_token_ndx), dim=1)
        return input_seq

    @classmethod
    def from_pretrained(cls, name, config):
        # Create instance of model
        model = cls(config)

        # Get checkpoint path and load pretrained model.
        checkpoint_path = os.path.join("./checkpoints", f"{name}.pth")
        model.load_state_dict(torch.load(checkpoint_path))
        
        return model
