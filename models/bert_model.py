import torch
import torch.nn as nn
import torch.nn.functional as F

import math
from dataclasses import dataclass

import os


@dataclass
class BertConfig:
    embed_dim: int = 768
    num_heads: int = 12
    dropout: float = 0.0
    ffn_num_inputs: int = 768
    ffn_hidden_dim: int = 768
    hidden_dim: int = 768
    norm_shape: int = 768
    kqv_size: int = 768
    num_blocks: int = 12
    bias: bool = False

    def __init__(self, vocab_size, context_length, **kwargs):
        self.vocab_size = vocab_size
        self.context_length = context_length

    @classmethod
    def from_pretrained(cls, pretrained_path):
        pass


class DotProductAttention(nn.Module):

    def __init__(self, config, **kwargs):
        super(DotProductAttention, self).__init__(**kwargs)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, keys, queries, values, attn_mask=None):
        # Shape of queries: (`batch_size`, `no. of queries`, d)
        # Shape of keys: (`batch_size`, `no. of key-value pairs`, d)
        # Shape of values: (`batch_size`, `no. of key-value pairs`, d)
        d = queries.shape[-1]
        scores = queries @ keys.transpose(1, 2) / math.sqrt(d)
        self.attn_weights = self.masked_softmax(scores, attn_mask)
        return self.dropout(self.attn_weights) @ values

    def masked_softmax(self, X, attn_mask):
        if attn_mask is None:
            return F.softmax(X, dim=-1)
        else:
            shape = X.shape
            if attn_mask.dim() == 1:
                attn_mask = torch.repeat_interleave(attn_mask, shape[1])

                X = self.sequence_mask(
                    X.reshape(-1), shape[-1], attn_mask, value=-1e6)
                return F.sfotmax(X.reshape(shape), dim=-1)

    def sequence_mask(self, X, valid_len, value=0):
        "Mask irrelevant entries (<pad> tokens) in sequence."
        maxlen = X.size(1)
        mask = torch.arange((maxlen), dtype=torch.float32, device=X.device)[
            None, :] < valid_len[:, None]
        X[~mask] = value
        return X


class MultiheadAttention(nn.Module):

    def __init__(self, config, **kwargs):
        super(MultiheadAttention, self).__init__(**kwargs)
        self.num_heads = config.num_heads
        self.attn_mechanism = DotProductAttention(config)
        self.k_W = nn.Linear(
            config.kqv_size, config.hidden_dim, bias=config.bias)
        self.q_W = nn.Linear(
            config.kqv_size, config.hidden_dim, bias=config.bias)
        self.v_W = nn.Linear(
            config.kqv_size, config.hidden_dim, bias=config.bias)
        self.o_W = nn.Linear(config.hidden_dim,
                             config.hidden_dim, bias=config.bias)

    def forward(self, keys, queries, values, attn_mask):
        keys = self.transpose_for_mltha(self.k_W(keys), self.num_heads)
        queries = self.transpose_for_mltha(self.q_W(queries), self.num_heads)
        values = self.transpose_for_mltha(self.v_W(values), self.num_heads)

        if attn_mask is not None:
            attn_mask = torch.repeat_interleave(
                attn_mask, repeats=self.num_heads, dim=0
            )

        output = self.attn_mechanism(keys, queries, values)

        output_cat = self.transpose_output(output, self.num_heads)
        return self.o_W(output_cat)

    def transpose_for_mltha(self, X, num_heads):
        # Reshape `X` (keys, queries, values) to allow
        # for parallel computation of multiple attention heads.
        X = X.reshape(X.shape[0], X.shape[1], num_heads, -1)
        X = X.permute(0, 2, 1, 3)
        return X.reshape(-1, X.shape[2], X.shape[3])

    def transpose_output(self, X, num_heads):
        # After reshaping to compute multiple attention heads in parallel,
        # revert to original shape.
        X = X.reshape(-1, num_heads, X.shape[1], X.shape[2])
        X = X.permute(0, 2, 1, 3)
        return X.reshape(X.shape[0], X.shape[1], -1)


class FFN(nn.Module):
    """Positionwise feed-forward network."""

    def __init__(self, config, **kwargs):
        super(FFN, self).__init__(**kwargs)
        self.positionwise_ffn = nn.Sequential(
            nn.Linear(config.ffn_num_inputs, config.ffn_hidden_dim),
            nn.ReLU(),
            nn.Linear(config.ffn_hidden_dim, config.hidden_dim)
        )

    def forward(self, X):
        return self.positionwise_ffn(X)


class AddNorm(nn.Module):
    """Residual Block followed by Layer Normalization."""

    def __init__(self, config, **kwargs):
        super(AddNorm, self).__init__(**kwargs)
        self.dropout = nn.Dropout(config.dropout)
        self.ln = nn.LayerNorm(config.norm_shape)

    def forward(self, X, Y):
        return self.ln(self.dropout(Y) + X)


class EncoderBlock(nn.Module):

    def __init__(self, config, **kwargs):
        super(EncoderBlock, self).__init__(**kwargs)
        self.attn_mechanism = MultiheadAttention(config)
        self.addnorm1 = AddNorm(config)
        self.ffn = FFN(config)
        self.addnorm2 = AddNorm(config)

    def forward(self, X, valid_len):
        Y = self.addnorm1(X, self.attn_mechanism(X, X, X, valid_len))
        return self.addnorm2(Y, self.ffn(Y))


class BERTEncoder(nn.Module):

    def __init__(self, config, **kwargs):
        super(BERTEncoder, self).__init__(**kwargs)
        self.token_embeddings = nn.Embedding(
            config.vocab_size, config.embed_dim)
        self.segment_embeddings = nn.Embedding(2, config.embed_dim)

        self.enc_blks = nn.Sequential()
        for i in range(config.num_blocks):
            self.enc_blks.add_module(str(i), EncoderBlock(config))

        self.pos_encodings = nn.Parameter(torch.randn(
            1, config.context_length, config.embed_dim
        ))

    def forward(self, tokens, segments, attn_mask):
        # Shape of X: (`batch_size`, `context_length`, `hidden_dim`)
        X = self.token_embeddings(tokens) + self.segment_embeddings(segments)
        X += self.pos_encodings.data[:, :X.shape[1], :]

        for blk in self.enc_blks:
            X = blk(X, attn_mask)

        return X


class Masked_LM(nn.Module):
    """Masked Language Model of the Bert Task"""

    def __init__(self, config, **kwargs):
        super(Masked_LM, self).__init__(**kwargs)
        self.mlp = nn.Sequential(
            nn.Linear(config.ffn_num_inputs, config.hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(config.hidden_dim),
            nn.Linear(config.hidden_dim, config.vocab_size)
        )

    def forward(self, X, pred_positions):
        num_pred_positions = pred_positions.shape[1]
        pred_positions = pred_positions.reshape(-1)
        batch_size = X.shape[0]
        batch_ndx = torch.arange(0, batch_size)
        batch_ndx = torch.repeat_interleave(batch_ndx, num_pred_positions)
        masked_X = X[batch_ndx, pred_positions]
        masked_X = masked_X.reshape((batch_size, num_pred_positions, -1))
        mlm_Y_hat = self.mlp(masked_X)
        return mlm_Y_hat


class NextSentencePrediction(nn.Module):
    """The next sentence prediction task for BERT"""

    def __init__(self, config, **kwargs):
        super(NextSentencePrediction, self).__init__(**kwargs)
        self.output = nn.Linear(config.ffn_num_inputs, 2)

    def forward(self, X):
        return self.output(X)


class BERTModel(nn.Module):

    def __init__(self, config, **kwargs):
        super(BERTModel, self).__init__(**kwargs)
        self.encoder = BERTEncoder(config)

        self.hidden = nn.Sequential(
            nn.Linear(config.hidden_dim, config.hidden_dim),
            nn.Tanh()
        )

        self.mlm = Masked_LM(config)
        self.nsp = NextSentencePrediction(config)

    def forward(self, tokens, segments, attn_mask=None, pred_positions=None):
        X_encoded = self.encoder(tokens, segments, attn_mask)

        if pred_positions is not None:
            mlm_Y_hat = self.mlm(X_encoded, pred_positions)
        else:
            mlm_Y_hat = None

        # The hidden layer of MLP classifier for next sentence prediction
        # <cls> token has index 0
        nsp_Y_hat = self.nsp(self.hidden(X_encoded[:, 0, :]))
        return X_encoded, mlm_Y_hat, nsp_Y_hat

    @classmethod
    def from_pretrained(cls, name, config):
        # Create instance of model
        model = cls(config)

        # Get checkpoint path and load pretrained model.
        checkpoint_path = os.path.join("./checkpoints", f"{name}.pth")
        model.load_state_dict(torch.load(checkpoint_path))
        
        return model
