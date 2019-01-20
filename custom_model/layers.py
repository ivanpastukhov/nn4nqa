from torch import nn
import torch
from torch.nn import functional as F
from torch.autograd import Variable
import logging
import math


# Источники: https://arxiv.org/pdf/1706.03762.pdf - Attention Is All You Need, §3.5
#            "How to code The Transformer in Pytorch" - Samuel Lynn-Evans
#            "The Annotated Transformer" - http://nlp.seas.harvard.edu/2018/04/03/attention.html
#            "The Illustrated Transformer" - Jay Alammar, http://jalammar.github.io/illustrated-transformer/

logging.basicConfig(level=logging.WARNING)


class Attention(nn.Module):
    def __init__(self, hidden_size):
        super(Attention, self).__init__()
        self.attn = nn.Linear(hidden_size, hidden_size)

    def score(self, hidden, output):
        energy = self.attn(output)
        # Permute: (batch_size, seq_len, hidden_size) -> (batch_size, hidden_size, seq_len) для перемножения тензоров
        energy = energy.permute(0, 2, 1)
        # Permute: (batch_size, 1, seq_len) -> (batch_size, seq_len, 1)
        score = torch.bmm(hidden, energy).permute(0,2,1)
        return score

    def forward(self, hidden, outputs):
        energies = self.score(hidden, outputs)
        energies = energies.t()
        return F.softmax(energies, dim=1).unsqueeze(1)


class SelfAttention():
    def __init__(self, dropout=None, positional_encoding=False):
        if dropout:
            # TODO: dropout
            raise NotImplementedError()
        if positional_encoding:
            # TODO: positional encoding
            raise NotImplementedError()
        pass

    @staticmethod
    def self_attention(query, key, value):
        d_k = value.size(-1)
        score = torch.bmm(query, key.permute(0,2,1))
        score = score / math.sqrt(d_k)
        # TODO: потенциально слабое место с направлением softmax'a.
        p_att = F.softmax(score, dim=-1)
        score = torch.bmm(p_att, value)
        return score, p_att


class MultiheadAttention(nn.Module):
    def __init__(self, n_heads, emb_size, att_size=64, dropout=None):
        super(MultiheadAttention, self).__init__()
        if dropout:
            # TODO: dropout
            raise NotImplementedError
        self.n_heads = n_heads
        self.emb_size = emb_size
        self.att_size = att_size
        self.attention = SelfAttention().self_attention
        # (W_q) n_heads times:
        self.linear_query = nn.ModuleList([nn.Linear(self.emb_size, self.att_size) for _ in range(self.n_heads)])
        # (W_k) n_heads times:
        self.linear_key = nn.ModuleList([nn.Linear(self.emb_size, self.att_size) for _ in range(self.n_heads)])
        # (W_v) n_heads times:
        self.linear_value = nn.ModuleList([nn.Linear(self.emb_size, self.att_size) for _ in range(self.n_heads)])
        # Fields for keeping attended values and attention_probabilities
        self.att_probas = []    # n_heads х n_sentences x max_len x max_len
        self.scores = []
        # Linear layer to transform concatenated heads
        self.output_linear = nn.Linear(n_heads*att_size, emb_size)

    def forward(self, query, key, value):
        # for each head:
        for head in range(self.n_heads):
            q = self.linear_query[head](query)
            k = self.linear_key[head](key)
            v = self.linear_value[head](value)
            # Scaled dot-product attention:
            score, p_att = self.attention(q,k,v)
            self.att_probas.append(p_att)
            self.scores.append(score)
        # Concatenate resulting matrices concat(z_0, z_1, ... z__n_heads)
        scores = torch.cat(self.scores, -1)
        # Transform concatenated
        scores = self.output_linear(scores)
        # Update attention probabilities for every head
        att_probas = self.att_probas
        # Reset scores and probabilities
        self.scores = []
        self.att_probas = []
        return scores, att_probas


class AttentionFlattener(nn.Module):
    def __init__(self, seq_len):
        super(AttentionFlattener, self).__init__()
        self.attention_matrix = None
        self.linear = nn.Linear(seq_len, 1)
        self.softmax = nn.Softmax(0)
        pass

    def forward(self, x):
        self.attention_matrix = x
        scores = self.linear(self.attention_matrix)
        scores = self.softmax(scores)
        return scores


class PositionalEncoding(nn.Module):
    # Нужен для работы self-Attention.
    # Positional Encoding Matrix имеет тот же размер, что и Input Sentence Matrix:
    # (seq_len x emb_len) = (seq_len x i_len)
    # Источники: https://arxiv.org/pdf/1706.03762.pdf - Attention Is All You Need, §3.5

    def __init__(self, hidden_size, dropout, max_len=5000):
        """
        Positional encoding class.
        :param hidden_size: emb_size in order to pos_encoding vector has same len as embedding to be summed together.
        :param dropout: dropout rate.
        :param max_len: sentence's size.

        Return:
        Input with positional encoding. Has same size as in input.
        """
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, hidden_size)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, hidden_size, 2) *
                             -(math.log(10000.0) / hidden_size))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + Variable(self.pe[:, :x.size(1)],
                         requires_grad=False)
        return self.dropout(x)