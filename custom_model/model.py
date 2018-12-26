from torch import nn
import torch
from torch.nn import functional as F
import logging
logging.basicConfig(level=logging.DEBUG)

class Encoder(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_size, bidirectional=False, pretrained_emb=False,
                 dropout=False):
        super(Encoder, self).__init__()
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.hidden_size = hidden_size
        self.dropout = dropout
        self.embedding = nn.Embedding(num_embeddings=self.vocab_size,
                                      embedding_dim=self.embed_dim)
        # TODO: добавить другие RNN
        self.rnn = nn.LSTM(input_size=self.embed_dim,
                               hidden_size=self.hidden_size)

        if bidirectional:
            # TODO: ...
            raise NotImplementedError()
        if pretrained_emb:
            # TODO: ...
            raise  NotImplementedError()
        if dropout:
            # TODO: ...
            raise  NotImplementedError()

    def forward(self, input_seq, hidden=None):
        # TODO: torch.nn.utils.rnn.pack_padded_sequence
        embedded = self.embedding(input_seq)
        outputs, _ = self.rnn(embedded, hidden)
        return outputs

class SimpleNet(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_size):
        super(SimpleNet, self).__init__()
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.hidden_size = hidden_size
        self.encoder_l = Encoder(self.vocab_size,
                                 self.embed_dim,
                                 self.hidden_size)
        self.encoder_r = Encoder(self.vocab_size,
                                 self.embed_dim,
                                 self.hidden_size)
        self.hidden = nn.Linear(hidden_size*2, 64)
        self.answer = nn.Linear(64, 2)

    def forward(self, input_seq_l, input_seq_r):
        outputs_l = self.encoder_l(input_seq_l)
        outputs_l = torch.mean(outputs_l, 1)
        outputs_r = self.encoder_r(input_seq_r)
        outputs_r = torch.mean(outputs_r, 1)
        concatenated = torch.cat((outputs_l, outputs_r), 1)
        fc = self.hidden(concatenated)
        ans = F.softmax(self.answer(fc))
        return ans
