from torch import nn
import torch
from torch.nn import functional as F
from torch.autograd import Variable
import logging
logging.basicConfig(level=logging.DEBUG)
import sys
import time
import math


# Источники: https://arxiv.org/pdf/1706.03762.pdf - Attention Is All You Need, §3.5
#            "How to code The Transformer in Pytorch" - Samuel Lynn-Evans
#            "The Annotated Transformer" - http://nlp.seas.harvard.edu/2018/04/03/attention.html
#            "The Illustrated Transformer" - Jay Alammar, http://jalammar.github.io/illustrated-transformer/


class Attention(nn.Module):
    def __init__(self, hidden_size):
        super(Attention, self).__init__()
        self.attn = nn.Linear(hidden_size, hidden_size)

    def score(self, hidden, output):
        logging.debug('hidden.size(): {}'.format(hidden.size()))
        logging.debug('output.size(): {}'.format(output.size()))
        energy = self.attn(output)
        score = torch.sum(hidden * energy, dim=2)
        return score

    def forward(self, hidden, outputs):
        energies = self.score(hidden, outputs)
        energies = energies.t()
        return F.softmax(energies, dim=1).unsqueeze(1)


# class SelfAttention(nn.Module):



class PositionalEncoding(nn.Module):
    # Нужен для работы self-Attention.
    # Positional Encoding Matrix имеет тот же размер, что и Input Sentence Matrix:
    # (seq_len x emb_len) = (seq_len x i_len)
    # Источники: https://arxiv.org/pdf/1706.03762.pdf - Attention Is All You Need, §3.5

    def __init__(self, d_model, dropout, max_len=5000):
        """
        Positional encoding class.
        :param d_model: emb_size in order to pos_encoding vector has same len as embedding to be summed together.
        :param dropout: dropout rate.
        :param max_len: sentence's size.

        Return:
        Input with positional encoding. Has same size as in input.
        """
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) *
                             -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + Variable(self.pe[:, :x.size(1)],
                         requires_grad=False)
        return self.dropout(x)


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


class BaseModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.losses = []
        self.steps = []

    def fit(self, X_left, X_right, y_train, batch_size, epochs, loss_function, optimizer, device):
        """
        Fit the model.
        :param X_left: pytorch.Tensor object, sequences of encoded phrases. Usually questions.
        :param X_right: pytorch.Tensor object, sequences of encoded phrases. Usually answers.
        :param y_train: pytorch.Tensor object, array of target labels
        :param batch_size: int, size of batch.
        :param epochs: int, number of epochs.
        :param loss_function: function, scalar must be returned.
        :param optimizer: torch optimizer object
        :param device: str, 'cuda' or 'cpu'
        :return:
        """
        self._fit(X_left, X_right, y_train, batch_size, epochs, loss_function, optimizer, device)  # custom logic

    def _fit(self, X_left, X_right, y_train, batch_size, epochs, loss_function, optimizer, device):
        assert len(X_left) == len(y_train) == len(X_right)
        len_dataset = len(X_left)
        step = 0
        optimizer = optimizer(self.parameters())
        self.to(device)
        print('Training...')
        for epoch in range(epochs):
            start_time = time.time()
            lb = 0
            rb = batch_size
            while lb < len_dataset:
                x_l_batch = X_left[lb:rb].to(device)
                x_r_batch = X_right[lb:rb].to(device)
                y_train_batch = y_train[lb:rb].to(device)
                y_pred_batch = self.__call__(x_l_batch, x_r_batch)
                loss = loss_function(y_pred_batch, y_train_batch).to(device)
                self.losses.append(loss.item())
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                # update counters
                rb += batch_size
                lb += batch_size
                step += 1
                self.steps.append(step)
                # progress bar
                sys.stdout.write("")
                sys.stdout.flush()
            end_time = time.time()
            print('Epoch: {}, loss: {:0.5f}. {:0.2} [s] per epoch'.format(epoch, loss, end_time-start_time))
        print('Done!')


class SimpleNet(BaseModel):
    def __init__(self, vocab_size, embed_dim, hidden_size):
        super().__init__()
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
        ans = F.softmax(self.answer(fc), dim=1)
        return ans


