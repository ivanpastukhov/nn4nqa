from torch import nn
import torch
from torch.nn import functional as F
from torch.autograd import Variable
import logging
logging.basicConfig(level=logging.WARNING)
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

    def validation_loss(self, X_l_val, X_r_val, y_val):
        y_pred = self.__call__(X_l_val, X_r_val)
        loss = self.loss_function(y_pred, y_val)
        return loss

    def fit(self, X_left, X_right, y_train, batch_size, epochs, loss_function, optimizer, device,
            val_data=None):
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
        :param val_data: validation data like (X_val, y_val) is used to obtain vlidation results during training process.
        :return:
        """
        self._fit(X_left, X_right, y_train, batch_size, epochs, loss_function, optimizer, device, val_data)  # custom logic

    def _fit(self, X_left, X_right, y_train, batch_size, epochs, loss_function, optimizer, device, val_data):
        self.loss_function = loss_function
        self.optimizer = optimizer
        self.device = device
        if val_data:
            x_val, y_val = val_data
            x_l_val, x_r_val = x_val
            x_l_val = x_l_val.to(device)
            x_r_val = x_r_val.to(device)
            y_val = y_val.to(device)
            self.validation = True
        else:
            self.validation = False
        assert len(X_left) == len(y_train) == len(X_right)
        len_dataset = len(X_left)
        step = 0
        # Initialize optimizer
        self.optimizer = self.optimizer(self.parameters())
        self.to(self.device)
        print('Training...')
        for epoch in range(epochs):
            start_time = time.time()
            lb = 0
            rb = batch_size
            while lb < len_dataset:
                x_l_batch = X_left[lb:rb].to(self.device)
                x_r_batch = X_right[lb:rb].to(self.device)
                y_train_batch = y_train[lb:rb].to(self.device)
                y_pred_batch = self.__call__(x_l_batch, x_r_batch)
                loss = self.loss_function(y_pred_batch, y_train_batch).to(self.device)
                self.losses.append(loss.item())
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                # update counters
                rb += batch_size
                lb += batch_size
                step += 1
                self.steps.append(step)
                # progress bar
                sys.stdout.write("")
                sys.stdout.flush()
            end_time = time.time()
            torch.cuda.empty_cache()
            print('Epoch: {}, loss: {:0.5f}. {:0.2} [s] per epoch'.format(epoch, loss, end_time-start_time))
            if self.validation:
                val_loss = self.validation_loss(x_l_val, x_r_val, y_val)
                print('       val_loss: {:0.5f}'.format(val_loss))
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

class SAttendedSimpleNet(SimpleNet):
    def __init__(self, vocab_size, embed_dim, rnn_hidden_size,
                 attention_size, n_heads):
        super(SAttendedSimpleNet, self).__init__(vocab_size, embed_dim,
                                                 rnn_hidden_size)
        self.l_attention = MultiheadAttention(n_heads, rnn_hidden_size,
                                              att_size=attention_size)
        self.l_probas = None
    def forward(self, input_seq_l, input_seq_r):
        outputs_l = self.encoder_l(input_seq_l)
        outputs_l, self.l_probas = self.l_attention(outputs_l, outputs_l, outputs_l)
        outputs_l = torch.mean(outputs_l, 1)
        outputs_r = self.encoder_r(input_seq_r)
        outputs_r = torch.mean(outputs_r, 1)
        concatenated = torch.cat((outputs_l, outputs_r), 1)
        fc = self.hidden(concatenated)
        ans = F.softmax(self.answer(fc), dim=1)
        return ans


