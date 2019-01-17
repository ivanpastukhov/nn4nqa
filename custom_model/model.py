from torch import nn
import torch
from torch.nn import functional as F
from torch.autograd import Variable
import logging
logging.basicConfig(level=logging.WARNING)
import sys
import time
import math
import numpy as np

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


class Encoder(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_size, bidirectional=False, dropout=False,
                 emb_weights=None):
        super(Encoder, self).__init__()
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.hidden_size = hidden_size
        self.dropout = dropout
        self.embedding = nn.Embedding(num_embeddings=self.vocab_size,
                                      embedding_dim=self.embed_dim, padding_idx=0)
        # TODO: добавить другие RNN
        self.rnn = nn.LSTM(input_size=self.embed_dim,
                               hidden_size=self.hidden_size,
                           batch_first=True)

        if bidirectional:
            # TODO: ...
            raise NotImplementedError()
        if dropout:
            # TODO: ...
            raise  NotImplementedError()
        if not isinstance(emb_weights, type(None)):

            if emb_weights.shape != (self.vocab_size, self.embed_dim):
                raise ValueError('Size of embedding matrix must be equal to ones used in initialization.')
            emb_weights = torch.from_numpy(emb_weights).float()
            self.embedding = nn.Embedding.from_pretrained(emb_weights, freeze=True)

    def freeze_embeddings(self):
        self.embedding.freeze = True

    def unfreeze_embeddings(self):
        self.embedding.freeze = False

    def forward(self, input_seq, hidden=None):
        # TODO: torch.nn.utils.rnn.pack_padded_sequence
        embedded = self.embedding(input_seq)
        outputs, _ = self.rnn(embedded, hidden)
        return outputs


class BaseModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.losses = []
        self.val_losses = []
        self.steps = []

    def forward(self, *input):
        raise NotImplementedError('Class must be implemented in child class.')

    def validation_loss(self, X_l_val, X_r_val, y_val):
        y_pred = self.__call__(X_l_val, X_r_val)
        loss = self.loss_function(y_pred, y_val)
        return loss

    def fit(self, X_left, X_right, y_train, batch_size, epochs, loss_function, optimizer, device,
            clip=None, val_data=None):
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
        self._fit(X_left, X_right, y_train, batch_size, epochs, loss_function, optimizer, device, clip, val_data)  # custom logic

    def _fit(self, X_left, X_right, y_train, batch_size, epochs, loss_function, optimizer, device, clip,  val_data):
        self.loss_function = loss_function
        self.optimizer = optimizer
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
        # TODO: проследить, чтобы в optimizer'e обновлялись параметры при разморозке эмбеддингов
        self.optimizer = self.optimizer(filter(lambda x: x.requires_grad, self.parameters()))
        if clip:
            _ = nn.utils.clip_grad_norm(self.parameters(), clip)
        self.to(device)
        print('Training...')
        # self.optimizer.zero_grad()
        for epoch in range(epochs):
            start_time = time.time()
            lb = 0
            rb = batch_size
            epoch_losses = []
            while lb < len_dataset:
                #TODO: Использовать torch data.DataLoader вместо этого
                x_l_batch = X_left[lb:rb].to(device)
                x_r_batch = X_right[lb:rb].to(device)
                y_train_batch = y_train[lb:rb].to(device)
                y_pred_batch = self.__call__(x_l_batch, x_r_batch)
                batch_loss = self.loss_function(y_pred_batch, y_train_batch).to(device)
                epoch_losses.append(batch_loss.item())
                batch_loss.backward()
                self.optimizer.step()
                self.optimizer.zero_grad()
                # update counters
                rb += batch_size
                lb += batch_size
                step += 1
                self.steps.append(step)
                # progress bar
                sys.stdout.write("")
                sys.stdout.flush()
            epoch_loss = np.mean(epoch_losses)
            self.losses.append(epoch_loss)
            end_time = time.time()
            if self.validation:
                with torch.no_grad():
                    val_loss = self.validation_loss(x_l_val, x_r_val, y_val)
                    self.val_losses.append(val_loss.item())
                    print('Epoch: {}, loss: {:0.5f}. {:0.2} [s] per epoch. Val loss: {:0.5f}'.format(epoch, epoch_loss, end_time - start_time, val_loss))
            else:
                print('Epoch: {}, loss: {:0.5f}. {:0.2} [s] per epoch'.format(epoch, epoch_loss, end_time - start_time))

        print('Done!')


class SimpleNet(BaseModel):
    '''Самая простая сетка. Предложения кодируются в один вектор и сравниваются.'''
    def __init__(self, vocab_size, embed_dim, hidden_size, emb_weights=None):
        super().__init__()
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.hidden_size = hidden_size
        self.encoder_l = Encoder(self.vocab_size,
                                 self.embed_dim,
                                 self.hidden_size,
                                 emb_weights=emb_weights)
        self.encoder_r = Encoder(self.vocab_size,
                                 self.embed_dim,
                                 self.hidden_size,
                                 emb_weights=emb_weights)
        self.hidden = nn.Linear(hidden_size*2, 64)
        self.answer = nn.Linear(64, 2)

    def forward(self, input_seq_l, input_seq_r):
        outputs_l = self.encoder_l(input_seq_l)
        logging.debug('Outputs_size: {}'.format(outputs_l.size()))
        outputs_l = outputs_l[:, -1, :]
        outputs_r = self.encoder_r(input_seq_r)
        outputs_r = outputs_r[:, -1, :]
        concatenated = torch.cat((outputs_l, outputs_r), 1)
        fc = self.hidden(concatenated)
        ans = F.softmax(self.answer(fc), dim=1)
        return ans


class SAttendedSimpleNet(SimpleNet):
    '''Вторая версия сетки: вопрос проходит через Multihead attention, полученные вектора усредняются в один вектор.'''
    def __init__(self, vocab_size, embed_dim, rnn_hidden_size,
                 attention_size, n_heads, emb_weights=None):
        super(SAttendedSimpleNet, self).__init__(vocab_size, embed_dim,
                                                 rnn_hidden_size, emb_weights=emb_weights)
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


class SAttendedNet(SimpleNet):
    '''
    Третий вариант сетки. Вопрос и ответ проходят через Multihead, из которого достаётся матрица скоров (Attention
    probabilities), затем эта матрица сжимается в вектор скоров с помощью AttentionFlattener. Исходные вектора
    перемножаются на скоры и суммируются в один вектор.
    '''
    def __init__(self, vocab_size, embed_dim, rnn_hidden_size,
                 attention_size, n_heads, l_seq_len, r_seq_len, emb_weights=None):
        super(SAttendedNet, self).__init__(vocab_size, embed_dim,
                                                 rnn_hidden_size, emb_weights=emb_weights)
        self.l_attention = MultiheadAttention(n_heads, rnn_hidden_size,
                                              att_size=attention_size)
        self.r_attention = MultiheadAttention(n_heads, rnn_hidden_size,
                                              att_size=attention_size)
        self.l_flatten = AttentionFlattener(l_seq_len)
        self.r_flatten = AttentionFlattener(r_seq_len)
        self.l_probas = None
        self.r_probas = None
        self.l_scores = None
        self.r_scores = None


    def forward(self, input_seq_l, input_seq_r):
        outputs_l = self.encoder_l(input_seq_l)
        _, self.l_probas = self.l_attention(outputs_l, outputs_l, outputs_l)
        self.l_probas = self.l_probas[0]
        self.l_scores = self.l_flatten(self.l_probas)
        outputs_l = self.l_scores * outputs_l
        left = torch.sum(outputs_l, dim=1)

        outputs_r = self.encoder_r(input_seq_r)
        _, self.r_probas = self.r_attention(outputs_r, outputs_r, outputs_r)
        self.r_probas = self.r_probas[0]
        self.r_scores = self.r_flatten(self.r_probas)
        outputs_r = self.r_scores * outputs_r
        right = torch.sum(outputs_r, dim=1)

        concatenated = torch.cat((left, right), 1)
        logging.debug('Concatenated: {}'.format(concatenated.size()))
        fc = self.hidden(concatenated)
        ans = F.softmax(self.answer(fc), dim=1)
        return ans


class CrossAttentionNet(SimpleNet):
    '''
    Четвёртый вариант сетки. В Multihead'ы подаются параметры из обеих веток сетки, поэтому получаем 2 матрицы:
    как токены вопросов скореллированы с токенами ответов и наоборот.
    '''
    def __init__(self, vocab_size, embed_dim, rnn_hidden_size,
                 attention_size, n_heads, l_seq_len, r_seq_len, emb_weights=None):
        super(CrossAttentionNet, self).__init__(vocab_size, embed_dim,
                                                 rnn_hidden_size, emb_weights=emb_weights)
        self.l_attention = MultiheadAttention(n_heads, rnn_hidden_size,
                                              att_size=attention_size)
        self.r_attention = MultiheadAttention(n_heads, rnn_hidden_size,
                                              att_size=attention_size)
        self.l_flatten = AttentionFlattener(l_seq_len)
        self.r_flatten = AttentionFlattener(r_seq_len)
        self.l_linear = nn.Linear(rnn_hidden_size, rnn_hidden_size)
        self.r_linear = nn.Linear(rnn_hidden_size, rnn_hidden_size)
        self.l_probas = None
        self.r_probas = None
        self.l_scores = None
        self.r_scores = None
        self.similarity = nn.CosineSimilarity()
        self.final_layer = nn.Linear(1,2)


    def forward(self, input_seq_l, input_seq_r):
        outputs_l = self.encoder_l(input_seq_l)
        _, self.l_probas = self.l_attention(outputs_l, outputs_l, outputs_l)
        self.l_probas = self.l_probas[0]
        self.l_scores = self.l_flatten(self.l_probas)
        outputs_l = self.l_scores * outputs_l
        left = torch.sum(outputs_l, dim=1)
        left = self.l_linear(left)

        outputs_r = self.encoder_r(input_seq_r)
        _, self.r_probas = self.r_attention(outputs_r, outputs_r, outputs_r)
        self.r_probas = self.r_probas[0]
        self.r_scores = self.r_flatten(self.r_probas)
        outputs_r = self.r_scores * outputs_r
        right = torch.sum(outputs_r, dim=1)
        right = self.l_linear(right)
        ans = self.similarity(left, right).unsqueeze(1)
        ans = self.final_layer(ans)
        ans = F.softmax(ans, dim=1)
        return ans