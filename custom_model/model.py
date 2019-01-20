from torch import nn
import torch
from torch.nn import functional as F
import logging
import sys
import time
import numpy as np
from custom_model.layers import MultiheadAttention, AttentionFlattener


logging.basicConfig(level=logging.WARNING)


class Encoder(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_size, bidirectional=False, dropout=False,
                 emb_weights=None, batch_first=True):
        super(Encoder, self).__init__()
        if not batch_first:
            raise NotImplementedError('The input size must be (batch, seq, feature)')
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.hidden_size = hidden_size
        self.dropout = dropout
        self.embedding = nn.Embedding(num_embeddings=self.vocab_size,
                                      embedding_dim=self.embed_dim)
        if bidirectional:
            # TODO: ...
            raise NotImplementedError()
        if not isinstance(emb_weights, type(None)):
            if emb_weights.shape != (self.vocab_size, self.embed_dim):
                raise ValueError('Size of embedding matrix must be equal to ones used in initialization.')
            emb_weights = torch.from_numpy(emb_weights).float()
            self.embedding = nn.Embedding.from_pretrained(emb_weights, freeze=True)
        # TODO: добавить другие RNN
        if dropout:
            self.rnn = nn.LSTM(input_size=self.embed_dim,
                               hidden_size=self.hidden_size,
                               num_layers=1,
                               batch_first=batch_first,
                               dropout=dropout,
                               bidirectional=bidirectional)
        else:
            self.rnn = nn.LSTM(input_size=self.embed_dim,
                               hidden_size=self.hidden_size,
                               num_layers=1,
                               batch_first=batch_first,
                               bidirectional=bidirectional)

    def freeze_embeddings(self):
        self.embedding.freeze = True

    def unfreeze_embeddings(self):
        self.embedding.freeze = False

    def forward(self, input_seq, hidden=None):
        # TODO: torch.nn.utils.rnn.pack_padded_sequence
        embedded = self.embedding(input_seq)
        logging.debug('Embedded size: {}'.format(embedded.size()))
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
        loss = self.loss_function(y_pred, y_val.squeeze(1))
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
        logging.debug(
            'X_left_size: {}, y_train.size: {}, X_right_size: {}'.format(X_left.size(), y_train.size(), X_right.size()))
        # assert len(X_left) == len(y_train) == len(X_right)
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
                x_l_batch = X_left[:,lb:rb].to(device)
                x_r_batch = X_right[:,lb:rb].to(device)
                y_train_batch = y_train[lb:rb].t().to(device)
                y_pred_batch = self.__call__(x_l_batch, x_r_batch)
                a = y_pred_batch
                b = y_train_batch.squeeze(1)
                batch_loss = self.loss_function(a, b).to(device)
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
    def __init__(self, vocab_size, embed_dim, hidden_size, emb_weights=None, dropout=None):
        super().__init__()
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.hidden_size = hidden_size
        self.encoder_l = Encoder(self.vocab_size,
                                 self.embed_dim,
                                 self.hidden_size,
                                 emb_weights=emb_weights,
                                 dropout=dropout)
        self.encoder_r = Encoder(self.vocab_size,
                                 self.embed_dim,
                                 self.hidden_size,
                                 emb_weights=emb_weights,
                                 dropout=dropout)
        self.hidden = nn.Linear(hidden_size*2, 64)
        self.answer = nn.Linear(64, 2)

    def forward(self, input_seq_l, input_seq_r):
        logging.debug('Inputs size: {},{}'.format(input_seq_l.size(), input_seq_r.size()))
        outputs_l = self.encoder_l(input_seq_l)[-1]
        outputs_r = self.encoder_r(input_seq_r)[-1]
        logging.debug('Outputs_l,r: {},{}'.format(outputs_l.size(), outputs_r.size()))
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