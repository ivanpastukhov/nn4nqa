import collections




class Voc:
    def __init__(self):
        self.token2index = {'<PAD>': 0, '<SOS>': 1, '<EOS>': 2, '<UNK>': 3}
        self.index2token = {v: k for k, v in self.token2index.items()}
        self.voclen = len(self.token2index)
        self.__lookslike_len__ = 10

    def extend_vocab(self, iterable):
        if not isinstance(iterable, collections.Iterable):
            raise ValueError('Value must be an iterable.')
        else:
            iterable = set(iterable)
            iterable = iterable - self.token2index.keys()
            ids = range(self.voclen, len(iterable) + self.voclen)
            self.token2index.update(dict(zip(iterable, ids)))
            self.index2token = {v: k for k, v in self.token2index.items()}
            self.voclen = len(self.token2index)

    def __call__(self):
        print('Vocabulary size: ', self.voclen)
        print('token2index looks like: ', list(self.token2index.items())[:self.__lookslike_len__], ', ...')
        print('index2token looks like: ', list(self.index2token.items())[:self.__lookslike_len__], ', ...')