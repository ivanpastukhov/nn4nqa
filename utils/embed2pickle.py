
# Запускаем, чтобы преобразовать pre-trained embedding'и в словари и сатрицу весов.

import pickle
import bcolz
import numpy as np
from tqdm import tqdm


glove_path = '../data/embedding/glove.6B/'

words = []
idx = 0
word2idx = {}
vectors = bcolz.carray(np.zeros(1), rootdir=f'{glove_path}/6B.50.dat', mode='w')

with open(f'{glove_path}/glove.6B.50d.txt', 'rb') as f:
    for _, l in enumerate(tqdm(f)):
        line = l.decode().split()
        word = line[0]
        words.append(word)
        word2idx[word] = idx
        idx += 1
        vect = np.array(line[1:]).astype(np.float)
        vectors.append(vect)

print(vectors[0])
vectors = bcolz.carray(vectors[1:].reshape((400001, 50)), rootdir=f'{glove_path}/6B.50.dat', mode='w')
vectors.flush()
pickle.dump(words, open(f'{glove_path}/6B.50_words.pkl', 'wb'))
pickle.dump(word2idx, open(f'{glove_path}/6B.50_idx.pkl', 'wb'))