import pandas as pd
import torch
import numpy as np
import os
import pickle
import matplotlib.pyplot as plt
from custom_model.model import SimpleNet, SAttendedSimpleNet, SAttendedNet, CrossAttentionNet
import seaborn as sns
from sklearn.metrics import roc_auc_score


USE_CUDA = torch.cuda.is_available()
device = torch.device("cuda" if USE_CUDA else "cpu")
print('Device: ', device)

def read_pickle(fname):
    with open(fname, 'rb') as fin:
        return pickle.load(fin)

df_train = pd.read_pickle('./data/processed/wikiqa_df_train.pickle')
df_test = pd.read_pickle('./data/processed/wikiqa_df_test.pickle')
df_test, df_val = np.split(df_test.sample(frac=1., random_state=42), 2)
emb_weights = np.load('./data/processed/index2vector.npy')

vocab_size = emb_weights.shape[0]
embed_dim = emb_weights.shape[1]

# df_train = df_train.iloc[:100]

print('Train shape: {} \n\
Test shape: {} \n\
Val shape {}: '.format(df_train.shape, df_test.shape, df_val.shape))


net_simple = SimpleNet(vocab_size, embed_dim, 64, emb_weights)
# net_att = SAttendedSimpleNet(voc['voc_len'], 128, 128, 64, 3)
net_att = SAttendedNet(vocab_size, embed_dim, 64, 32, 1, 22, 287, emb_weights)
net_crossover = CrossAttentionNet(vocab_size, embed_dim, 64, 32, 1, 22, 287, emb_weights)

Xq = np.array(df_train.Question_encoded.values[:50].tolist())
Xa = np.array(df_train.Sentence_encoded.values[:50].tolist())
t = np.array(df_train.Label.values[:50].tolist())

print(torch.from_numpy(Xq).size())
Xq = torch.from_numpy(Xq).permute(1,0)
Xa = torch.from_numpy(Xa).permute(1,0)
t = torch.from_numpy(t).unsqueeze(0)

# print(Xq.size(), Xa.size(), t.size())

batch_size = 27
epochs = 30

Xq_val = np.array(df_val.Question_encoded.values.tolist())
Xa_val = np.array(df_val.Sentence_encoded.values.tolist())
t_val = np.array(df_val.Label.values.tolist())
val_data = [(torch.from_numpy(Xq_val).permute(1,0), torch.from_numpy(Xa_val).permute(1,0)), torch.from_numpy(t_val).unsqueeze(1)]

optimizer = torch.optim.Adam
# loss_func = torch.nn.CrossEntropyLoss(weight=torch.tensor([0.05, 0.95]).to(device))
loss_func = torch.nn.CrossEntropyLoss()

# net_crossover.fit(Xq, Xa, t, batch_size, epochs, loss_func, optimizer, device, 90., val_data)
net_simple.fit(Xq, Xa, t, batch_size, epochs, loss_func, optimizer, device, None, val_data)
net_att.fit(Xq, Xa, t, batch_size, epochs, loss_func, optimizer, device, 90., val_data)