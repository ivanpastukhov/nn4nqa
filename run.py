import pandas as pd
import torch
import numpy as np
import os
import pickle
from custom_model.model import SimpleNet, SAttendedSimpleNet, SAttendedNet

def run():
    USE_CUDA = torch.cuda.is_available()
    device = torch.device("cuda" if USE_CUDA else "cpu")
    print('Device: ', device)

    def read_pickle(fname):
        with open(fname, 'rb') as fin:
            return pickle.load(fin)

    df_train = pd.read_pickle('./data/processed/wikiqa_df_train.pickle')
    df_test = pd.read_pickle('./data/processed/wikiqa_df_test.pickle')
    voc = read_pickle('./data/processed/vocabulary.pickle')

    df_train = df_train.iloc[:12]

    print('Train shape: {} \n\
    Test shape: {}'.format(df_train.shape, df_test.shape))

    net_simple = SimpleNet(voc['voc_len'], 64, 64)
    net_att = SAttendedNet(voc['voc_len'], 128, 64, 32, 1, 22, 287)

    Xq = np.array(df_train.Question_encoded.values.tolist())
    Xa = np.array(df_train.Sentence_encoded.values.tolist())
    t = np.array(df_train.Label.values.tolist())

    Xq = torch.from_numpy(Xq)
    Xa = torch.from_numpy(Xa)
    t = torch.from_numpy(t)

    batch_size = 50
    epochs = 5

    optimizer = torch.optim.Adam
    loss_func = torch.nn.CrossEntropyLoss(weight=torch.tensor([0.05, 1.]).to(device))

    net_simple.fit(Xq, Xa, t, batch_size, epochs, loss_func, optimizer, device)
    net_att.fit(Xq, Xa, t, batch_size, epochs, loss_func, optimizer, device)
    net_att.parameters()

if __name__ == '__main__':
    run()