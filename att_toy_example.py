import numpy as np
from pprint import pprint
import torch
from custom_model.model import Attention, PositionalEncoding, SelfAttention

torch.manual_seed(42)

net = Attention(8)
encod = PositionalEncoding(8, 0.2, 7)
selfatt = SelfAttention()

hidden = torch.randn((3, 1, 8))
outputs = torch.randn((3, 7, 8))

# print('Results: \n', net(hidden, outputs))
# print(encod(outputs).size())

q = torch.randn((3, 9, 5))
k = torch.randn((3, 7, 5))
v = torch.randn((3, 7, 5))
scores, probas = selfatt.self_attention(q,k,v)

score = scores[2]
value = v[2]
proba = probas[2]
print('Proba sum: {}'.format(proba.sum()))

print('Value_size: ', value.size())
print('Proba size: ', proba.size())
print('Score size: ', score.size())
resp = []
for i in proba:
    print('Sum: ', i.sum())
    resp.append(torch.matmul(i.reshape((1, 7)), value))
print('Hand value: ', sum(resp))



print('Sum_proba:', proba[4].sum())
print('Score: \n',  score.sum(-2))