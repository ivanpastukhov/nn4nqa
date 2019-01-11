import numpy as np
from pprint import pprint
import torch
from custom_model.model import Attention, PositionalEncoding

torch.manual_seed(42)

net = Attention(8)

encod = PositionalEncoding(8, 0.2, 7)

hidden = torch.randn((3, 1, 8))
pprint(hidden.size())
outputs = torch.randn((3, 7, 8))

print('Results: \n', net(hidden, outputs))


# print(encod(outputs).size())
