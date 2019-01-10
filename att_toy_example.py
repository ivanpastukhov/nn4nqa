import numpy as np
from pprint import pprint
import torch
from custom_model.model import Attention, PositionalEncoding

net = Attention(8)

encod = PositionalEncoding(8, 0.2, 7)


hidden = torch.randn((1, 3, 8))
pprint(hidden.size())
outputs = torch.randn((7, 3, 8))

print('Results ', net(hidden, outputs))


print(encod(outputs).size())