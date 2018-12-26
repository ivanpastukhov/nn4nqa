from custom_model.model import SimpleNet
from custom_model.train import Train
import numpy as np
import torch

net = SimpleNet(13, 3, 4)

q = torch.from_numpy(np.array(([[1,2,4,5,6,7],[4,3,2,9,5,4]])))
a = torch.from_numpy(np.array(([[1,2,4,5],[4,3,2,9]])))
t = torch.from_numpy(np.array(([0,1])))

print(net(q,a))

optimizer = torch.optim.SGD(net.parameters(), lr=0.2)
loss_func = torch.nn.CrossEntropyLoss()

trainer = Train()
trainer.fit(q,a,t,net,1,1000,loss_func,optimizer)

print(net(q,a))
