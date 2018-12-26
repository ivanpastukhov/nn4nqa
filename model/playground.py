from custom_model.model import SimpleNet
from custom_model.train import Train
import numpy as np
import torch
from matplotlib import pyplot as plt

net = SimpleNet(13, 3, 4)

q = torch.from_numpy(np.array(([[1,2,4,5,6,7],[4,3,2,9,5,4]])))
a = torch.from_numpy(np.array(([[1,2,4,5],[4,3,2,9]])))
t = torch.from_numpy(np.array(([0,1])))
ts = t.size()


print(net(q,a))
print(t.size(), len(q))
optimizer = torch.optim.SGD(net.parameters(), lr=0.5)
loss_func = torch.nn.CrossEntropyLoss()

trainer = Train()
trainer.fit(q,a,t,net,1,100,loss_func,optimizer)

plt.plot(trainer.steps, trainer.losses)
plt.show()