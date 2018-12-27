import pandas as pd
import logging
from custom_model.model import SimpleNet
logging.basicConfig(level=logging.DEBUG)

class Train():
    def __init__(self):
        self.steps = []
        self.losses = []
        pass

    def fit(self, X_train_q, X_train_a,  y_train, model, batch_size, epochs, loss_function, optimizer, device):
        assert len(X_train_q) == len(y_train) == len(X_train_a)
        len_dataset = len(X_train_a)
        step = 0
        for epoch in range(epochs):
            lb = 0
            rb = batch_size
            while rb <= len_dataset:
                x_q_batch = X_train_q[lb:rb].to(device)
                x_a_batch = X_train_a[lb:rb].to(device)
                y_train_batch = y_train[lb:rb].to(device)
                y_pred_batch = model(x_q_batch, x_a_batch)
                loss = loss_function(y_pred_batch, y_train_batch).to(device)
                self.losses.append(loss.item())
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                rb += batch_size
                lb += batch_size
                step += 1
                self.steps.append(step)
            print('Epoch: {}, loss: {}'.format(epoch, loss))
