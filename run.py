from model.model import QAMatching
import numpy as np
import torch


def main():
    man = QAMatching(11, 5, 12, )

    l = 5
    q_batch = torch.tensor(np.random.random_integers(0, 10, (2,12)))
    q_batch_length = [5,5]
    d_batch = torch.tensor(np.random.random_integers(0, 10, (2,8)))
    d_batch_length = [5,5]
    print(q_batch)
    print(man(q_batch, q_batch_length, d_batch, d_batch_length, training=False))
    print('Parameters: ', len(list(man.parameters())))


if __name__ == '__main__':
    main()
