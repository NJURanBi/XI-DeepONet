import numpy as np
from Generate_data import generate_train_data, generate_test_data
import torch
import torch.nn as nn
import time
import os
import argparse
from torch import autograd
from Tools import get_batch
from Network import XIDeepONet
import torch.optim as optim

import matplotlib.pyplot as plt

def main(args):
    device = 'cpu'
    width = 100

    X_test = generate_test_data(data_file=args.data_file, device=device)
    model = XIDeepONet(sensor_dim=100, h_dim=width, in_dim=2, actv=nn.Tanh()).to(device)
    model.load_state_dict(torch.load('best_model.mdl'))
    pre_label = model(X_test[0], X_test[1], X_test[2])
    relative_l2 = ((X_test[-1] - pre_label) ** 2).sum()
    relative_l2 = torch.sqrt(relative_l2 / (((X_test[-1]) ** 2).sum()))
    print(relative_l2)
    pre_label = torch.squeeze(pre_label, 1)
    pre_label = pre_label.detach().numpy()
    real = X_test[-1]
    real = torch.squeeze(real, 1)
    real = real.detach().numpy()
    x = np.linspace(0,1,1000)

    plt.plot(x, real, color='blue', label='True solution')
    plt.plot(x, pre_label, color='red', label='Pre solution')
    plt.title(f'Relative-L2 error is {relative_l2}')
    plt.legend()
    plt.show()


if __name__ == '__main__':
    # torch.cuda.set_device(0)
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_file', type=str, default='data/')
    parser.add_argument('--cuda', type=str, default=True)
    parser.add_argument('--batch', type=list, default=[1000, 1000, 1000])
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--epoch', type=int, default=40000)
    parser.add_argument('--sample_num', type=int, default=100)
    parser.add_argument('--p', type=int,default=20)
    parser.add_argument('--print_epoch', type=int, default=200)

    args = parser.parse_args()
    main(args)

