import numpy as np
import torch
import torch.nn as nn
import time
import os
import argparse
from Tools import grad, get_batch
from Network import XIDeepONet
from torch import optim, autograd
from matplotlib import pyplot as plt
from Tools import To_tensor
from Generate_data import generate_test_data
alpha = 0.5

if torch.cuda.is_available():
    device ='cuda'
else:
    device = 'cpu'

model = XIDeepONet(sensor_dim=233, h_dim=100, in_dim=7, actv=nn.Tanh()).to(device).float()
model.load_state_dict(torch.load('best_model.mdl'))
test_data = generate_test_data(100000, device)
pre_label = model(test_data[0], test_data[1], test_data[2])
relative_l2 = ((test_data[-1] - pre_label) ** 2).sum()
relative_l2 = torch.sqrt(relative_l2 / (((test_data[-1]) ** 2).sum()))
print('Rela L2 loss: ', relative_l2.item())








