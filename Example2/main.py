import torch
import torch.nn as nn
import numpy as np
import argparse
import datetime
import time
from Network import XIDeepONet
from Train_NN import optimize_parameters_adam
from Generate_data import generate_train_data, generate_test_data
'''-------------------------Empty cache and check devices-------------------------'''
torch.cuda.empty_cache()
torch.set_default_dtype(torch.float64)
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
print('device = ', device)

def main(paras):
    model = XIDeepONet(sensor_dim=60, h_dim=paras.width, in_dim=3, actv=nn.Tanh()).to(device).float()
    train_xop, train_xon, train_xb, train_xif = generate_train_data(paras.sample_num, device)
    test_data = generate_test_data(1000, device)
    model = optimize_parameters_adam(lr=paras.lr,
                                     epochs=paras.epoch,
                                     print_epoch=paras.print_epoch,
                                     train_xop=train_xop,
                                     train_xon=train_xon,
                                     train_xb=train_xb,
                                     train_xif=train_xif,
                                     batch=paras.batch,
                                     model=model,
                                     test_data=test_data,
                                     device=device)
    torch.save(model.state_dict(), 'best_model.mdl')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch', type=list, default=[2000, 1000, 1000])
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--epoch', type=int, default=40000)
    parser.add_argument('--sample_num', type=int, default=20)
    parser.add_argument('--print_epoch', type=int, default=200)
    parser.add_argument('--width', type=int, default=100)
    paras = parser.parse_args()
    main(paras)