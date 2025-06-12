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
    model = XIDeepONet(sensor_dim=100, h_dim=100, in_dim=2, actv=nn.Tanh()).to(device).float()
    model.load_state_dict(torch.load('best_model.mdl'))
    train_left, train_right, train_interface, train_sample_bl, train_sample_br = generate_train_data(paras.sample_num, device=device, p=paras.p)
    test_data = generate_test_data(data_file=paras.data_file, device=device)
    model = optimize_parameters_adam(lr=paras.lr,
                                     epochs=paras.epoch,
                                     print_epoch=paras.print_epoch,
                                     train_xop=train_right,
                                     train_xon=train_left,
                                     train_xbl=train_sample_bl,
                                     train_xbr=train_sample_br,
                                     train_xif=train_interface,
                                     batch=paras.batch,
                                     model=model,
                                     test_data=test_data,
                                     device=device)
    torch.save(model.state_dict(), 'best_model.mdl')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_file', type=str, default='data/')
    parser.add_argument('--cuda', type=str, default=True)
    parser.add_argument('--batch', type=list, default=[5000, 5000, 5000])
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--epoch', type=int, default=40000)
    parser.add_argument('--sample_num', type=int, default=10000)
    parser.add_argument('--p', type=int, default=10)
    parser.add_argument('--print_epoch', type=int, default=200)
    paras = parser.parse_args()
    main(paras)