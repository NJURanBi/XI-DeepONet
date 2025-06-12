# -*- coding: utf-8 -*-
# @Time    : 2024/12/5 下午3:42
# @Author  : NJU_RanBi

import torch
import torch.nn as nn

class XIDeepONet(nn.Module):
    def __init__(self, sensor_dim, h_dim, in_dim, actv = nn.Tanh()):
        super(XIDeepONet, self).__init__()
        self.actv1 = actv
        #self.actv2 = nn.ReLU()
        self.actv2 = actv
        self.b_phi_linear_input = nn.Linear(sensor_dim, h_dim)
        self.b_phi_linear1 = nn.Linear(h_dim, h_dim)
        self.b_phi_linear2 = nn.Linear(h_dim, h_dim)
        self.b_phi_linear3 = nn.Linear(h_dim, h_dim)
        self.b_phi_linear4 = nn.Linear(h_dim, h_dim)

        self.b_f_linear_input = nn.Linear(sensor_dim, h_dim)
        self.b_f_linear1 = nn.Linear(h_dim, h_dim)
        self.b_f_linear2 = nn.Linear(h_dim, h_dim)
        self.b_f_linear3 = nn.Linear(h_dim, h_dim)
        self.b_f_linear4 = nn.Linear(h_dim, h_dim)

        self.t_linear_input = nn.Linear(in_dim, h_dim)
        self.t_linear1 = nn.Linear(h_dim, h_dim)
        self.t_linear2 = nn.Linear(h_dim, h_dim)
        self.t_linear3 = nn.Linear(h_dim, h_dim)
        self.t_linear4 = nn.Linear(h_dim, h_dim)

        self.p = self.__init_params()

    def forward(self,f_tr, phi_tr, x_tr):

        phi_tr = phi_tr
        out = self.actv2(self.b_phi_linear_input(phi_tr))
        out = self.actv2(self.b_phi_linear1(out))
        out = self.actv2(self.b_phi_linear2(out))
        out = self.actv2(self.b_phi_linear3(out))
        branch_phi = self.actv2(self.b_phi_linear4(out))

        f_tr = f_tr
        out = self.actv2(self.b_f_linear_input(f_tr))
        out = self.actv2(self.b_f_linear1(out))
        out = self.actv2(self.b_f_linear2(out))
        out = self.actv2(self.b_f_linear3(out))
        branch_f = self.actv2(self.b_f_linear4(out))

        y = x_tr
        out = self.actv1(self.t_linear_input(y))
        out = self.actv1(self.t_linear1(out))
        out = self.actv1(self.t_linear2(out))
        out = self.actv1(self.t_linear3(out))
        trunk = self.actv1(self.t_linear4(out))

        output = torch.sum(branch_f * branch_phi * trunk, dim=-1, keepdim=True) + self.p['bias']

        return output

    def __init_params(self):
        params = nn.ParameterDict()
        params['bias'] = nn.Parameter(torch.zeros([1]))
        return params


class DeepONet(nn.Module):
    def __init__(self, sensor_dim, h_dim, in_dim, actv = nn.ReLU):
        super(DeepONet, self).__init__()
        self.actv = actv

        self.b_beta_linear_input = nn.Linear(sensor_dim, h_dim)
        self.b_beta_linear1 = nn.Linear(h_dim, h_dim)
        self.b_beta_linear2 = nn.Linear(h_dim, h_dim)
        self.b_beta_linear3 = nn.Linear(h_dim, h_dim)
        self.b_beta_linear4 = nn.Linear(h_dim, h_dim)


        self.t_linear_input = nn.Linear(in_dim, h_dim)
        self.t_linear1 = nn.Linear(h_dim, h_dim)
        self.t_linear2 = nn.Linear(h_dim, h_dim)
        self.t_linear3 = nn.Linear(h_dim, h_dim)
        self.t_linear4 = nn.Linear(h_dim, h_dim)

        self.p = self.__init_params()

    def forward(self, X):

        beta_tr = X[0]
        out = self.actv(self.b_beta_linear_input(beta_tr))
        out = self.actv(self.b_beta_linear1(out))
        out = self.actv(self.b_beta_linear2(out))
        out = self.actv(self.b_beta_linear3(out))
        branch_beta = self.actv(self.b_beta_linear4(out))

        y = X[1]
        out = self.actv(self.t_linear_input(y))
        out = self.actv(self.t_linear1(out))
        out = self.actv(self.t_linear2(out))
        out = self.actv(self.t_linear3(out))
        trunk = self.actv(self.t_linear4(out))

        output = torch.sum(branch_beta * trunk, dim=-1, keepdim=True) + self.p['bias']

        return output

    def __init_params(self):
        params = nn.ParameterDict()
        params['bias'] = nn.Parameter(torch.zeros([1]))
        return params
