# -*- coding: utf-8 -*-
# @Time    : 2024/12/7 上午11:47
# @Author  : NJU_RanBi
import torch
import functools
import numpy as np
from torch import optim, autograd
import torch.nn as nn
import matplotlib.pyplot as plt
from Tools import get_batch


def optimize_parameters_adam(lr, epochs, print_epoch, train_xop, train_xon, train_xb, train_xif, batch, model, test_data, device):
    def loss_omega_p(data):
        data_op = data[2]
        data_op = data_op.to(device)
        data_op.requires_grad_()
        f_sensor = data[0]
        phi_sensor = data[1]
        output_o1 = model(f_sensor, phi_sensor, data_op)
        grad_op = autograd.grad(outputs=output_o1, inputs=data_op,
                                grad_outputs=torch.ones_like(output_o1),
                                create_graph=True, retain_graph=True, only_inputs=True)[0]
        du_dx = grad_op[:, 0]
        du_dx = torch.unsqueeze(du_dx, 1)
        du_dy = grad_op[:, 1]
        du_dy = torch.unsqueeze(du_dy, 1)
        du_dz = grad_op[:, 2]
        du_dz = torch.unsqueeze(du_dz, 1)
        # u对x的一阶导的导数
        grad_op_1 = autograd.grad(outputs=du_dx, inputs=data_op,
                                  grad_outputs=torch.ones_like(du_dx),
                                  create_graph=True, retain_graph=True, only_inputs=True)[0]
        du2_dx2 = grad_op_1[:, 0]
        du2_dx2 = torch.unsqueeze(du2_dx2, 1)
        du2_dxdz = grad_op_1[:, 2]
        du2_dxdz = torch.unsqueeze(du2_dxdz, 1)
        # u对y的一阶导的导数
        grad_op_2 = autograd.grad(outputs=du_dy, inputs=data_op,
                                  grad_outputs=torch.ones_like(du_dy),
                                  create_graph=True, retain_graph=True, only_inputs=True)[0]
        du2_dy2 = grad_op_2[:, 1]
        du2_dy2 = torch.unsqueeze(du2_dy2, 1)
        du2_dydz = grad_op_2[:, 2]
        du2_dydz = torch.unsqueeze(du2_dydz, 1)
        # u对z的一阶导的导数
        grad_op_3 = autograd.grad(outputs=du_dz, inputs=data_op,
                                  grad_outputs=torch.ones_like(du_dz),
                                  create_graph=True, retain_graph=True, only_inputs=True)[0]
        du2_dz2 = grad_op_3[:, 2]
        du2_dz2 = torch.unsqueeze(du2_dz2, 1)
        du2_dzdx = grad_op_3[:, 0]
        du2_dzdx = torch.unsqueeze(du2_dzdx, 1)
        du2_dzdy = grad_op_3[:, 1]
        du2_dzdy = torch.unsqueeze(du2_dzdy, 1)
        dphi_dx = 2 * data_op[:, 0]
        dphi_dx = torch.unsqueeze(dphi_dx, 1)
        dphi_dy = 2 * data_op[:, 1]
        dphi_dy = torch.unsqueeze(dphi_dy, 1)
        dphi2_dx2 = 2 * torch.ones_like(dphi_dx)
        dphi2_dy2 = 2 * torch.ones_like(dphi_dy)
        DU2_dx2 = du2_dx2 + du2_dxdz * dphi_dx + du2_dzdx * dphi_dx + du2_dz2 * dphi_dx * dphi_dx + du_dz * dphi2_dx2
        DU2_dy2 = du2_dy2 + du2_dydz * dphi_dy + du2_dzdy * dphi_dy + du2_dz2 * dphi_dy * dphi_dy + du_dz * dphi2_dy2
        laplace = DU2_dx2 + DU2_dy2
        loss = nn.MSELoss()(laplace, data[-1] / 1000)
        return loss

    def loss_omega_n(data):
        data_on = data[2]
        data_on = data_on.to(device)
        data_on.requires_grad_()
        f_sensor = data[0]
        phi_sensor = data[1]
        output_o2 = model(f_sensor, phi_sensor, data_on)
        grad_on = autograd.grad(outputs=output_o2, inputs=data_on,
                                grad_outputs=torch.ones_like(output_o2),
                                create_graph=True, retain_graph=True, only_inputs=True)[0]
        du_dx = grad_on[:, 0]
        du_dx = torch.unsqueeze(du_dx, 1)
        du_dy = grad_on[:, 1]
        du_dy = torch.unsqueeze(du_dy, 1)
        du_dz = grad_on[:, 2]
        du_dz = torch.unsqueeze(du_dz, 1)
        # u对x的一阶导的导数
        grad_on_1 = autograd.grad(outputs=du_dx, inputs=data_on,
                                  grad_outputs=torch.ones_like(du_dx),
                                  create_graph=True, retain_graph=True, only_inputs=True)[0]
        du2_dx2 = grad_on_1[:, 0]
        du2_dx2 = torch.unsqueeze(du2_dx2, 1)
        du2_dxdz = grad_on_1[:, 2]
        du2_dxdz = torch.unsqueeze(du2_dxdz, 1)
        # u对y的一阶导的导数
        grad_on_2 = autograd.grad(outputs=du_dy, inputs=data_on,
                                  grad_outputs=torch.ones_like(du_dy),
                                  create_graph=True, retain_graph=True, only_inputs=True)[0]
        du2_dy2 = grad_on_2[:, 1]
        du2_dy2 = torch.unsqueeze(du2_dy2, 1)
        du2_dydz = grad_on_2[:, 2]
        du2_dydz = torch.unsqueeze(du2_dydz, 1)
        # u对z的一阶导的导数
        grad_on_3 = autograd.grad(outputs=du_dz, inputs=data_on,
                                  grad_outputs=torch.ones_like(du_dz),
                                  create_graph=True, retain_graph=True, only_inputs=True)[0]
        du2_dz2 = grad_on_3[:, 2]
        du2_dz2 = torch.unsqueeze(du2_dz2, 1)
        du2_dzdx = grad_on_3[:, 0]
        du2_dzdx = torch.unsqueeze(du2_dzdx, 1)
        du2_dzdy = grad_on_3[:, 1]
        du2_dzdy = torch.unsqueeze(du2_dzdy, 1)
        dphi_dx = -2 * data_on[:, 0]
        dphi_dx = torch.unsqueeze(dphi_dx, 1)
        dphi_dy = -2 * data_on[:, 1]
        dphi_dy = torch.unsqueeze(dphi_dy, 1)
        dphi2_dx2 = -2 * torch.ones_like(dphi_dx)
        dphi2_dy2 = -2 * torch.ones_like(dphi_dy)
        DU2_dx2 = du2_dx2 + du2_dxdz * dphi_dx + du2_dzdx * dphi_dx + du2_dz2 * dphi_dx * dphi_dx + du_dz * dphi2_dx2
        DU2_dy2 = du2_dy2 + du2_dydz * dphi_dy + du2_dzdy * dphi_dy + du2_dz2 * dphi_dy * dphi_dy + du_dz * dphi2_dy2
        laplace = DU2_dx2 + DU2_dy2
        loss = nn.MSELoss()(laplace, data[-1])
        return loss

    def loss_boundary(data):
        data_b = data[2]
        data_b = data_b.to(device)
        f_sensor = data[0]
        phi_sensor = data[1]
        output_b = model(f_sensor, phi_sensor, data_b)
        loss = nn.MSELoss()(output_b, data[-1])
        return loss

    def loss_interface(data):
        data_if = data[2]
        data_if = data_if.to(device)
        data_if.requires_grad_()
        f_sensor = data[0]
        phi_sensor = data[1]
        output_if = model(f_sensor, phi_sensor, data_if)
        grad_if = autograd.grad(outputs=output_if, inputs=data_if,
                                grad_outputs=torch.ones_like(output_if),
                                create_graph=True, retain_graph=True, only_inputs=True)[0]
        du_dx = grad_if[:, 0]
        du_dx = torch.unsqueeze(du_dx, 1)
        du_dy = grad_if[:, 1]
        du_dy = torch.unsqueeze(du_dy, 1)
        du_dz = grad_if[:, 2]
        du_dz = torch.unsqueeze(du_dz, 1)
        dphi_dx = 2 * data_if[:, 0]
        dphi_dx = torch.unsqueeze(dphi_dx, 1)
        dphi_dy = 2 * data_if[:, 1]
        dphi_dy = torch.unsqueeze(dphi_dy, 1)
        n = data[-1]
        dUp_dn = torch.sum((torch.cat((du_dx + du_dz * dphi_dx, du_dy + du_dz * dphi_dy), 1) * n), 1)
        dUn_dn = torch.sum((torch.cat((du_dx - du_dz * dphi_dx, du_dy - du_dz * dphi_dy), 1) * n), 1)
        loss = torch.mean((dUp_dn - dUn_dn / 1000) ** 2)
        return loss

    optimizer = optim.Adam(model.parameters(), lr=lr)

    for i in range(epochs):
        input_p, input_n, input_b, input_if = get_batch(batch, train_xop, train_xon, train_xb, train_xif, device=device)
        optimizer.zero_grad()
        loss1 = loss_omega_p(input_p)
        loss2 = loss_omega_n(input_n)
        loss3 = loss_boundary(input_b)
        loss4 = loss_interface(input_if)
        loss = loss1 + loss2 + 100 * loss3 + 10 * loss4
        loss.backward(retain_graph=True)
        optimizer.step()

        if i % print_epoch == 0:
            print('epoch {}: training loss'.format(i), loss.item(), optimizer.param_groups[0]['lr'])
            print(loss1.item(), loss2.item(), loss3.item(), loss4.item())
            ###
            pre_label = model(test_data[0], test_data[1], test_data[2])
            relative_l2 = ((test_data[-1] - pre_label) ** 2).sum()
            relative_l2 = torch.sqrt(relative_l2 / (((test_data[-1]) ** 2).sum()))
            print('Rela L2 loss: ', relative_l2.item())
            print('\n')

        if (i + 1) % int(epochs / 100) == 0:
            optimizer.param_groups[0]['lr'] = optimizer.param_groups[0]['lr'] * 0.95
    return model