# -*- coding: utf-8 -*-
# @Time    : 2024/12/5 下午3:42
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
        data_op = data[3]
        data_op = data_op.to(device)
        data_op.requires_grad_()
        phi_sensor = data[0]
        f_sensor = data[1]
        b_sensor = data[2]
        output_o1 = model(phi_sensor, f_sensor, b_sensor, data_op)
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
        dphi_dx = data[4]
        dphi_dy = data[5]
        dphi2_dx2 = data[6]
        dphi2_dy2 = data[7]
        DU2_dx2 = du2_dx2 + du2_dxdz * dphi_dx + du2_dzdx * dphi_dx + du2_dz2 * dphi_dx * dphi_dx + du_dz * dphi2_dx2
        DU2_dy2 = du2_dy2 + du2_dydz * dphi_dy + du2_dzdy * dphi_dy + du2_dz2 * dphi_dy * dphi_dy + du_dz * dphi2_dy2
        laplace = DU2_dx2 + DU2_dy2
        loss = torch.mean(torch.pow(laplace - data[8], 2))
        return loss

    def loss_omega_n(data):
        data_on = data[3]
        data_on = data_on.to(device)
        data_on.requires_grad_()
        phi_sensor = data[0]
        f_sensor = data[1]
        b_sensor = data[2]
        output_o2 = model(phi_sensor, f_sensor, b_sensor, data_on)
        grad_on = autograd.grad(outputs=output_o2, inputs=data_on,
                                grad_outputs=torch.ones_like(output_o2),
                                create_graph=True, retain_graph=True, only_inputs=True)[0]
        du_dx = grad_on[:, 0]
        du_dx = torch.unsqueeze(du_dx, 1)
        du_dy = grad_on[:, 1]
        du_dy = torch.unsqueeze(du_dy, 1)
        # u对x的一阶导的导数
        grad_on_1 = autograd.grad(outputs=du_dx, inputs=data_on,
                                  grad_outputs=torch.ones_like(du_dx),
                                  create_graph=True, retain_graph=True, only_inputs=True)[0]
        du2_dx2 = grad_on_1[:, 0]
        du2_dx2 = torch.unsqueeze(du2_dx2, 1)
        # u对y的一阶导的导数
        grad_on_2 = autograd.grad(outputs=du_dy, inputs=data_on,
                                  grad_outputs=torch.ones_like(du_dy),
                                  create_graph=True, retain_graph=True, only_inputs=True)[0]
        du2_dy2 = grad_on_2[:, 1]
        du2_dy2 = torch.unsqueeze(du2_dy2, 1)
        DU2_dx2 = du2_dx2
        DU2_dy2 = du2_dy2
        laplace = DU2_dx2 + DU2_dy2
        loss = torch.mean(torch.pow(laplace - data[4], 2))
        return loss

    def loss_boundary(data):
        data_b = data[3]
        data_b = data_b.to(device)
        phi_sensor = data[0]
        f_sensor = data[1]
        b_sensor = data[2]
        output_b = model(phi_sensor, f_sensor, b_sensor, data_b)
        loss = torch.mean(torch.pow(output_b - data[4], 2))
        return loss

    def loss_interface(data):
        data_if = data[3]
        data_if = data_if.to(device)
        data_if.requires_grad_()
        phi_sensor = data[0]
        f_sensor = data[1]
        b_sensor = data[2]
        output_if = model(phi_sensor, f_sensor, b_sensor, data_if)
        grad_if = autograd.grad(outputs=output_if, inputs=data_if,
                                grad_outputs=torch.ones_like(output_if),
                                create_graph=True, retain_graph=True)[0]
        du_dz = grad_if[:, 2]
        du_dz = torch.unsqueeze(du_dz, 1)
        dphi_dx = data[4]
        dphi_dy = data[5]
        nor = data[7]
        n1 = nor[:, 0][:, None]
        n2 = nor[:, 1][:, None]
        loss = torch.mean(((dphi_dx * du_dz) * n1 + (dphi_dy * du_dz) * n2 - data[6]) ** 2)
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

        if i%print_epoch==0:
            print('epoch {}: training loss'.format(i), loss.item(), optimizer.param_groups[0]['lr'])
            print(loss1.item(),loss2.item(),loss3.item(),loss4.item())
            ###
            pre_label = model(test_data[0], test_data[1], test_data[2], test_data[3])
            relative_l2 = ((test_data[-1] - pre_label) ** 2).sum()
            relative_l2 = torch.sqrt(relative_l2 / (((test_data[-1]) ** 2).sum()))
            print('Rela L2 loss: ', relative_l2.item())
            print('\n')

        if (i + 1) % int(epochs / 100) == 0:
            optimizer.param_groups[0]['lr'] = optimizer.param_groups[0]['lr'] * 0.95
    return model