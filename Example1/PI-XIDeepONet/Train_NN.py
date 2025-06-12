import torch
import functools
import numpy as np
from torch import optim, autograd
import torch.nn as nn
import matplotlib.pyplot as plt
from Tools import get_batch


def optimize_parameters_adam(lr, epochs, print_epoch, train_xop, train_xon, train_xbl, train_xbr, train_xif, batch, model, test_data, device):
    def loss_omega_p(data):
        # Ω^+ 的损失函数
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
        du_dz = grad_op[:, 1]
        du_dz = torch.unsqueeze(du_dz, 1)

        grad_op_1 = autograd.grad(outputs=du_dx, inputs=data_op,
                                  grad_outputs=torch.ones_like(du_dx),
                                  create_graph=True, retain_graph=True, only_inputs=True)[0]
        du2_dx2 = grad_op_1[:, 0]
        du2_dx2 = torch.unsqueeze(du2_dx2, 1)
        du2_dxdz = grad_op_1[:, 1]
        du2_dxdz = torch.unsqueeze(du2_dxdz, 1)

        grad_op_2 = autograd.grad(outputs=du_dz, inputs=data_op,
                                  grad_outputs=torch.ones_like(du_dz),
                                  create_graph=True, retain_graph=True, only_inputs=True)[0]
        du2_dz2 = grad_op_2[:, 1]
        du2_dz2 = torch.unsqueeze(du2_dz2, 1)
        laplace = -0.5*(du2_dx2 + 2. * du2_dxdz + du2_dz2)
        loss = nn.MSELoss()(laplace, data[-1])
        return loss

    def loss_omega_n(data):
        # Ω^- 的损失函数
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
        du_dz = grad_on[:, 1]
        du_dz = torch.unsqueeze(du_dz, 1)

        grad_on_1 = autograd.grad(outputs=du_dx, inputs=data_on,
                                  grad_outputs=torch.ones_like(du_dx),
                                  create_graph=True, retain_graph=True, only_inputs=True)[0]
        du2_dx2 = grad_on_1[:, 0]
        du2_dx2 = torch.unsqueeze(du2_dx2, 1)
        du2_dxdz = grad_on_1[:, 1]
        du2_dxdz = torch.unsqueeze(du2_dxdz, 1)

        grad_on_2 = autograd.grad(outputs=du_dz, inputs=data_on,
                                  grad_outputs=torch.ones_like(du_dz),
                                  create_graph=True, retain_graph=True, only_inputs=True)[0]
        du2_dz2 = grad_on_2[:, 1]
        du2_dz2 = torch.unsqueeze(du2_dz2, 1)
        laplace = -0.1*(du2_dx2 - 2. * du2_dxdz + du2_dz2)
        loss = nn.MSELoss()(laplace, data[-1])
        return loss

    def loss_boundary(data):
        # 边界损失函数
        data_b = data[2]
        data_b = data_b.to(device)
        f_sensor = data[0]
        phi_sensor = data[1]
        output_b = model(f_sensor, phi_sensor, data_b)
        loss = nn.MSELoss()(output_b, 0. * output_b)
        return loss

    def loss_interface(data):
        # 界面法向导数损失函数
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
        du_dz = grad_if[:, 1]
        du_dz = torch.unsqueeze(du_dz, 1)
        grad_i1 = 0.1 * (du_dx - du_dz)
        grad_i2 = 0.5 * (du_dx + du_dz)
        loss = nn.MSELoss()(grad_i1, grad_i2)
        return loss

    optimizer = optim.Adam(model.parameters(), lr=lr)

    for i in range(epochs):
        input_left, input_right, input_gamma, input_boundaryl, input_boundaryr= get_batch(batch, train_xon, train_xop, train_xif, train_xbl, train_xbr, device=device)
        optimizer.zero_grad()
        loss1 = loss_omega_n(input_left)
        loss2 = loss_omega_p(input_right)
        loss3 = loss_boundary(input_boundaryl)
        loss4 = loss_boundary(input_boundaryr)
        loss5 = loss_interface(input_gamma)
        loss = loss1 + loss2 + 100 * loss3 + 100 * loss4 + loss5
        loss.backward(retain_graph=True)
        optimizer.step()

        if i % print_epoch == 0:
            print('epoch {}: training loss'.format(i), loss.item(),optimizer.param_groups[0]['lr'])
            print(loss1.item(), loss2.item(), loss3.item(), loss4.item(), loss5.item())
            ###
            pre_label = model(test_data[0], test_data[1], test_data[2])
            relative_l2 = ((test_data[-1] - pre_label) ** 2).sum()
            relative_l2 = torch.sqrt(relative_l2 / (((test_data[-1]) ** 2).sum()))
            print('Rela L2 loss: ', relative_l2.item())
            print('\n')

        if (i + 1) % int(epochs / 100) == 0:
            optimizer.param_groups[0]['lr'] = optimizer.param_groups[0]['lr'] * 0.95
    return model