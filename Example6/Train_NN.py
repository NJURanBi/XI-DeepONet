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
        du_da = grad_op[:, 0]
        du_da = torch.unsqueeze(du_da, 1)
        du_db = grad_op[:, 1]
        du_db = torch.unsqueeze(du_db, 1)
        du_dc = grad_op[:, 2]
        du_dc = torch.unsqueeze(du_dc, 1)
        du_dd = grad_op[:, 3]
        du_dd = torch.unsqueeze(du_dd, 1)
        du_de = grad_op[:, 4]
        du_de = torch.unsqueeze(du_de, 1)
        du_df = grad_op[:, 5]
        du_df = torch.unsqueeze(du_df, 1)
        du_dg = grad_op[:, 6]
        du_dg = torch.unsqueeze(du_dg, 1)
        # u对a的一阶导的导数
        grad_op_1 = autograd.grad(outputs=du_da, inputs=data_op,
                                  grad_outputs=torch.ones_like(du_da),
                                  create_graph=True, retain_graph=True, only_inputs=True)[0]
        du2_da2 = grad_op_1[:, 0]
        du2_da2 = torch.unsqueeze(du2_da2, 1)
        du2_dadg = grad_op_1[:, 6]
        du2_dadg = torch.unsqueeze(du2_dadg, 1)
        # u对b的一阶导的导数
        grad_op_2 = autograd.grad(outputs=du_db, inputs=data_op,
                                  grad_outputs=torch.ones_like(du_db),
                                  create_graph=True, retain_graph=True, only_inputs=True)[0]
        du2_db2 = grad_op_2[:, 1]
        du2_db2 = torch.unsqueeze(du2_db2, 1)
        du2_dbdg = grad_op_2[:, 6]
        du2_dbdg = torch.unsqueeze(du2_dbdg, 1)
        # u对c的一阶导的导数
        grad_op_3 = autograd.grad(outputs=du_dc, inputs=data_op,
                                  grad_outputs=torch.ones_like(du_dc),
                                  create_graph=True, retain_graph=True, only_inputs=True)[0]
        du2_dc2 = grad_op_3[:, 2]
        du2_dc2 = torch.unsqueeze(du2_dc2, 1)
        du2_dcdg = grad_op_3[:, 6]
        du2_dcdg = torch.unsqueeze(du2_dcdg, 1)
        # u对d的一阶导的导数
        grad_op_4 = autograd.grad(outputs=du_dd, inputs=data_op,
                                  grad_outputs=torch.ones_like(du_dd),
                                  create_graph=True, retain_graph=True, only_inputs=True)[0]
        du2_dd2 = grad_op_4[:, 3]
        du2_dd2 = torch.unsqueeze(du2_dd2, 1)
        du2_dddg = grad_op_4[:, 6]
        du2_dddg = torch.unsqueeze(du2_dddg, 1)
        # u对e的一阶导的导数
        grad_op_5 = autograd.grad(outputs=du_de, inputs=data_op,
                                  grad_outputs=torch.ones_like(du_de),
                                  create_graph=True, retain_graph=True, only_inputs=True)[0]
        du2_de2 = grad_op_5[:, 4]
        du2_de2 = torch.unsqueeze(du2_de2, 1)
        du2_dedg = grad_op_5[:, 6]
        du2_dedg = torch.unsqueeze(du2_dedg, 1)
        # u对f的一阶导的导数
        grad_op_6 = autograd.grad(outputs=du_df, inputs=data_op,
                                  grad_outputs=torch.ones_like(du_df),
                                  create_graph=True, retain_graph=True, only_inputs=True)[0]
        du2_df2 = grad_op_6[:, 5]
        du2_df2 = torch.unsqueeze(du2_df2, 1)
        du2_dfdg = grad_op_6[:, 6]
        du2_dfdg = torch.unsqueeze(du2_dfdg, 1)
        # u对g的一阶导的导数
        grad_op_7 = autograd.grad(outputs=du_dg, inputs=data_op,
                                  grad_outputs=torch.ones_like(du_dg),
                                  create_graph=True, retain_graph=True, only_inputs=True)[0]
        du2_dg2 = grad_op_7[:, 6]
        du2_dg2 = torch.unsqueeze(du2_dg2, 1)
        du2_dgda = grad_op_7[:, 0]
        du2_dgda = torch.unsqueeze(du2_dgda, 1)
        du2_dgdb = grad_op_7[:, 1]
        du2_dgdb = torch.unsqueeze(du2_dgdb, 1)
        du2_dgdc = grad_op_7[:, 2]
        du2_dgdc = torch.unsqueeze(du2_dgdc, 1)
        du2_dgdd = grad_op_7[:, 3]
        du2_dgdd = torch.unsqueeze(du2_dgdd, 1)
        du2_dgde = grad_op_7[:, 4]
        du2_dgde = torch.unsqueeze(du2_dgde, 1)
        du2_dgdf = grad_op_7[:, 5]
        du2_dgdf = torch.unsqueeze(du2_dgdf, 1)
        dphi_da = 2 * data_op[:, 0]
        dphi_da = torch.unsqueeze(dphi_da, 1)
        dphi_db = 2 * data_op[:, 1]
        dphi_db = torch.unsqueeze(dphi_db, 1)
        dphi_dc = 2 * data_op[:, 2]
        dphi_dc = torch.unsqueeze(dphi_dc, 1)
        dphi_dd = 2 * data_op[:, 3]
        dphi_dd = torch.unsqueeze(dphi_dd, 1)
        dphi_de = 2 * data_op[:, 4]
        dphi_de = torch.unsqueeze(dphi_de, 1)
        dphi_df = 2 * data_op[:, 5]
        dphi_df = torch.unsqueeze(dphi_df, 1)
        dphi2_da2 = 2 * torch.ones_like(dphi_da)
        dphi2_db2 = 2 * torch.ones_like(dphi_db)
        dphi2_dc2 = 2 * torch.ones_like(dphi_dc)
        dphi2_dd2 = 2 * torch.ones_like(dphi_dd)
        dphi2_de2 = 2 * torch.ones_like(dphi_de)
        dphi2_df2 = 2 * torch.ones_like(dphi_df)

        Dphi2_da2 = du2_da2 + du2_dadg * dphi_da + du2_dgda * dphi_da + du2_dg2 * dphi_da * dphi_da + du_dg * dphi2_da2
        Dphi2_db2 = du2_db2 + du2_dbdg * dphi_db + du2_dgdb * dphi_db + du2_dg2 * dphi_db * dphi_db + du_dg * dphi2_db2
        Dphi2_dc2 = du2_dc2 + du2_dcdg * dphi_dc + du2_dgdc * dphi_dc + du2_dg2 * dphi_dc * dphi_dc + du_dg * dphi2_dc2
        Dphi2_dd2 = du2_dd2 + du2_dddg * dphi_dd + du2_dgdd * dphi_dd + du2_dg2 * dphi_dd * dphi_dd + du_dg * dphi2_dd2
        Dphi2_de2 = du2_de2 + du2_dedg * dphi_de + du2_dgde * dphi_de + du2_dg2 * dphi_de * dphi_de + du_dg * dphi2_de2
        Dphi2_df2 = du2_df2 + du2_dfdg * dphi_df + du2_dgdf * dphi_df + du2_dg2 * dphi_df * dphi_df + du_dg * dphi2_df2
        laplace = Dphi2_da2+Dphi2_db2+Dphi2_dc2+Dphi2_dd2+Dphi2_de2+Dphi2_df2
        loss = nn.MSELoss()(laplace, data[-1])
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
        du_da = grad_on[:, 0]
        du_da = torch.unsqueeze(du_da, 1)
        du_db = grad_on[:, 1]
        du_db = torch.unsqueeze(du_db, 1)
        du_dc = grad_on[:, 2]
        du_dc = torch.unsqueeze(du_dc, 1)
        du_dd = grad_on[:, 3]
        du_dd = torch.unsqueeze(du_dd, 1)
        du_de = grad_on[:, 4]
        du_de = torch.unsqueeze(du_de, 1)
        du_df = grad_on[:, 5]
        du_df = torch.unsqueeze(du_df, 1)
        du_dg = grad_on[:, 6]
        du_dg = torch.unsqueeze(du_dg, 1)
        # u对a的一阶导的导数
        grad_on_1 = autograd.grad(outputs=du_da, inputs=data_on,
                                  grad_outputs=torch.ones_like(du_da),
                                  create_graph=True, retain_graph=True, only_inputs=True)[0]
        du2_da2 = grad_on_1[:, 0]
        du2_da2 = torch.unsqueeze(du2_da2, 1)
        du2_dadg = grad_on_1[:, 6]
        du2_dadg = torch.unsqueeze(du2_dadg, 1)
        # u对b的一阶导的导数
        grad_on_2 = autograd.grad(outputs=du_db, inputs=data_on,
                                  grad_outputs=torch.ones_like(du_db),
                                  create_graph=True, retain_graph=True, only_inputs=True)[0]
        du2_db2 = grad_on_2[:, 1]
        du2_db2 = torch.unsqueeze(du2_db2, 1)
        du2_dbdg = grad_on_2[:, 6]
        du2_dbdg = torch.unsqueeze(du2_dbdg, 1)
        # u对c的一阶导的导数
        grad_on_3 = autograd.grad(outputs=du_dc, inputs=data_on,
                                  grad_outputs=torch.ones_like(du_dc),
                                  create_graph=True, retain_graph=True, only_inputs=True)[0]
        du2_dc2 = grad_on_3[:, 2]
        du2_dc2 = torch.unsqueeze(du2_dc2, 1)
        du2_dcdg = grad_on_3[:, 6]
        du2_dcdg = torch.unsqueeze(du2_dcdg, 1)
        # u对d的一阶导的导数
        grad_on_4 = autograd.grad(outputs=du_dd, inputs=data_on,
                                  grad_outputs=torch.ones_like(du_dd),
                                  create_graph=True, retain_graph=True, only_inputs=True)[0]
        du2_dd2 = grad_on_4[:, 3]
        du2_dd2 = torch.unsqueeze(du2_dd2, 1)
        du2_dddg = grad_on_4[:, 6]
        du2_dddg = torch.unsqueeze(du2_dddg, 1)
        # u对e的一阶导的导数
        grad_on_5 = autograd.grad(outputs=du_de, inputs=data_on,
                                  grad_outputs=torch.ones_like(du_de),
                                  create_graph=True, retain_graph=True, only_inputs=True)[0]
        du2_de2 = grad_on_5[:, 4]
        du2_de2 = torch.unsqueeze(du2_de2, 1)
        du2_dedg = grad_on_5[:, 6]
        du2_dedg = torch.unsqueeze(du2_dedg, 1)
        # u对f的一阶导的导数
        grad_on_6 = autograd.grad(outputs=du_df, inputs=data_on,
                                  grad_outputs=torch.ones_like(du_df),
                                  create_graph=True, retain_graph=True, only_inputs=True)[0]
        du2_df2 = grad_on_6[:, 5]
        du2_df2 = torch.unsqueeze(du2_df2, 1)
        du2_dfdg = grad_on_6[:, 6]
        du2_dfdg = torch.unsqueeze(du2_dfdg, 1)
        # u对g的一阶导的导数
        grad_on_7 = autograd.grad(outputs=du_dg, inputs=data_on,
                                  grad_outputs=torch.ones_like(du_dg),
                                  create_graph=True, retain_graph=True, only_inputs=True)[0]
        du2_dg2 = grad_on_7[:, 6]
        du2_dg2 = torch.unsqueeze(du2_dg2, 1)
        du2_dgda = grad_on_7[:, 0]
        du2_dgda = torch.unsqueeze(du2_dgda, 1)
        du2_dgdb = grad_on_7[:, 1]
        du2_dgdb = torch.unsqueeze(du2_dgdb, 1)
        du2_dgdc = grad_on_7[:, 2]
        du2_dgdc = torch.unsqueeze(du2_dgdc, 1)
        du2_dgdd = grad_on_7[:, 3]
        du2_dgdd = torch.unsqueeze(du2_dgdd, 1)
        du2_dgde = grad_on_7[:, 4]
        du2_dgde = torch.unsqueeze(du2_dgde, 1)
        du2_dgdf = grad_on_7[:, 5]
        du2_dgdf = torch.unsqueeze(du2_dgdf, 1)
        dphi_da = -2 * data_on[:, 0]
        dphi_da = torch.unsqueeze(dphi_da, 1)
        dphi_db = -2 * data_on[:, 1]
        dphi_db = torch.unsqueeze(dphi_db, 1)
        dphi_dc = -2 * data_on[:, 2]
        dphi_dc = torch.unsqueeze(dphi_dc, 1)
        dphi_dd = -2 * data_on[:, 3]
        dphi_dd = torch.unsqueeze(dphi_dd, 1)
        dphi_de = -2 * data_on[:, 4]
        dphi_de = torch.unsqueeze(dphi_de, 1)
        dphi_df = -2 * data_on[:, 5]
        dphi_df = torch.unsqueeze(dphi_df, 1)
        dphi2_da2 = -2 * torch.ones_like(dphi_da)
        dphi2_db2 = -2 * torch.ones_like(dphi_db)
        dphi2_dc2 = -2 * torch.ones_like(dphi_dc)
        dphi2_dd2 = -2 * torch.ones_like(dphi_dd)
        dphi2_de2 = -2 * torch.ones_like(dphi_de)
        dphi2_df2 = -2 * torch.ones_like(dphi_df)

        Dphi2_da2 = du2_da2 + du2_dadg * dphi_da + du2_dgda * dphi_da + du2_dg2 * dphi_da * dphi_da + du_dg * dphi2_da2
        Dphi2_db2 = du2_db2 + du2_dbdg * dphi_db + du2_dgdb * dphi_db + du2_dg2 * dphi_db * dphi_db + du_dg * dphi2_db2
        Dphi2_dc2 = du2_dc2 + du2_dcdg * dphi_dc + du2_dgdc * dphi_dc + du2_dg2 * dphi_dc * dphi_dc + du_dg * dphi2_dc2
        Dphi2_dd2 = du2_dd2 + du2_dddg * dphi_dd + du2_dgdd * dphi_dd + du2_dg2 * dphi_dd * dphi_dd + du_dg * dphi2_dd2
        Dphi2_de2 = du2_de2 + du2_dedg * dphi_de + du2_dgde * dphi_de + du2_dg2 * dphi_de * dphi_de + du_dg * dphi2_de2
        Dphi2_df2 = du2_df2 + du2_dfdg * dphi_df + du2_dgdf * dphi_df + du2_dg2 * dphi_df * dphi_df + du_dg * dphi2_df2
        laplace = Dphi2_da2 + Dphi2_db2 + Dphi2_dc2 + Dphi2_dd2 + Dphi2_de2 + Dphi2_df2
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
        du_dg = grad_if[:, 6]
        du_dg = torch.unsqueeze(du_dg, 1)
        dphi_da = 4 * data_if[:, 0]
        dphi_da = torch.unsqueeze(dphi_da, 1)
        dphi_db = 4 * data_if[:, 1]
        dphi_db = torch.unsqueeze(dphi_db, 1)
        dphi_dc = 4 * data_if[:, 2]
        dphi_dc = torch.unsqueeze(dphi_dc, 1)
        dphi_dd = 4 * data_if[:, 3]
        dphi_dd = torch.unsqueeze(dphi_dd, 1)
        dphi_de = 4 * data_if[:, 4]
        dphi_de = torch.unsqueeze(dphi_de, 1)
        dphi_df = 4 * data_if[:, 5]
        dphi_df = torch.unsqueeze(dphi_df, 1)
        n = data[-1]
        psi = 2*data[3][:, 0]
        loss = torch.mean(torch.pow(torch.sum(torch.cat((du_dg * dphi_da, du_dg * dphi_db,
                                                          du_dg * dphi_dc, du_dg * dphi_dd,
                                                          du_dg * dphi_de, du_dg * dphi_df,), dim=1) * n, 1) - psi, 2))
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