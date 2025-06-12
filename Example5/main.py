import numpy as np
from Genaratedata_pi_ion import generate_test_data, generate_train_data
import torch
import torch.nn as nn
import time
import os
import argparse
from Tools import grad, get_batch
from Net_type import XIDeepONet
from torch import optim, autograd

def main(args):
    if torch.cuda.is_available() and args.cuda == True:
        device ='cuda:0'
    else:
        device = 'cpu'
    model_type ='ionet'

    def loss_omega_p(data):
        data_op = data[2]
        data_op = data_op.to(device)
        data_op.requires_grad_()
        phi_sensor = data[1]
        output_o1 = model_1(phi_sensor, data_op)
        grad_op = autograd.grad(outputs=output_o1, inputs=data_op,
                                grad_outputs=torch.ones_like(output_o1),
                                create_graph=True, retain_graph=True, only_inputs=True)[0]
        du_dx = grad_op[:, 0]
        du_dx = torch.unsqueeze(du_dx, 1)
        du_dy = grad_op[:, 1]
        du_dy = torch.unsqueeze(du_dy, 1)
        du_dz = grad_op[:, 2]
        du_dz = torch.unsqueeze(du_dz, 1)
        du_dh = grad_op[:, 3]
        du_dh = torch.unsqueeze(du_dh, 1)
        # u对x的一阶导的导数
        grad_op_1 = autograd.grad(outputs=du_dx, inputs=data_op,
                                  grad_outputs=torch.ones_like(du_dx),
                                  create_graph=True, retain_graph=True, only_inputs=True)[0]
        du2_dx2 = grad_op_1[:, 0]
        du2_dx2 = torch.unsqueeze(du2_dx2, 1)
        du2_dxdh = grad_op_1[:, 3]
        du2_dxdh = torch.unsqueeze(du2_dxdh, 1)
        # u对y的一阶导的导数
        grad_op_2 = autograd.grad(outputs=du_dy, inputs=data_op,
                                  grad_outputs=torch.ones_like(du_dy),
                                  create_graph=True, retain_graph=True, only_inputs=True)[0]
        du2_dy2 = grad_op_2[:, 1]
        du2_dy2 = torch.unsqueeze(du2_dy2, 1)
        du2_dydh = grad_op_2[:, 3]
        du2_dydh = torch.unsqueeze(du2_dydh, 1)
        # u对z的一阶导的导数
        grad_op_3 = autograd.grad(outputs=du_dz, inputs=data_op,
                                  grad_outputs=torch.ones_like(du_dz),
                                  create_graph=True, retain_graph=True, only_inputs=True)[0]
        du2_dz2 = grad_op_3[:, 2]
        du2_dz2 = torch.unsqueeze(du2_dz2, 1)
        du2_dzdh = grad_op_3[:, 3]
        du2_dzdh = torch.unsqueeze(du2_dzdh, 1)
        # u对h的一阶导的导数
        grad_op_4 = autograd.grad(outputs=du_dh, inputs=data_op,
                                  grad_outputs=torch.ones_like(du_dh),
                                  create_graph=True, retain_graph=True, only_inputs=True)[0]
        du2_dh2 = grad_op_4[:, 3]
        du2_dh2 = torch.unsqueeze(du2_dh2, 1)
        dphi_dx = data[3]
        dphi_dy = data[4]
        dphi_dz = data[5]
        phi_laplace = data[6]
        laplace = (du2_dx2 + du2_dy2 + du2_dz2) + 2.*(du2_dxdh * dphi_dx + du2_dydh * dphi_dy + du2_dzdh * dphi_dz) \
                  + du2_dh2 * (dphi_dx ** 2 + dphi_dy ** 2 + dphi_dz ** 2) + du_dh * phi_laplace
        loss = nn.MSELoss()(laplace, data[-1])
        return loss

    def loss_omega_n(data):
        data_on = data[2]
        data_on = data_on.to(device)
        data_on.requires_grad_()
        phi_sensor = data[1]
        output_o2 = model_1(phi_sensor, data_on)
        grad_op = autograd.grad(outputs=output_o2, inputs=data_on,
                                grad_outputs=torch.ones_like(output_o2),
                                create_graph=True, retain_graph=True, only_inputs=True)[0]
        du_dx = grad_op[:, 0]
        du_dx = torch.unsqueeze(du_dx, 1)
        du_dy = grad_op[:, 1]
        du_dy = torch.unsqueeze(du_dy, 1)
        du_dz = grad_op[:, 2]
        du_dz = torch.unsqueeze(du_dz, 1)
        # u对x的一阶导的导数
        grad_op_1 = autograd.grad(outputs=du_dx, inputs=data_on,
                                  grad_outputs=torch.ones_like(du_dx),
                                  create_graph=True, retain_graph=True, only_inputs=True)[0]
        du2_dx2 = grad_op_1[:, 0]
        du2_dx2 = torch.unsqueeze(du2_dx2, 1)
        # u对y的一阶导的导数
        grad_op_2 = autograd.grad(outputs=du_dy, inputs=data_on,
                                  grad_outputs=torch.ones_like(du_dy),
                                  create_graph=True, retain_graph=True, only_inputs=True)[0]
        du2_dy2 = grad_op_2[:, 1]
        du2_dy2 = torch.unsqueeze(du2_dy2, 1)
        # u对z的一阶导的导数
        grad_op_3 = autograd.grad(outputs=du_dz, inputs=data_on,
                                  grad_outputs=torch.ones_like(du_dz),
                                  create_graph=True, retain_graph=True, only_inputs=True)[0]
        du2_dz2 = grad_op_3[:, 2]
        du2_dz2 = torch.unsqueeze(du2_dz2, 1)

        a_x = data[3]
        dadx = data[4]
        dady = data[5]
        dadz = data[6]
        u_lapl = du2_dx2 + du2_dy2 + du2_dz2
        laplace = a_x * u_lapl + dadx * du_dx + dady * du_dy + dadz * du_dz
        loss = nn.MSELoss()(laplace, data[-1])
        return loss

    def loss_boundary(data):
        data_b = data[2]
        data_b = data_b.to(device)
        phi_sensor = data[1]
        output_b = model_1(phi_sensor, data_b)
        loss = nn.MSELoss()(output_b, data[-1])
        return loss

    def loss_interface(data):
        data_if = data[2]
        data_if = data_if.to(device)
        data_if.requires_grad_()
        phi_sensor = data[1]
        output_if = model_1(phi_sensor, data_if)
        grad_if = autograd.grad(outputs=output_if, inputs=data_if,
                                grad_outputs=torch.ones_like(output_if),
                                create_graph=True, retain_graph=True, only_inputs=True)[0]
        du_dx = grad_if[:, 0]
        du_dx = torch.unsqueeze(du_dx, 1)
        du_dy = grad_if[:, 1]
        du_dy = torch.unsqueeze(du_dy, 1)
        du_dz = grad_if[:, 2]
        du_dz = torch.unsqueeze(du_dz, 1)
        du_dh = grad_if[:, 3]
        du_dh = torch.unsqueeze(du_dh, 1)

        dphi_dx = data[3]
        dphi_dy = data[4]
        dphi_dz = data[5]
        n = data[-1]
        dUp_dn = torch.sum((torch.cat((du_dx + du_dh * dphi_dx, du_dy + du_dh * dphi_dy, du_dz + du_dh * dphi_dz), 1) * n), 1)
        dUn_dn = torch.sum((torch.cat((du_dx, du_dy, du_dz), 1) * n), 1)
        loss = torch.mean((dUp_dn - 1. * dUn_dn) ** 2)
        return loss

    # test data
    test_data = generate_test_data(1000, device=device)
    # train data
    train_xop, train_xon, train_xb, train_xif = generate_train_data(args.sample_num, device=device)

    width = 150
    model_1 = XIDeepONet(sensor_dim=136, h_dim=width, in_dim=4, actv=nn.Tanh()).to(device)
    #print("Total number of paramerters in networks is {}  ".format(sum(x.numel() for x in model_1.parameters())))

    optimizer=optim.Adam(model_1.parameters(),lr=args.lr)
    t0=time.time()

    for i in range(args.epoch):
        input_p, input_n, input_b, input_if = get_batch(args.batch, train_xop, train_xon, train_xb, train_xif, device=device)
        optimizer.zero_grad()
        loss1 = loss_omega_p(input_p)
        loss2 = loss_omega_n(input_n)
        loss3 = loss_boundary(input_b)
        loss4 = loss_interface(input_if)
        loss = loss1 + loss2 + 10*loss3 + loss4
        loss.backward(retain_graph=True)
        optimizer.step()

        if i%args.print_epoch==0:
            print('epoch {}: training loss'.format(i), loss.item(),optimizer.param_groups[0]['lr'])
            print(loss1.item(),loss2.item(),loss3.item(),loss4.item())
            ###
            pre_label = model_1(test_data[1], test_data[2])
            relative_l2 = ((test_data[-1] - pre_label) ** 2).sum()
            relative_l2 = torch.sqrt(relative_l2 / (((test_data[-1]) ** 2).sum()))

            print('Rela L2 loss: ', relative_l2.item())
            print('\n')

        if (i + 1) % int(args.epoch / 100) == 0:
            optimizer.param_groups[0]['lr'] = optimizer.param_groups[0]['lr'] * 0.95




    print('time', time.time() - t0)
    if not os.path.isdir('./' + 'model/'): os.makedirs('./' + 'model/')
    torch.save(model_1, 'model/' + 'pi_ion_' + model_type + '_' + str(width) + '_' + str(args.epoch) + '.pkl')

if __name__ == '__main__':
    #torch.cuda.set_device(-1)
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_file',type=str, default='data/data_01_02/')
    parser.add_argument('--cuda', type=str, default=True)
    parser.add_argument('--batch', type=list, default=[20,10,10])
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--epoch', type=int, default=0000)
    parser.add_argument('--sample_num', type=int, default=20)
    parser.add_argument('--print_epoch', type=int, default=200)


    args = parser.parse_args(args=[])
    main(args)
