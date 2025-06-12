import torch
import torch.nn as nn
import numpy as np
import time
import os
from Tools import get_batch_single
from Network import XIDeepONet, DeepONet
from Get_data_XIDeepONet import get_test_data, get_train_data
import argparse

alpha = np.array(0.5)


def main(args):
    if torch.cuda.is_available() and args.cuda == True:
        device = 'cuda'
    else:
        device = 'cpu'

    X_test = get_test_data(data_file=args.data_file, device=device)
    X_train = get_test_data(data_file=args.data_file, device=device)

    model_type = 'XIDeepONet'
    width = 100
    model_1 = XIDeepONet(sensor_dim=100, h_dim=width, in_dim=2, actv=nn.ReLU()).to(device)
    print("Total number of paramerters in networks is {}  ".format(sum(x.numel() for x in model_1.parameters())))

    import torch.optim as optim
    optimizer = optim.Adam(model_1.parameters(), lr=args.lr)
    t0 = time.time()

    for i in range(args.epoch):
        input = get_batch_single(args.batch, X_train, device=device)
        optimizer.zero_grad()

        loss = 10 * nn.MSELoss()(model_1(input), input[-1])
        loss.backward(retain_graph=True)
        optimizer.step()

        if i % args.print_epoch == 0:
            print('epoch {}: training loss'.format(i), loss.item(), optimizer.param_groups[0]['lr'])
            print('loss: ', loss.item())
            tt = time.time()
            pre_label = model_1(X_test)
            relative_l2 = ((X_test[-1] - pre_label) ** 2).sum()
            relative_l2 = torch.sqrt(relative_l2 / (((X_test[-1]) ** 2).sum()))

            print('Rela L2 loss is: ', relative_l2.item(), 'Test time:', (time.time() - tt) / 2)

        if (i + 1) % int(args.epoch / 100) == 0:
            optimizer.param_groups[0]['lr'] = optimizer.param_groups[0]['lr']

    print('time', time.time() - t0)
    if not os.path.isdir('./' + 'model/'): os.makedirs('./' + 'model/')
    torch.save(model_1, 'model/' + 'dd_ion_' + model_type + '_' + str(width) + '_' + str(args.epoch) + '.pkl')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_file', type=str, default='data/')
    parser.add_argument('--model_type', type=str, default='Stacked')
    parser.add_argument('--cuda', type=str, default=True)
    parser.add_argument('--batch', type=int, default=1000)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--epoch', type=int, default=10000)
    parser.add_argument('--sample_num', type=int, default=10000)
    parser.add_argument('--print_epoch', type=int, default=200)
    args = parser.parse_args(args=[])
    main(args)


