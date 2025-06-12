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
alpha = 0.7


def add_dimension(x, alpha):
    m = mark(x, alpha)
    index1 = m >= 0
    index2 = m < 0
    index1 = index1.float()
    index2 = index2.float()
    index1 = torch.unsqueeze(index1, 1)
    index2 = torch.unsqueeze(index2, 1)
    lf = level_set_fuction(x, alpha) * (index1 - index2)
    x_3 = torch.cat((x, lf), 1)
    return x_3

def level_set_fuction(x, alpha):
    x1 = x[:,0]
    x2 = x[:,1]
    lf = torch.pow(x1, 2) + torch.pow(x2, 2) - alpha ** 2
    lf = torch.unsqueeze(lf, 1)
    return lf

def mark(x, alpha):
    lf = level_set_fuction(x, alpha)
    m = torch.squeeze(lf ,1)
    return m

def u_x(x, alpha):
    m = mark(x, alpha)
    index = m
    r = torch.sqrt(torch.sum(torch.pow(x, 2), 1))
    up = r**3 / 1000
    un = r ** 3 + (0.001 - 1.)*alpha**3
    u_x = torch.where(index >= 0, up, un)
    u_x = torch.unsqueeze(u_x, 1)
    return u_x

def get_sensors_location(num, alpha):
    x1 = np.linspace(-1, 1, num)
    x2 = np.linspace(-1, 1, num)
    X, Y = np.meshgrid(x1, x2)
    X = X.flatten()[:, None]
    Y = Y.flatten()[:, None]
    x = np.hstack((X, Y))
    x1 = x[:, 0]
    x2 = x[:, 1]
    r2 = x1 ** 2 + x2 ** 2
    index = np.where(r2 <= 1)[0]
    x = x[index, :]
    r2 = r2[index]
    phi = r2 - alpha ** 2
    index_p = np.where(phi >= 0)[0]
    index_n = np.where(phi < 0)[0]
    xp = x[index_p, :]
    xn = x[index_n, :]
    return xp, xn

def get_f(p, x):
    r2 = x[:, 0]**2 + x[:, 1]**2
    f = 9. * p * np.sqrt(r2)
    return f

def get_phi(x, alpha):
    r2 = x[:, 0]**2 + x[:, 1]**2
    phi = np.abs(r2 - alpha**2)
    return phi

def get_data(x, alpha):
    p = 1
    sensor_resolution = 10  # 传感器的分辨率
    sensor_location_p, sensor_location_n = get_sensors_location(sensor_resolution, alpha)
    x_3 = add_dimension(x, alpha)
    sensor_f_p = get_f(p, sensor_location_p)
    sensor_f_n = get_f(p, sensor_location_n)
    sensor_f = np.hstack((sensor_f_p, sensor_f_n))
    sensor_phi_p = get_phi(sensor_location_p, alpha)
    sensor_phi_n = -get_phi(sensor_location_n, alpha)
    sensor_phi = np.hstack((sensor_phi_p, sensor_phi_n))
    sensor_f = torch.tensor(sensor_f).float()
    sensor_phi = torch.tensor(sensor_phi).float()
    data = (sensor_f, sensor_phi, x_3)
    return data

def get_points_circle(x):
    x1 = x[:, 0]
    x2 = x[:, 1]
    r2 = x1**2 + x2**2
    index = torch.where(r2 <= 1.)[0]
    data = x[index, :]
    return data


if torch.cuda.is_available():
    device ='cuda'
else:
    device = 'cpu'
model_type ='ionet'
model = XIDeepONet(sensor_dim=60, h_dim=100, in_dim=3, actv=nn.Tanh()).float()
model.load_state_dict(torch.load('best_model.mdl'))
'''------------------------Plot-------------------------'''
with torch.no_grad():
    x1 = torch.linspace(-1, 1, 201)
    x2 = torch.linspace(-1, 1, 201)
    X, Y = torch.meshgrid(x1, x2)
    Z = torch.cat((Y.flatten()[:, None], Y.T.flatten()[:, None]), dim=1)
    Z = get_points_circle(Z)
    ur = u_x(Z, alpha)
    data = get_data(Z, alpha)
    dataz = add_dimension(Z, alpha)
    pred = model(data[0], data[1], data[2])
    X_tsd = Z
    abserr = torch.abs(pred - ur)

l2 = torch.sqrt(torch.mean(torch.pow(pred - ur, 2)))
rel_l2 = l2 / torch.sqrt(torch.mean(torch.pow(ur, 2)))
print('l2相对误差:', rel_l2.item())

np.savetxt('X_std.txt', Z)
np.savetxt(f'pred_{alpha}.txt', pred)
np.savetxt(f'abserr_{alpha}.txt', abserr)
# set up a figure twice as wide as it is tall
fig = plt.figure(figsize=plt.figaspect(0.3))

#===============
#  First subplot
#===============
ax = fig.add_subplot(1, 1, 1, projection='3d')
surf = ax.scatter(X_tsd[:,0:1], X_tsd[:,1:2], pred, c=pred, cmap='plasma')
fig.colorbar(surf, shrink=0.5, aspect=5)
plt.xlabel('x')
plt.ylabel('y')
plt.title('Approximate solution')
ax.view_init(elev=30, azim=30)
plt.savefig(f"Exa2,pred_u,r={alpha}.jpg", bbox_inches='tight', dpi=600)
plt.clf()

ax = fig.add_subplot(1, 1, 1, projection='3d')
surf = ax.scatter(X_tsd[:,0:1], X_tsd[:,1:2], abserr, c=abserr, cmap='plasma')
fig.colorbar(surf, shrink=0.5, aspect=5)
plt.xlabel('x')
plt.ylabel('y')
plt.title('Error distribution')
ax.view_init(elev=30, azim=30)
plt.savefig(f"Exa2,error,r={alpha}.jpg", bbox_inches='tight', dpi=600)
plt.show()






