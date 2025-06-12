# -*- coding: utf-8 -*-
# @Time    : 2024/12/5 下午6:31
# @Author  : NJU_RanBi
import torch
import torch.nn as nn
import numpy as np
import argparse
import datetime
import time
from Network import XIDeepONet
from Train_NN import optimize_parameters_adam
from Generate_data import generate_train_data
import matplotlib.pyplot as plt
from Tools import To_tensor

r0 = 0.5
r1 = 0.
r2 = -0.
k0 = 5


def get_sensors_location(num, r0, r1, r2, k0):
    x1 = np.linspace(-1, 1, num)
    x2 = np.linspace(-1, 1, num)
    X, Y = np.meshgrid(x1, x2)
    X = X.flatten()[:, None]
    Y = Y.flatten()[:, None]
    x = np.hstack((X, Y))
    phi = level_set_function(x, r0, r1, r2, k0)
    index_p = np.where(phi >= 0)[0]
    index_n = np.where(phi < 0)[0]
    xp = x[index_p, :]
    xn = x[index_n, :]
    return xp, xn

def get_boudary_sensors_location(num):
    x1 = np.linspace(-1, 1, num)[:9][:, None]
    x2 = np.linspace(-1, 1, num)[1:][:, None]
    xb1 = np.hstack((x1, np.ones_like(x1)))
    xb2 = np.hstack((x2, -np.ones_like(x2)))
    xb3 = np.hstack((np.ones_like(x1), x2))
    xb4 = np.hstack((-np.ones_like(x2), x1))
    xb = np.vstack((xb1, xb2, xb3, xb4))
    return xb

def get_theta(x):
    x1 = x[:, 0]
    x2 = x[:, 1]
    theta = np.arctan2(x2, x1)
    theta = theta[:, None]
    return theta


def get_phi(x, r0, r1, r2, k0):
    phi = level_set_function(x, r0, r1, r2, k0)[:, 0]
    return phi

def level_set_function(x, r0, r1, r2, k0):
    theta = get_theta(x)
    x1 = x[:, 0]
    x2 = x[:, 1]
    r = np.sqrt(x1 ** 2 + x2 ** 2)[:, None]
    lf = r - r0 - r1 * np.sin(theta) - r2 * np.sin(k0 * theta)
    return lf


def add_dimension(x, r0, r1, r2, k0):
    phi = level_set_function(x, r0, r1, r2, k0)
    index = phi >= 0.
    index = index.astype(float)
    x_add = np.hstack((x, np.abs(phi * index)))
    return x_add


def u_x(x, r0, r1, r2, k0):
    x1 = x[:, 0]
    x2 = x[:, 1]
    theta = get_theta(x)[:, 0]
    r = np.sqrt(x1 ** 2 + x2 ** 2)
    lf = level_set_function(x, r0, r1, r2, k0)[:, 0]
    up = (r0 * r / (r0 + r2 * np.sin(k0 * theta)))**3 + r0**3 * (1/1000 - 1)
    un = (r0 * r / (r0 + r2 * np.sin(k0 * theta)))**3 / 1000
    u_x = np.where(lf > 0, up, un)
    u_x = u_x[:, None]
    return u_x


def up_x(x, r0, r1, r2, k0):
    r = np.sqrt(np.sum(np.power(x, 2), 1))[:, None]
    theta = get_theta(x)
    ux = (r0 * r / (r0 + r2 * np.sin(k0 * theta)))**3 + r0**3 * (1/1000 - 1)
    return ux


def un_x(x, r0, r1, r2, k0):
    r = np.sqrt(np.sum(np.power(x, 2), 1))[:, None]
    theta = get_theta(x)
    ux = (r0 * r / (r0 + r2 * np.sin(k0 * theta)))**3 / 1000
    return ux


def get_fp(x, r0, r1, r2, k0):
    h = 1e-3
    x1 = x[:, 0][:, None]
    x2 = x[:, 1][:, None]
    xl = np.hstack((x1 - h, x2))
    xr = np.hstack((x1 + h, x2))
    xt = np.hstack((x1, x2 + h))
    xb = np.hstack((x1, x2 - h))
    f = (up_x(xl, r0, r1, r2, k0) + up_x(xr, r0, r1, r2, k0) + up_x(xt, r0, r1, r2, k0) + up_x(xb, r0, r1, r2, k0) - 4 * up_x(x, r0, r1, r2, k0)) / (h ** 2)
    return f


def get_fn(x, r0, r1, r2, k0):
    h = 1e-3
    x1 = x[:, 0][:, None]
    x2 = x[:, 1][:, None]
    xl = np.hstack((x1 - h, x2))
    xr = np.hstack((x1 + h, x2))
    xt = np.hstack((x1, x2 + h))
    xb = np.hstack((x1, x2 - h))
    f = 1000 * (un_x(xl, r0, r1, r2, k0) + un_x(xr, r0, r1, r2, k0) + un_x(xt, r0, r1, r2, k0) + un_x(xb, r0, r1, r2, k0) - 4 * un_x(x, r0, r1, r2, k0)) / (h ** 2)
    return f



'''-------------------------Empty cache and check devices-------------------------'''
torch.cuda.empty_cache()
torch.set_default_dtype(torch.float64)
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
print('device = ', device)


model = XIDeepONet(sensor_dim1=100, sensor_dim2=36, h_dim=100, in_dim=3, actv=nn.Tanh()).to(device).float()
model.load_state_dict(torch.load('best_model.mdl'))

with torch.no_grad():
    x1 = np.linspace(-1, 1, 201)
    x2 = np.linspace(-1, 1, 201)
    X, Y = np.meshgrid(x1, x2)
    Z = np.hstack((X.flatten()[:, None], Y.flatten()[:, None]))
    dataz = add_dimension(Z, r0, r1, r2, k0)
    ur = u_x(Z, r0, r1, r2, k0)
    sensor_resolution = 10  # 传感器的分辨率
    sensor_location_p, sensor_location_n = get_sensors_location(sensor_resolution, r0, r1, r2, k0)
    sensor_boundary_location = get_boudary_sensors_location(sensor_resolution)
    sensor_f_p = get_fp(sensor_location_p, r0, r1, r2, k0)[:, 0]
    sensor_f_n = get_fn(sensor_location_n, r0, r1, r2, k0)[:, 0]
    sensor_f = np.hstack((sensor_f_p, sensor_f_n))

    sensor_g = up_x(sensor_boundary_location, r0, r1, r2, k0)[:, 0]

    sensor_phi_p = get_phi(sensor_location_p, r0, r1, r2, k0)
    sensor_phi_n = get_phi(sensor_location_n, r0, r1, r2, k0)
    sensor_phi = np.hstack((sensor_phi_p, sensor_phi_n))
    test_data = (sensor_phi * np.ones((201**2, sensor_resolution ** 2)), sensor_f * np.ones((201**2, sensor_resolution ** 2)),
                 sensor_g * np.ones((201**2, 4 * (sensor_resolution - 1))), dataz)
    test_data = To_tensor(test_data, device=device)

ur = ur.reshape(201, 201)
pred = model(test_data[0], test_data[1], test_data[2], test_data[3])
pred = pred.cpu().detach().numpy()
pred = pred.reshape(201, 201)

h = plt.imshow( pred - ur, interpolation='nearest', cmap='coolwarm',
                extent=[-1, 1, -1, 1],
                origin='lower', aspect='auto')
plt.title('Error distribution')
plt.xlabel('x')
plt.ylabel('y')
plt.colorbar(h)
plt.show()