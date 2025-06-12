# -*- coding: utf-8 -*-
# @Time    : 2024/12/6 下午7:52
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

t = 0.9
beta_p = 10.
beta_n = 1.

def Circle_x(t):
    x = 0.5*t - 0.25
    return x


def Circle_y(t):
    y = 0.5*t - 0.25
    return y


def Circle_r(t):
    r = 0.1 * t + 0.4
    return r

def get_sensors_location(num, t):
    x1 = np.linspace(-1, 1, num)
    x2 = np.linspace(-1, 1, num)
    X, Y = np.meshgrid(x1, x2)
    X = X.flatten()[:, None]
    Y = Y.flatten()[:, None]
    x = np.hstack((X, Y))
    phi = level_set_function(x, t)
    index_p = np.where(phi >= 0)[0]
    index_n = np.where(phi < 0)[0]
    xp = x[index_p, :]
    xn = x[index_n, :]
    return xp, xn


def get_boudary_sensors_location(num, t):
    x1 = np.linspace(-1, 1, num)[:9][:, None]
    x2 = np.linspace(-1, 1, num)[1:][:, None]
    xb1 = np.hstack((x1, np.ones_like(x1)))
    xb2 = np.hstack((x2, -np.ones_like(x2)))
    xb3 = np.hstack((np.ones_like(x1), x2))
    xb4 = np.hstack((-np.ones_like(x2), x1))
    xb = np.vstack((xb1, xb2, xb3, xb4))
    return xb

def level_set_function(x, t):
    x1 = x[:, 0][:, None]
    x2 = x[:, 1][:, None]
    lf = (x1 - Circle_x(t)) ** 2 + (x2 - Circle_y(t)) ** 2 - Circle_r(t) ** 2
    return lf


def add_dimension(x):
    m = mark(x)
    index1 = m >= 0
    index2 = m < 0
    index1 = index1.astype(float)
    index2 = index2.astype(float)
    index1 = index1[:, None]
    index2 = index2[:, None]
    lf = level_set_function(x, t) * (index1 - index2)
    x_3 = np.hstack((x, lf))
    return x_3


def mark(x):
    lf = level_set_function(x, t)
    m = lf[:, 0]
    return m

def u_x(x):
    c_x = Circle_x(t)
    c_y = Circle_y(t)
    c_r = Circle_r(t)
    x1 = x[:, 0]
    x2 = x[:, 1]
    m = mark(x)
    index = m
    u1 = np.exp(-(x1 - c_x)**2 - (x2 - c_y)**2) / 10 + (1 - 0.1) * np.exp(-c_r ** 2)
    u2 = np.exp(-(x1 - c_x)**2 - (x2 - c_y)**2)
    u_x = np.where(index >= 0, u1, u2)
    u_x = u_x[:, None]
    return u_x

def get_phi(x, t):
    phi = np.abs(level_set_function(x, t))[:, 0]
    return phi


def get_f(x, t):
    c_x = Circle_x(t)
    c_y = Circle_y(t)
    x1 = x[:, 0]
    x2 = x[:, 1]
    f = 4 * np.exp(-(x1 - c_x)**2 - (x2 - c_y)**2) * ((x1 - c_x)**2 + (x2 - c_y)**2 - 1)
    return f


def get_g(x, t):
    c_x = Circle_x(t)
    c_y = Circle_y(t)
    c_r = Circle_r(t)
    x1 = x[:, 0]
    x2 = x[:, 1]
    g = np.exp(-(x1 - c_x)**2 - (x2 - c_y)**2) / 10 + (1 - 0.1) * np.exp(-c_r ** 2)
    return g

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
    Z = np.hstack((Y.flatten()[:, None], Y.T.flatten()[:, None]))
    dataz = add_dimension(Z)
    ur = u_x(Z)
    sensor_resolution = 10  # 传感器的分辨率
    sensor_location_p, sensor_location_n = get_sensors_location(sensor_resolution, t)
    sensor_boundary_location = get_boudary_sensors_location(sensor_resolution, t)
    sensor_f_p = get_f(sensor_location_p, t) 
    sensor_f_n = get_f(sensor_location_n, t)
    sensor_f = np.hstack((sensor_f_p, sensor_f_n))
    sensor_g = get_g(sensor_boundary_location, t)
    sensor_phi_p = get_phi(sensor_location_p, t)
    sensor_phi_n = -get_phi(sensor_location_n, t)
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