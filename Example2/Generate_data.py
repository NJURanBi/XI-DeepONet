# -*- coding: utf-8 -*-
# @Time    : 2024/12/7 上午11:46
# @Author  : NJU_RanBi
import sys

import matplotlib.pyplot as plt

sys.path.append('..')
import numpy as np
from Tools import To_tensor

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

def get_omega_points(num, alpha):
    x1 = np.random.uniform(-1., 1., num)[:, None]
    x2 = np.random.uniform(-1., 1., num)[:, None]
    x = np.hstack((x1, x2))
    r2 = x1**2 + x2**2
    index = np.where(r2 <= 1)[0]
    x = x[index, :]
    r2 = r2[index, :]
    phi = r2 - alpha**2
    index_p = np.where(phi >= 0)[0]
    index_n = np.where(phi < 0)[0]
    xop = x[index_p, :]
    xon = x[index_n, :]
    return xop, xon

def get_boundary_points(num):
    theta = np.random.uniform(0, 2. * np.pi, num)
    x1 = 1. * np.cos(theta)[:, None]
    x2 = 1. * np.sin(theta)[:, None]
    x = np.hstack((x1, x2))
    return x

def get_interface_points(num, alpha):
    theta = np.random.uniform(0, 2. * np.pi, num)
    x1 = alpha * np.cos(theta)[:, None]
    x2 = alpha * np.sin(theta)[:, None]
    x = np.hstack((x1, x2))
    return x

def add_dimension(x, alpha):
    phi = get_phi(x, alpha)[:, None]
    x_add = np.hstack((x, phi))
    return x_add

def get_f(p, x):
    r2 = x[:, 0]**2 + x[:, 1]**2
    f = 9. * p * np.sqrt(r2)
    return f

def get_phi(x, alpha):
    r2 = x[:, 0]**2 + x[:, 1]**2
    phi = np.abs(r2 - alpha**2)
    return phi

def get_g(x):
    r = np.sqrt(x[:, 0] ** 2 + x[:, 1] ** 2)
    g = r ** 3 / 1000
    return g

def get_normal_vector(x, alpha):
    n = x/alpha
    return n


def get_data(p, alpha):
    sensor_resolution = 10 #传感器的分辨率
    omega_num = 800
    boundary_num = 300 #注意为每条边的采样点数
    interface_num = 200
    sensor_location_p, sensor_location_n = get_sensors_location(sensor_resolution, alpha)
    xop, xon = get_omega_points(omega_num, alpha)
    xb = get_boundary_points(boundary_num)
    xif = get_interface_points(interface_num, alpha)
    xop_data = add_dimension(xop, alpha)
    xon_data = add_dimension(xon, alpha)
    xb_data = add_dimension(xb, alpha)
    xif_data = add_dimension(xif, alpha)

    sensor_f_p = get_f(p, sensor_location_p)
    sensor_f_n = get_f(p, sensor_location_n)
    sensor_f = np.hstack((sensor_f_p, sensor_f_n))
    sensor_phi_p = get_phi(sensor_location_p, alpha)
    sensor_phi_n = -get_phi(sensor_location_n, alpha)
    sensor_phi = np.hstack((sensor_phi_p, sensor_phi_n))

    f_xop = get_f(p, xop)[:, None]
    f_xon = get_f(p, xon)[:, None]
    g_b = get_g(xb)[:, None]
    n = get_normal_vector(xif, alpha)

    data_xop = np.hstack((sensor_f*np.ones((len(f_xop), 60)), sensor_phi*np.ones((len(f_xop), 60)), xop_data, f_xop))
    data_xon = np.hstack((sensor_f*np.ones((len(f_xon), 60)), sensor_phi*np.ones((len(f_xon), 60)), xon_data, f_xon))
    data_xb = np.hstack((sensor_f*np.ones((len(g_b), 60)), sensor_phi*np.ones((len(g_b), 60)), xb_data, g_b))
    data_xif = np.hstack((sensor_f*np.ones((interface_num, 60)), sensor_phi*np.ones((interface_num, 60)), xif_data, n))

    return data_xop, data_xon, data_xb, data_xif

def generate_train_data(sample_num, device):

    alpha = np.random.uniform(0.4, 0.8, sample_num)
    p = np.random.uniform(0.5, 2., sample_num)

    data_xop, data_xon, data_xb, data_xif = get_data(p[0], alpha[0])
    for i in range(1, sample_num):
        data_xop_new, data_xon_new, data_xb_new, data_xif_new = get_data(p[i], alpha[i])
        data_xop = np.vstack((data_xop, data_xop_new))
        data_xon = np.vstack((data_xon, data_xon_new))
        data_xb = np.vstack((data_xb, data_xb_new))
        data_xif = np.vstack((data_xif, data_xif_new))

    train_xop = (data_xop[:, :60], data_xop[:, 60:120], data_xop[:, 120:123], data_xop[:, -1][:, None])
    train_xon = (data_xon[:, :60], data_xon[:, 60:120], data_xon[:, 120:123], data_xon[:, -1][:, None])
    train_xb = (data_xb[:, :60], data_xb[:, 60:120], data_xb[:, 120:123], data_xb[:, -1][:, None])
    train_xif = (data_xif[:, :60], data_xif[:, 60:120], data_xif[:, 120:123], data_xif[:, 123:])
    train_xop = To_tensor(train_xop, device=device)
    train_xon = To_tensor(train_xon, device=device)
    train_xb = To_tensor(train_xb, device=device)
    train_xif = To_tensor(train_xif, device=device)

    return train_xop, train_xon, train_xb, train_xif

def get_u_p(x):
    r = np.sqrt(x[:, 0] ** 2 + x[:, 1] ** 2)
    u_p = r**3 / 1000
    return u_p

def get_u_n(x):
    r = np.sqrt(x[:, 0] ** 2 + x[:, 1] ** 2)
    u_n = r ** 3 + (0.001 - 1.)*0.5**3
    return u_n

def generate_test_data(num, device):
    alpha = 0.5
    p = 1
    sensor_resolution = 10 #传感器的分辨率
    sensor_location_p, sensor_location_n = get_sensors_location(sensor_resolution, alpha)
    xop, xon = get_omega_points(num, alpha)
    x = np.vstack((xop, xon))
    x = add_dimension(x, alpha)
    u_p = get_u_p(xop)[:, None]
    u_n = get_u_n(xon)[:, None]
    u = np.vstack((u_p, u_n))

    sensor_f_p = get_f(p, sensor_location_p)
    sensor_f_n = get_f(p, sensor_location_n)
    sensor_f = np.hstack((sensor_f_p, sensor_f_n))
    sensor_phi_p = get_phi(sensor_location_p, alpha)
    sensor_phi_n = -get_phi(sensor_location_n, alpha)
    sensor_phi = np.hstack((sensor_phi_p, sensor_phi_n))

    test_data = (sensor_f*np.ones((len(x), 60)), sensor_phi*np.ones((len(x), 60)), x, u)
    test_data = To_tensor(test_data, device=device)
    return test_data

