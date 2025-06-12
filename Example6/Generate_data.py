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
    x3 = np.linspace(-1, 1, num)
    x4 = np.linspace(-1, 1, num)
    x5 = np.linspace(-1, 1, num)
    x6 = np.linspace(-1, 1, num)
    X1, X2, X3, X4, X5, X6 = np.meshgrid(x1, x2, x3, x4, x5, x6)
    X1 = X1.flatten()[:, None]
    X2 = X2.flatten()[:, None]
    X3 = X3.flatten()[:, None]
    X4 = X4.flatten()[:, None]
    X5 = X5.flatten()[:, None]
    X6 = X6.flatten()[:, None]
    x = np.hstack((X1, X2, X3, X4, X5, X6))
    x1 = x[:, 0]
    x2 = x[:, 1]
    x3 = x[:, 2]
    x4 = x[:, 3]
    x5 = x[:, 4]
    x6 = x[:, 5]
    r2 = x1 ** 2 + x2 ** 2 + x3 ** 2 + x4 ** 2 + x5 ** 2 + x6 ** 2
    index = np.where(r2 <= 0.36)[0]
    x = x[index, :]
    r2 = r2[index]
    phi = r2 - alpha ** 2
    index_p = np.where(phi >= 0)[0]
    index_n = np.where(phi < 0)[0]
    xp = x[index_p, :]
    xn = x[index_n, :]
    return xp, xn

def get_omega_points(num, alpha):
    x1 = np.random.uniform(-0.6, 0.6, num)[:, None]
    x2 = np.random.uniform(-0.6, 0.6, num)[:, None]
    x3 = np.random.uniform(-0.6, 0.6, num)[:, None]
    x4 = np.random.uniform(-0.6, 0.6, num)[:, None]
    x5 = np.random.uniform(-0.6, 0.6, num)[:, None]
    x6 = np.random.uniform(-0.6, 0.6, num)[:, None]
    x = np.hstack((x1, x2, x3, x4, x5, x6))
    r2 = x1**2 + x2**2 + x3**2 + x4**2 + x5**2 + x6**2
    index = np.where(r2 <= 0.36)[0]
    x = x[index, :]
    r2 = r2[index, :]
    phi = r2 - alpha**2
    index_p = np.where(phi >= 0)[0]
    index_n = np.where(phi < 0)[0]
    xop = x[index_p, :]
    xon = x[index_n, :]
    return xop, xon

def get_boundary_points(num):
    x1 = np.random.uniform(-1, 1, num)[:, None]
    x2 = np.random.uniform(-1, 1, num)[:, None]
    x3 = np.random.uniform(-1, 1, num)[:, None]
    x4 = np.random.uniform(-1, 1, num)[:, None]
    x5 = np.random.uniform(-1, 1, num)[:, None]
    x6 = np.random.uniform(-1, 1, num)[:, None]
    x = np.hstack((x1, x2, x3, x4, x5, x6))
    r = np.sqrt(x1 ** 2 + x2 ** 2 + x3 ** 2 + x4 ** 2 + x5 ** 2 + x6 ** 2)
    xb = 0.6 * x / r
    return xb

def get_interface_points(num, alpha):
    x1 = np.random.uniform(-1, 1, num)[:, None]
    x2 = np.random.uniform(-1, 1, num)[:, None]
    x3 = np.random.uniform(-1, 1, num)[:, None]
    x4 = np.random.uniform(-1, 1, num)[:, None]
    x5 = np.random.uniform(-1, 1, num)[:, None]
    x6 = np.random.uniform(-1, 1, num)[:, None]
    x = np.hstack((x1, x2, x3, x4, x5, x6))
    r = np.sqrt(x1 ** 2 + x2 ** 2 + x3 ** 2 + x4 ** 2 + x5 ** 2 + x6 ** 2)
    xif = alpha * x / r
    return xif

def add_dimension(x, alpha):
    phi = get_phi(x, alpha)[:, None]
    x_add = np.hstack((x, phi))
    return x_add

def get_f_p(x, alpha):
    r = np.sqrt(np.sum(np.power(x, 2), 1))
    f = -12 * np.exp(alpha**2 - r ** 2) + 4 * np.power(r, 2) * np.exp(alpha**2 - r ** 2) - \
         np.sin(x[:, 0]) - np.sin(x[:, 1]) - np.sin(x[:, 2]) - np.sin(x[:, 3]) - np.sin(x[:, 4])
    return f

def get_f_n(x, alpha):
    r = np.sqrt(np.sum(np.power(x, 2), 1))
    f = -24 * np.cos(alpha**2 - r ** 2) - 8 * np.power(r, 2) * np.sin(alpha**2 - r ** 2) - \
         np.sin(x[:, 0]) - np.sin(x[:, 1]) - np.sin(x[:, 2]) - np.sin(x[:, 3]) - np.sin(x[:, 4])
    return f

def get_phi(x, alpha):
    r2 = x[:, 0]**2 + x[:, 1]**2 + x[:, 2]**2 + x[:, 3]**2 + x[:, 4]**2 + x[:, 5]**2
    phi = np.abs(r2 - alpha**2)
    return phi

def get_g(x, alpha):
    r = np.sqrt(np.sum(np.power(x, 2), 1))
    g = np.exp(alpha**2 - np.power(r, 2)) + np.sin(x[:, 0]) + np.sin(x[:, 1]) + np.sin(x[:, 2]) + np.sin(x[:, 3]) + np.sin(x[:, 4])
    return g

def get_normal_vector(x, alpha):
    n = x/alpha
    return n

def get_data(alpha):
    sensor_resolution = 7 #传感器的分辨率
    omega_num = 10000
    boundary_num = 400
    interface_num = 300
    sensor_location_p, sensor_location_n = get_sensors_location(sensor_resolution, alpha)
    xop, xon = get_omega_points(omega_num, alpha)
    xb = get_boundary_points(boundary_num)
    xif = get_interface_points(interface_num, alpha)
    xop_data = add_dimension(xop, alpha)
    xon_data = add_dimension(xon, alpha)
    xb_data = add_dimension(xb, alpha)
    xif_data = add_dimension(xif, alpha)

    sensor_f_p = get_f_p(sensor_location_p, alpha)
    sensor_f_n = get_f_n(sensor_location_n, alpha)
    sensor_f = np.hstack((sensor_f_p, sensor_f_n))
    sensor_phi_p = get_phi(sensor_location_p, alpha)
    sensor_phi_n = -get_phi(sensor_location_n, alpha)
    sensor_phi = np.hstack((sensor_phi_p, sensor_phi_n))

    f_xop = get_f_p(xop, alpha)[:, None]
    f_xon = get_f_n(xon, alpha)[:, None]
    g_b = get_g(xb, alpha)[:, None]
    n = get_normal_vector(xif, alpha)
    r0 = alpha * np.ones((len(xif), 1))
    data_xop = np.hstack((sensor_f*np.ones((len(f_xop), len(sensor_f))), sensor_phi*np.ones((len(f_xop), len(sensor_phi))), xop_data, f_xop))
    data_xon = np.hstack((sensor_f*np.ones((len(f_xon), len(sensor_f))), sensor_phi*np.ones((len(f_xon), len(sensor_phi))), xon_data, f_xon))
    data_xb = np.hstack((sensor_f*np.ones((len(g_b), len(sensor_f))), sensor_phi*np.ones((len(g_b), len(sensor_phi))), xb_data, g_b))
    data_xif = np.hstack((sensor_f*np.ones((interface_num, len(sensor_f))), sensor_phi*np.ones((interface_num, len(sensor_phi))), xif_data, r0, n))

    return data_xop, data_xon, data_xb, data_xif


def generate_train_data(sample_num, device):

    alpha = np.random.uniform(0.4, 0.5, sample_num)

    data_xop, data_xon, data_xb, data_xif = get_data(alpha[0])
    for i in range(1, sample_num):
        data_xop_new, data_xon_new, data_xb_new, data_xif_new = get_data(alpha[i])
        data_xop = np.vstack((data_xop, data_xop_new))
        data_xon = np.vstack((data_xon, data_xon_new))
        data_xb = np.vstack((data_xb, data_xb_new))
        data_xif = np.vstack((data_xif, data_xif_new))

    train_xop = (data_xop[:, :233], data_xop[:, 233:466], data_xop[:, 466:473], data_xop[:, -1][:, None])
    train_xon = (data_xon[:, :233], data_xon[:, 233:466], data_xon[:, 466:473], data_xon[:, -1][:, None])
    train_xb = (data_xb[:, :233], data_xb[:, 233:466], data_xb[:, 466:473], data_xb[:, -1][:, None])
    train_xif = (data_xif[:, :233], data_xif[:, 233:466], data_xif[:, 466:473], data_xif[:, 473][:, None], data_xif[:, -6:])
    train_xop = To_tensor(train_xop, device=device)
    train_xon = To_tensor(train_xon, device=device)
    train_xb = To_tensor(train_xb, device=device)
    train_xif = To_tensor(train_xif, device=device)

    return train_xop, train_xon, train_xb, train_xif

def get_u_p(x):
    r = np.sqrt(np.sum(np.power(x, 2), 1))
    u_p = np.exp(0.25 - np.power(r,2)) + np.sin(x[:,0]) + np.sin(x[:,1]) + np.sin(x[:,2]) + np.sin(x[:,3]) + np.sin(x[:,4])
    return u_p

def get_u_n(x):
    r = np.sqrt(np.sum(np.power(x, 2), 1))
    u_n = 1 + 2*np.sin(0.25 - np.power(r, 2)) + np.sin(x[:,0]) + np.sin(x[:,1]) + np.sin(x[:,2]) + np.sin(x[:,3]) + np.sin(x[:,4])
    return u_n

def generate_test_data(num, device):
    alpha = 0.5
    sensor_resolution = 7
    sensor_location_p, sensor_location_n = get_sensors_location(sensor_resolution, alpha)
    xop, xon = get_omega_points(num, alpha)
    x = np.vstack((xop, xon))
    x = add_dimension(x, alpha)
    u_p = get_u_p(xop)[:, None]
    u_n = get_u_n(xon)[:, None]
    u = np.vstack((u_p, u_n))

    sensor_f_p = get_f_p(sensor_location_p, alpha)
    sensor_f_n = get_f_n(sensor_location_n, alpha)
    sensor_f = np.hstack((sensor_f_p, sensor_f_n))
    sensor_phi_p = get_phi(sensor_location_p, alpha)
    sensor_phi_n = -get_phi(sensor_location_n, alpha)
    sensor_phi = np.hstack((sensor_phi_p, sensor_phi_n))

    test_data = (sensor_f*np.ones((len(x), len(sensor_f))), sensor_phi*np.ones((len(x), len(sensor_phi))), x, u)
    test_data = To_tensor(test_data, device=device)
    return test_data

