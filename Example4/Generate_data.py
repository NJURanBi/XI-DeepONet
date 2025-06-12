# -*- coding: utf-8 -*-
# @Time    : 2024/12/6 下午12:58
# @Author  : NJU_RanBi
import sys
import numpy as np
import torch
sys.path.append('..')
import matplotlib.pyplot as plt
from Tools import To_tensor


def Circle_x(t):
    x = 0.5 * t - 0.25
    return x


def Circle_y(t):
    y = 0.5 * t - 0.25
    return y


def Circle_r(t):
    r = 0.1 * t + 0.4
    return r


def level_set_funtion(x, t):
    x1 = x[:, 0][:, None]
    x2 = x[:, 1][:, None]
    lf = (x1 - Circle_x(t)) ** 2 + (x2 - Circle_y(t)) ** 2 - Circle_r(t) ** 2
    return lf


def get_sensors_location(num, t):
    x1 = np.linspace(-1, 1, num)
    x2 = np.linspace(-1, 1, num)
    X, Y = np.meshgrid(x1, x2)
    X = X.flatten()[:, None]
    Y = Y.flatten()[:, None]
    x = np.hstack((X, Y))
    phi = level_set_funtion(x, t)
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


def get_omega_points(num, t):
    x1 = np.random.uniform(-1., 1., num)[:, None]
    x2 = np.random.uniform(-1., 1., num)[:, None]
    x = np.hstack((x1, x2))
    phi = level_set_funtion(x, t)
    index_p = np.where(phi >= 0)[0]
    index_n = np.where(phi < 0)[0]
    xop = x[index_p, :]
    xon = x[index_n, :]
    return xop, xon


def get_boundary_points(num):
    xb1 = np.random.uniform(-1., 1., num)[:, None]
    xb2 = np.random.uniform(-1., 1., num)[:, None]
    xb3 = np.random.uniform(-1., 1., num)[:, None]
    xb4 = np.random.uniform(-1., 1., num)[:, None]
    xb1 = np.hstack((xb1, np.ones_like(xb1)))
    xb2 = np.hstack((xb2, -np.ones_like(xb2)))
    xb3 = np.hstack((np.ones_like(xb3), xb3))
    xb4 = np.hstack((-np.ones_like(xb4), xb4))
    xb = np.vstack((xb1, xb2, xb3, xb4))
    return xb


def get_interface_points(num, t):
    theta = np.random.uniform(0, 2 * np.pi, num)
    x1 = Circle_x(t) + Circle_r(t) * np.cos(theta)
    x2 = Circle_y(t) + Circle_r(t) * np.sin(theta)
    x = np.hstack((x1[:, None], x2[:, None]))
    return x

def mark(x, t):
    lf = level_set_funtion(x, t)
    m = lf[:, 0]
    return m

def add_dimension(x, t):
    m = mark(x, t)
    index1 = m >= 0
    index2 = m < 0
    index1 = index1.astype(float)
    index2 = index2.astype(float)
    index1 = index1[:, None]
    index2 = index2[:, None]
    lf = level_set_funtion(x, t) * (index1 - index2)
    x_3 = np.hstack((x, lf))
    return x_3


def get_phi(x, t):
    phi = np.abs(level_set_funtion(x, t))[:, 0]
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


def get_normal_vector(x, t):
    c_x = Circle_x(t)
    c_y = Circle_y(t)
    x1 = x[:, 0][:, None]
    x2 = x[:, 1][:, None]
    n1 = 2 * (x1 - c_x)
    n2 = 2 * (x2 - c_y)
    e1 = n1 / np.sqrt(n1 * n1 + n2 * n2)
    e2 = n2 / np.sqrt(n1 * n1 + n2 * n2)
    n = np.hstack((e1, e2))
    return n


def get_dphi_dx(x, t):
    c_x = Circle_x(t)
    x1 = x[:, 0][:, None]
    phix = 2 * (x1 - c_x)
    return phix


def get_dphi_dy(x, t):
    c_y = Circle_y(t)
    x2 = x[:, 1][:, None]
    phiy = 2 * (x2 - c_y)
    return phiy


def get_data(t, beta_p, beta_n):
    sensor_resolution = 10 #传感器的分辨率
    omega_num = 400
    boundary_num = 50 #注意为每条边的采样点数
    interface_num = 100
    sensor_location_p, sensor_location_n = get_sensors_location(sensor_resolution, t)
    sensor_boundary_location = get_boudary_sensors_location(sensor_resolution, t)
    xop, xon = get_omega_points(omega_num, t)
    xb = get_boundary_points(boundary_num)
    xif = get_interface_points(interface_num, t)
    xop_data = add_dimension(xop, t)
    xon_data = add_dimension(xon, t)
    xb_data = add_dimension(xb, t)
    xif_data = add_dimension(xif, t)

    sensor_f_p = get_f(sensor_location_p, t)
    sensor_f_n = get_f(sensor_location_n, t)
    sensor_f = np.hstack((sensor_f_p, sensor_f_n))

    sensor_g = get_g(sensor_boundary_location, t)

    sensor_phi_p = get_phi(sensor_location_p, t)
    sensor_phi_n = -get_phi(sensor_location_n, t)
    sensor_phi = np.hstack((sensor_phi_p, sensor_phi_n))

    f_xop = get_f(xop, t)[:, None] / beta_p
    f_xon = get_f(xon, t)[:, None] / beta_n
    g_b = get_g(xb, t)[:, None]
    n = get_normal_vector(xif, t)
    dphip_dx = get_dphi_dx(xop, t)
    dphip_dy = get_dphi_dy(xop, t)
    dphin_dx = get_dphi_dx(xon, t)
    dphin_dy = get_dphi_dy(xon, t)
    dphi_if_dx = get_dphi_dx(xif, t)
    dphi_if_dy = get_dphi_dy(xif, t)

    data_xop = np.hstack((sensor_phi*np.ones((len(xop), sensor_resolution**2)), sensor_f*np.ones((len(xop), sensor_resolution**2)), sensor_g*np.ones((len(xop), 4*(sensor_resolution - 1))), xop_data, dphip_dx, dphip_dy, f_xop))
    data_xon = np.hstack((sensor_phi*np.ones((len(xon), sensor_resolution**2)), sensor_f*np.ones((len(xon), sensor_resolution**2)), sensor_g*np.ones((len(xon), 4*(sensor_resolution - 1))), xon_data, dphin_dx, dphin_dy, f_xon))
    data_xb = np.hstack((sensor_phi*np.ones((len(xb), sensor_resolution**2)), sensor_f*np.ones((len(xb), sensor_resolution**2)), sensor_g*np.ones((len(xb), 4*(sensor_resolution - 1))), xb_data, g_b))
    data_xif = np.hstack((sensor_phi*np.ones((len(xif), sensor_resolution**2)), sensor_f*np.ones((len(xif), sensor_resolution**2)), sensor_g*np.ones((len(xif), 4*(sensor_resolution - 1))), xif_data, dphi_if_dx, dphi_if_dy, n))

    return data_xop, data_xon, data_xb, data_xif


def generate_train_data(sample_num, device):
    t = np.linspace(0, 1, sample_num)
    beta_p = 10.
    beta_n = 1.

    data_xop, data_xon, data_xb, data_xif = get_data(t[0], beta_p, beta_n)
    for i in range(1, sample_num):
        data_xop_new, data_xon_new, data_xb_new, data_xif_new = get_data(t[i], beta_p, beta_n)
        data_xop = np.vstack((data_xop, data_xop_new))
        data_xon = np.vstack((data_xon, data_xon_new))
        data_xb = np.vstack((data_xb, data_xb_new))
        data_xif = np.vstack((data_xif, data_xif_new))

    train_xop = (data_xop[:, :100], data_xop[:, 100:200], data_xop[:, 200:236], data_xop[:, 236:239], data_xop[:, -3][:, None], data_xop[:, -2][:, None], data_xop[:, -1][:, None])
    train_xon = (data_xon[:, :100], data_xon[:, 100:200], data_xon[:, 200:236], data_xon[:, 236:239], data_xon[:, -3][:, None], data_xon[:, -2][:, None], data_xon[:, -1][:, None])
    train_xb = (data_xb[:, :100], data_xb[:, 100:200], data_xb[:, 200:236], data_xb[:, 236:239], data_xb[:, -1][:, None])
    train_xif = (data_xif[:, :100], data_xif[:, 100:200], data_xif[:, 200:236], data_xif[:, 236:239], data_xif[:, -4][:, None], data_xif[:, -3][:, None], data_xif[:, -2:])
    train_xop = To_tensor(train_xop, device=device)
    train_xon = To_tensor(train_xon, device=device)
    train_xb = To_tensor(train_xb, device=device)
    train_xif = To_tensor(train_xif, device=device)

    return train_xop, train_xon, train_xb, train_xif


def get_u_p(x, t):
    c_x = Circle_x(t)
    c_y = Circle_y(t)
    c_r = Circle_r(t)
    x1 = x[:, 0]
    x2 = x[:, 1]
    u_p = np.exp(-(x1 - c_x)**2 - (x2 - c_y)**2) / 10 + (1 - 0.1) * np.exp(-c_r ** 2)
    return u_p

def get_u_n(x, t):
    c_x = Circle_x(t)
    c_y = Circle_y(t)
    x1 = x[:, 0]
    x2 = x[:, 1]
    u_n = np.exp(-(x1 - c_x)**2 - (x2 - c_y)**2)
    return u_n

def generate_test_data(num, device):
    t = 0.5
    beta_p = 10.
    beta_n = 1.
    sensor_resolution = 10 #传感器的分辨率
    sensor_location_p, sensor_location_n = get_sensors_location(sensor_resolution, t)
    sensor_boundary_location = get_boudary_sensors_location(sensor_resolution, t)
    xop, xon = get_omega_points(num, t)
    x = np.vstack((xop, xon))
    x = add_dimension(x, t)
    u_p = get_u_p(xop, t)[:, None]
    u_n = get_u_n(xon, t)[:, None]
    u = np.vstack((u_p, u_n))

    sensor_f_p = get_f(sensor_location_p, t)
    sensor_f_n = get_f(sensor_location_n, t)
    sensor_f = np.hstack((sensor_f_p, sensor_f_n))

    sensor_g = get_g(sensor_boundary_location, t)

    sensor_phi_p = get_phi(sensor_location_p, t)
    sensor_phi_n = -get_phi(sensor_location_n, t)
    sensor_phi = np.hstack((sensor_phi_p, sensor_phi_n))

    test_data = (sensor_phi*np.ones((num, sensor_resolution**2)), sensor_f*np.ones((num, sensor_resolution**2)), sensor_g*np.ones((num, 4*(sensor_resolution - 1))), x, u)
    test_data = To_tensor(test_data, device=device)
    return test_data



