# -*- coding: utf-8 -*-
# @Time    : 2024/12/5 下午3:44
# @Author  : NJU_RanBi
import sys
import numpy as np
import torch
sys.path.append('..')
from Tools import To_tensor
import matplotlib.pyplot as plt


def get_theta(x):
    x1 = x[:, 0]
    x2 = x[:, 1]
    theta = np.arctan2(x2, x1)
    theta = theta[:, None]
    return theta


def level_set_funtion(x, r0, r1, r2, k0):
    theta = get_theta(x)
    x1 = x[:, 0]
    x2 = x[:, 1]
    r = np.sqrt(x1 ** 2 + x2 ** 2)[:, None]
    lf = r - r0 - r1 * np.sin(theta) - r2 * np.sin(k0 * theta)
    return lf


def get_sensors_location(num, r0, r1, r2, k0):
    x1 = np.linspace(-1, 1, num)
    x2 = np.linspace(-1, 1, num)
    X, Y = np.meshgrid(x1, x2)
    X = X.flatten()[:, None]
    Y = Y.flatten()[:, None]
    x = np.hstack((X, Y))
    phi = level_set_funtion(x, r0, r1, r2, k0)
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

def get_omega_points(num, r0, r1, r2, k0):
    x1 = np.random.uniform(-1., 1., num)[:, None]
    x2 = np.random.uniform(-1., 1., num)[:, None]
    x = np.hstack((x1, x2))
    phi = level_set_funtion(x, r0, r1, r2, k0)
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


def get_interface_points(num, r0, r1, r2, k0):
    theta = np.random.uniform(0, 2 * np.pi, num)
    r = r0 + r1 * np.sin(theta) + r2 * np.sin(k0 * theta)
    x1 = r * np.cos(theta)
    x2 = r * np.sin(theta)
    x = np.hstack((x1[:, None], x2[:, None]))
    return x


def add_dimension(x, r0, r1, r2, k0):
    phi = level_set_funtion(x, r0, r1, r2, k0)
    index = phi >= 0.
    index = index.astype(float)
    x_add = np.hstack((x, np.abs(phi * index)))
    return x_add


def get_phi(x, r0, r1, r2, k0):
    phi = level_set_funtion(x, r0, r1, r2, k0)[:, 0]
    return phi


def dsinkx_dx(x, k):
    r = np.sqrt(np.sum(np.power(x, 2), 1))
    r = r[:, None]
    theta = get_theta(x)
    f = -k / r * np.cos(k * theta) * np.sin(theta)
    return f


def dsinkx_dy(x, k):
    r = np.sqrt(np.sum(np.power(x, 2), 1))
    r = r[:, None]
    theta = get_theta(x)
    f = k / r * np.cos(k * theta) * np.cos(theta)
    return f


def dsinkx_dy(x, k):
    r = np.sqrt(np.sum(np.power(x, 2), 1))
    r = r[:, None]
    theta = get_theta(x)
    f = k / r * np.cos(k * theta) * np.cos(theta)
    return f


def d2sinkx_dx2(x, k):
    r = np.sqrt(np.sum(np.power(x, 2), 1))
    r = r[:, None]
    theta = get_theta(x)
    f = k / r / r * np.cos(theta) * np.sin(theta) * np.cos(k * theta) - k * k / r / r * np.sin(theta) * np.sin(
        theta) * np.sin(k * theta) + k / r / r * np.cos(theta) * np.sin(theta) * np.cos(k * theta)
    return f


def d2sinkx_dy2(x, k):
    r = np.sqrt(np.sum(np.power(x, 2), 1))
    r = r[:, None]
    theta = get_theta(x)
    f = -k / r / r * np.cos(theta) * np.sin(theta) * np.cos(k * theta) - k * k / r / r * np.cos(theta) * np.cos(
        theta) * np.sin(k * theta) - k / r / r * np.cos(theta) * np.sin(theta) * np.cos(k * theta)
    return f


def dr_dx(x):
    theta = get_theta(x)
    f = np.cos(theta)
    return f


def dr_dy(x):
    theta = get_theta(x)
    f = np.sin(theta)
    return f


def d2r_dx2(x):
    r = np.sqrt(np.sum(np.power(x, 2), 1))
    r = r[:, None]
    theta = get_theta(x)
    f = np.sin(theta) ** 2 / r
    return f


def d2r_dy2(x):
    r = np.sqrt(np.sum(np.power(x, 2), 1))
    r = r[:, None]
    theta = get_theta(x)
    f = np.cos(theta) ** 2 / r
    return f


def get_dphi_dx(x, r0, r1, r2, k0):
    phix = dr_dx(x) - r1 * dsinkx_dx(x, 1) - r2 * dsinkx_dx(x, k0)
    return phix

def get_dphi_dy(x, r0, r1, r2, k0):
    phiy = dr_dy(x) - r1 * dsinkx_dy(x, 1) - r2 * dsinkx_dy(x, k0)
    return phiy

def get_d2phi_dx2(x, r0, r1, r2, k0):
    phix2 = d2r_dx2(x) - r1 * d2sinkx_dx2(x, 1) - r2 * d2sinkx_dx2(x, k0)
    return phix2

def get_d2phi_dy2(x, r0, r1, r2, k0):
    phiy2 = d2r_dy2(x) - r1 * d2sinkx_dy2(x, 1) - r2 * d2sinkx_dy2(x, k0)
    return phiy2


def get_normal_vector(x, r0, r1, r2, k0):
    n1 = get_dphi_dx(x, r0, r1, r2, k0)
    n2 = get_dphi_dy(x, r0, r1, r2, k0)
    e1 = n1 / np.sqrt(n1 * n1 + n2 * n2)
    e2 = n2 / np.sqrt(n1 * n1 + n2 * n2)
    n = np.hstack((e1, e2))
    return n


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
    h = 1e-4
    x1 = x[:, 0][:, None]
    x2 = x[:, 1][:, None]
    xl = np.hstack((x1 - h, x2))
    xr = np.hstack((x1 + h, x2))
    xt = np.hstack((x1, x2 + h))
    xb = np.hstack((x1, x2 - h))
    f = (up_x(xl, r0, r1, r2, k0) + up_x(xr, r0, r1, r2, k0) + up_x(xt, r0, r1, r2, k0) + up_x(xb, r0, r1, r2, k0) - 4 * up_x(x, r0, r1, r2, k0)) / (h ** 2)
    return f


def get_fn(x, r0, r1, r2, k0):
    h = 1e-4
    x1 = x[:, 0][:, None]
    x2 = x[:, 1][:, None]
    xl = np.hstack((x1 - h, x2))
    xr = np.hstack((x1 + h, x2))
    xt = np.hstack((x1, x2 + h))
    xb = np.hstack((x1, x2 - h))
    f = 1000 * (un_x(xl, r0, r1, r2, k0) + un_x(xr, r0, r1, r2, k0) + un_x(xt, r0, r1, r2, k0) + un_x(xb, r0, r1, r2, k0) - 4 * un_x(x, r0, r1, r2, k0)) / (h ** 2)
    return f


def get_psi(x, n, r0, r1, r2, k0):
    h = 1e-6
    xn = x + h * n
    psi = (up_x(xn, r0, r1, r2, k0) - up_x(x, r0, r1, r2, k0)) / h - (un_x(xn, r0, r1, r2, k0) - un_x(x, r0, r1, r2, k0)) / h
    return psi


def get_data(r0, r1, r2, k0):
    sensor_resolution = 10 #传感器的分辨率
    omega_num = 400
    boundary_num = 50 #注意为每条边的采样点数
    interface_num = 100
    sensor_location_p, sensor_location_n = get_sensors_location(sensor_resolution, r0, r1, r2, k0)
    sensor_boundary_location = get_boudary_sensors_location(sensor_resolution)
    xop, xon = get_omega_points(omega_num, r0, r1, r2, k0)
    xb = get_boundary_points(boundary_num)
    xif = get_interface_points(interface_num, r0, r1, r2, k0)
    xop_data = add_dimension(xop, r0, r1, r2, k0)
    xon_data = add_dimension(xon, r0, r1, r2, k0)
    xb_data = add_dimension(xb, r0, r1, r2, k0)
    xif_data = add_dimension(xif, r0, r1, r2, k0)

    sensor_f_p = get_fp(sensor_location_p, r0, r1, r2, k0)[:, 0]
    sensor_f_n = get_fn(sensor_location_n, r0, r1, r2, k0)[:, 0]
    sensor_f = np.hstack((sensor_f_p, sensor_f_n))

    sensor_g = up_x(sensor_boundary_location, r0, r1, r2, k0)[:, 0]

    sensor_phi_p = get_phi(sensor_location_p, r0, r1, r2, k0)
    sensor_phi_n = get_phi(sensor_location_n, r0, r1, r2, k0)
    sensor_phi = np.hstack((sensor_phi_p, sensor_phi_n))

    f_xop = get_fp(xop, r0, r1, r2, k0)
    f_xon = get_fn(xon, r0, r1, r2, k0) / 1000
    g_b = up_x(xb, r0, r1, r2, k0)
    n = get_normal_vector(xif, r0, r1, r2, k0)
    psi = get_psi(xif, n, r0, r1, r2, k0)
    dphip_dx = get_dphi_dx(xop, r0, r1, r2, k0)
    dphip_dy = get_dphi_dy(xop, r0, r1, r2, k0)
    dphi2p_dx2 = get_d2phi_dx2(xop, r0, r1, r2, k0)
    dphi2p_dy2 = get_d2phi_dy2(xop, r0, r1, r2, k0)

    dphi_if_dx = get_dphi_dx(xif, r0, r1, r2, k0)
    dphi_if_dy = get_dphi_dy(xif, r0, r1, r2, k0)
    data_xop = np.hstack((sensor_phi*np.ones((len(xop), sensor_resolution**2)), sensor_f*np.ones((len(xop), sensor_resolution**2)), sensor_g*np.ones((len(xop), 4*(sensor_resolution - 1))), xop_data, dphip_dx, dphip_dy, dphi2p_dx2, dphi2p_dy2, f_xop))
    data_xon = np.hstack((sensor_phi*np.ones((len(xon), sensor_resolution**2)), sensor_f*np.ones((len(xon), sensor_resolution**2)), sensor_g*np.ones((len(xon), 4*(sensor_resolution - 1))), xon_data, f_xon))
    data_xb = np.hstack((sensor_phi*np.ones((len(xb), sensor_resolution**2)), sensor_f*np.ones((len(xb), sensor_resolution**2)), sensor_g*np.ones((len(xb), 4*(sensor_resolution - 1))), xb_data, g_b))
    data_xif = np.hstack((sensor_phi*np.ones((interface_num, sensor_resolution**2)), sensor_f*np.ones((len(xif), sensor_resolution**2)), sensor_g*np.ones((len(xif), 4*(sensor_resolution - 1))), xif_data, dphi_if_dx, dphi_if_dy, psi, n))

    return data_xop, data_xon, data_xb, data_xif


def generate_train_data(sample_num, device):
    r0 = np.random.uniform(0.5, 0.7, sample_num)
    r1 = 0.
    r2 = np.random.uniform(-0.15, 0.15, sample_num)
    k0 = 5

    data_xop, data_xon, data_xb, data_xif = get_data(r0[0], r1, r2[0], k0)
    for i in range(1, sample_num):
        data_xop_new, data_xon_new, data_xb_new, data_xif_new = get_data(r0[i], r1, r2[i], k0)
        data_xop = np.vstack((data_xop, data_xop_new))
        data_xon = np.vstack((data_xon, data_xon_new))
        data_xb = np.vstack((data_xb, data_xb_new))
        data_xif = np.vstack((data_xif, data_xif_new))

    train_xop = (data_xop[:, :100], data_xop[:, 100:200], data_xop[:, 200:236], data_xop[:, 236:239], data_xop[:, -5][:, None], data_xop[:, -4][:, None], data_xop[:, -3][:, None], data_xop[:, -2][:, None], data_xop[:, -1][:, None])
    train_xon = (data_xon[:, :100], data_xon[:, 100:200], data_xon[:, 200:236], data_xon[:, 236:239], data_xon[:, -1][:, None])
    train_xb = (data_xb[:, :100], data_xb[:, 100:200], data_xb[:, 200:236], data_xb[:, 236:239], data_xb[:, -1][:, None])
    train_xif = (data_xif[:, :100], data_xif[:, 100:200], data_xif[:, 200:236], data_xif[:, 236:239], data_xif[:, -5][:, None], data_xif[:, -4][:, None], data_xif[:, -3][:, None], data_xif[:, -2:])
    train_xop = To_tensor(train_xop, device=device)
    train_xon = To_tensor(train_xon, device=device)
    train_xb = To_tensor(train_xb, device=device)
    train_xif = To_tensor(train_xif, device=device)

    return train_xop, train_xon, train_xb, train_xif


def generate_test_data(num, device):
    r0 = 0.6
    r1 = 0.
    r2 = 0.1
    k0 = 5
    sensor_resolution = 10  # 传感器的分辨率
    sensor_location_p, sensor_location_n = get_sensors_location(sensor_resolution, r0, r1, r2, k0)
    sensor_boundary_location = get_boudary_sensors_location(sensor_resolution)
    xop, xon = get_omega_points(num, r0, r1, r2, k0)
    x = np.vstack((xop, xon))
    x = add_dimension(x, r0, r1, r2, k0)
    u_p = up_x(xop, r0, r1, r2, k0)
    u_n = un_x(xon, r0, r1, r2, k0)
    u = np.vstack((u_p, u_n))

    sensor_f_p = get_fp(sensor_location_p, r0, r1, r2, k0)[:, 0]
    sensor_f_n = get_fn(sensor_location_n, r0, r1, r2, k0)[:, 0]
    sensor_f = np.hstack((sensor_f_p, sensor_f_n))

    sensor_g = up_x(sensor_boundary_location, r0, r1, r2, k0)[:, 0]

    sensor_phi_p = get_phi(sensor_location_p, r0, r1, r2, k0)
    sensor_phi_n = get_phi(sensor_location_n, r0, r1, r2, k0)
    sensor_phi = np.hstack((sensor_phi_p, sensor_phi_n))

    test_data = (sensor_phi * np.ones((num, sensor_resolution ** 2)), sensor_f * np.ones((num, sensor_resolution ** 2)),
                 sensor_g * np.ones((num, 4 * (sensor_resolution - 1))), x, u)
    test_data = To_tensor(test_data, device=device)
    return test_data