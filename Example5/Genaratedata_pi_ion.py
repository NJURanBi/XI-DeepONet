import sys
sys.path.append('..')
import numpy as np
from Tools import To_tensor
import matplotlib.pyplot as plt
import torch
import torch.nn as nn

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from torch import optim, autograd

def level_set_funtion(x, r0, t1, t2, t3, n1, n2, n3, theta1, theta2, theta3):
    x1 = x[:, 0]
    x2 = x[:, 1]
    x3 = x[:, 2]
    theta = np.arctan2(x2, x1)
    A1 = t1 * np.cos(n1 * (theta - theta1))
    A2 = t2 * np.cos(n2 * (theta - theta2))
    A3 = t3 * np.cos(n3 * (theta - theta3))
    A = (x1**2 + x2**2) / (x1**2 + x2**2 + x3**2)
    lf = np.sqrt(x1**2 + x2**2 + x3**2) - r0 * (1. + A**2 * (A1 + A2 + A3))
    return lf

def get_sensors_location(num, r0, t1, t2, t3, n1, n2, n3, theta1, theta2, theta3):
    x1 = np.linspace(-1, 1, num)
    x2 = np.linspace(-1, 1, num)
    x3 = np.linspace(-1, 1, num)
    X, Y, Z = np.meshgrid(x1, x2, x3)
    X = X.flatten()[:, None]
    Y = Y.flatten()[:, None]
    Z = Z.flatten()[:, None]
    x = np.hstack((X, Y, Z))
    r = np.sqrt(x[:, 0]**2 + x[:, 1]**2 + x[:, 2]**2)
    index = np.where((r>0.151)&(r<0.911))[0]
    x = x[index, :]
    lf = level_set_funtion(x, r0, t1, t2, t3, n1, n2, n3, theta1, theta2, theta3)
    index_p = np.where(lf >= 0)[0]
    index_n = np.where(lf < 0)[0]
    xp = x[index_p, :]
    xn = x[index_n, :]
    return xp, xn

def get_omega_points(num, r0, t1, t2, t3, n1, n2, n3, theta1, theta2, theta3):
    x1 = np.random.uniform(-1., 1., num)[:, None]
    x2 = np.random.uniform(-1., 1., num)[:, None]
    x3 = np.random.uniform(-1., 1., num)[:, None]
    x = np.hstack((x1, x2, x3))
    r = np.sqrt(x[:, 0] ** 2 + x[:, 1] ** 2 + x[:, 2] ** 2)
    index = np.where((r > 0.151) & (r < 0.911))[0]
    x = x[index, :]
    lf = level_set_funtion(x, r0, t1, t2, t3, n1, n2, n3, theta1, theta2, theta3)
    index_p = np.where(lf >= 0)[0]
    index_n = np.where(lf < 0)[0]
    xop = x[index_p, :]
    xon = x[index_n, :]
    return xop, xon

def get_boundary_in_points(num, radiu=0.151):
    x1 = np.random.uniform(-1., 1., num)[:, None]
    x2 = np.random.uniform(-1., 1., num)[:, None]
    x3 = np.random.uniform(-1., 1., num)[:, None]
    r = np.sqrt(x1 ** 2 + x2 ** 2 + x3 ** 2)
    x1 = radiu * (x1 / r)
    x2 = radiu * (x2 / r)
    x3 = radiu * (x3 / r)
    xb_in = np.hstack((x1, x2, x3))
    return xb_in

def get_boundary_out_points(num, radiu=0.911):
    x1 = np.random.uniform(-1., 1., num)[:, None]
    x2 = np.random.uniform(-1., 1., num)[:, None]
    x3 = np.random.uniform(-1., 1., num)[:, None]
    r = np.sqrt(x1 ** 2 + x2 ** 2 + x3 ** 2)
    x1 = radiu * (x1 / r)
    x2 = radiu * (x2 / r)
    x3 = radiu * (x3 / r)
    xb_out = np.hstack((x1, x2, x3))
    return xb_out

def get_interface_points(num, r0, t1, t2, t3, n1, n2, n3, theta1, theta2, theta3):
    x1 = np.random.uniform(-1., 1., num)[:, None]
    x2 = np.random.uniform(-1., 1., num)[:, None]
    x3 = np.random.uniform(-1., 1., num)[:, None]
    radiu = np.sqrt(x1 ** 2 + x2 ** 2 + x3 ** 2)
    phi = np.arccos(x3 / radiu)[:, 0]
    theta = np.random.uniform(-np.pi, np.pi, num)
    A1 = t1*np.cos(n1*(theta - theta1))
    A2 = t2*np.cos(n2*(theta - theta2))
    A3 = t3*np.cos(n3*(theta - theta3))
    r = r0 * (1. + np.sin(phi)**4 * (A1 + A2 + A3))
    x1 = r * np.sin(phi) * np.cos(theta)
    x2 = r * np.sin(phi) * np.sin(theta)
    x3 = r * np.cos(phi)
    x1 = x1[:, None]
    x2 = x2[:, None]
    x3 = x3[:, None]
    x = np.hstack((x1, x2, x3))
    return x

def add_dimension(x, r0, t1, t2, t3, n1, n2, n3, theta1, theta2, theta3):
    lf = level_set_funtion(x, r0, t1, t2, t3, n1, n2, n3, theta1, theta2, theta3)[:, None]
    index = lf >= 0.
    index = index.astype(float)
    x_add = np.hstack((x, np.abs(lf * index)))
    return x_add

def get_normal_vector(data, r0, t1, t2, t3, n1, n2, n3, theta1, theta2, theta3):
    x = data[:, 0]
    x = x[:, None]
    y = data[:, 1]
    y = y[:, None]
    z = data[:, 2]
    z = z[:, None]
    dsq0 = x * x + y * y
    zsq = z * z
    dsq = dsq0 + zsq
    thet = np.arctan2(y, x)
    gcf = 4 * dsq0 / (dsq ** 3)
    bn0 = t1 * n1
    bn1 = t2 * n2
    bn2 = t3 * n3
    cth0 = np.cos(n1 * (thet - theta1))
    cth1 = np.cos(n2 * (thet - theta2))
    cth2 = np.cos(n3 * (thet - theta3))
    sth0 = np.sin(n1 * (thet - theta1))
    sth1 = np.sin(n2 * (thet - theta2))
    sth2 = np.sin(n3 * (thet - theta3))
    tgrad = np.hstack((-y, x, np.zeros_like(z))) / dsq0
    # f(p) related
    fp = np.sqrt(dsq)
    fgrad = np.hstack((x, y, z)) / fp
    # g(p)
    gp = (dsq0 / dsq) ** 2
    ggrad = gcf * np.hstack((x * zsq, y * zsq, -dsq0 * z))
    # h(p)
    hp = t1 * cth0 + t2 * cth1 + t3 * cth2
    hgrad = - (bn0 * sth0 + bn1 * sth1 + bn2 * sth2) * tgrad
    # phi related
    phigrad = fgrad - r0 * (gp * hgrad + ggrad * hp)
    # output
    dqsqrt = np.sqrt(np.sum(phigrad ** 2, axis=1).reshape((len(x), 1)))
    norvec = phigrad / dqsqrt
    return norvec

def get_grad_phi(data, r0, t1, t2, t3, n1, n2, n3, theta1, theta2, theta3):
    x = data[:, 0]
    x = x[:, None]
    y = data[:, 1]
    y = y[:, None]
    z = data[:, 2]
    z = z[:, None]
    dsq0 = x * x + y * y
    zsq = z * z
    dsq = dsq0 + zsq
    thet = np.arctan2(y, x)
    gcf = 4 * dsq0 / (dsq ** 3)
    bn0 = t1 * n1
    bn1 = t2 * n2
    bn2 = t3 * n3
    cth0 = np.cos(n1 * (thet - theta1))
    cth1 = np.cos(n2 * (thet - theta2))
    cth2 = np.cos(n3 * (thet - theta3))
    sth0 = np.sin(n1 * (thet - theta1))
    sth1 = np.sin(n2 * (thet - theta2))
    sth2 = np.sin(n3 * (thet - theta3))
    tgrad = np.hstack( ( -y , x , np.zeros_like(z) ) )/dsq0
    # f(p) related
    fp = np.sqrt(dsq)
    fgrad = np.hstack( ( x , y , z ) )/fp
    flap  = 2./fp
    # g(p)
    gp = (dsq0/dsq)**2
    ggrad = gcf*np.hstack( ( x*zsq , y*zsq , -dsq0*z ) )
    glap  = gcf*( 4*zsq - dsq0 )
    # h(p)
    hp = t1*cth0 + t2*cth1 + t3*cth2
    hgrad = - ( bn0*sth0 + bn1*sth1 + bn2*sth2 ) * tgrad
    hlap  = - ( bn0*n1*cth0 + bn1*n2*cth1 + bn2*n3*cth2 )/dsq0
    # phi0
    qo = fp - r0*( 1 + gp*hp )
    # ggrad dot hgrad
    ggrad_dot_hgrad = np.sum( ggrad*hgrad , axis=1 ).reshape( (len(qo),1) )
    # phi related
    phigrad = fgrad - r0*( gp*hgrad + ggrad*hp )
    philaplace  = flap - r0*( gp*hlap + glap*hp + 2*ggrad_dot_hgrad )
    phi_x = phigrad[:, 0]
    phi_x = phi_x[:, None]
    phi_y = phigrad[:, 1]
    phi_y = phi_y[:, None]
    phi_z = phigrad[:, 2]
    phi_z = phi_z[:, None]
    return phi_x, phi_y, phi_z, philaplace

def coff_a_n(data):
    pm4 = 4*np.pi
    x = data[:, 0]
    x = x[:, None]
    y = data[:, 1]
    y = y[:, None]
    z = data[:, 2]
    z = z[:, None]
    tx = pm4 * x
    ty = pm4 * y
    ctx = np.cos(tx)
    cty = np.cos(ty)
    stx = np.sin(tx)
    sty = np.sin(ty)
    cz = np.cos(z)
    sz = np.sin(z)
    a_x = 10. + ( stx - sty )*cz
    dadx = pm4*ctx*cz
    dady = - pm4*cty*cz
    dadz = - ( stx - sty )*sz
    return a_x, dadx, dady, dadz

def get_grad_u_n(data):
    x = data[:, 0]
    x = x[:, None]
    y = data[:, 1]
    y = y[:, None]
    z = data[:, 2]
    z = z[:, None]
    c2x = np.cos(2 * x)
    c2y = np.cos(2 * y)
    s2x = np.sin(2 * x)
    s2y = np.sin(2 * y)
    ez = np.exp(z)
    du_dx = c2x * c2y * ez * 2
    du_dy = - s2x * s2y * ez * 2
    du_dz = s2x * c2y * ez
    u_lapl = - s2x * c2y * ez * 7
    return du_dx, du_dy, du_dz, u_lapl

def get_f_p(data):
    du_dx, du_dy, du_dz, u_lapl = get_grad_u_n(data)
    f_p = u_lapl
    return f_p

def get_f_n(data):
    du_dx, du_dy, du_dz, u_lapl = get_grad_u_n(data)
    a_x, dadx, dady, dadz = coff_a_n(data)
    f_n = a_x * u_lapl + dadx * du_dx + dady * du_dy + dadz * du_dz
    return f_n

def get_Phi(x, r0, t1, t2, t3, n1, n2, n3, theta1, theta2, theta3):
    Phi = np.abs(level_set_funtion(x, r0, t1, t2, t3, n1, n2, n3, theta1, theta2, theta3))
    return Phi

def get_g(data):
    x = data[:, 0]
    x = x[:, None]
    y = data[:, 1]
    y = y[:, None]
    z = data[:, 2]
    z = z[:, None]
    u_n = np.sin(2 * x) * np.cos(2 * y) * np.exp(z)
    return u_n

def get_data(r0, t1, t2, t3, n1, n2, n3, theta1, theta2, theta3):
    sensor_resolution = 8 #传感器的分辨率
    omega_num = 400
    boundary_in_num = 100
    boundary_out_num = 200
    interface_num = 100
    sensor_location_p, sensor_location_n = get_sensors_location(sensor_resolution, r0, t1, t2, t3, n1, n2, n3, theta1, theta2, theta3)
    xop, xon = get_omega_points(omega_num, r0, t1, t2, t3, n1, n2, n3, theta1, theta2, theta3)
    xb_in = get_boundary_in_points(boundary_in_num)
    xb_out = get_boundary_out_points(boundary_out_num)
    xif = get_interface_points(interface_num, r0, t1, t2, t3, n1, n2, n3, theta1, theta2, theta3)
    xop_data = add_dimension(xop, r0, t1, t2, t3, n1, n2, n3, theta1, theta2, theta3)
    xon_data = add_dimension(xon, r0, t1, t2, t3, n1, n2, n3, theta1, theta2, theta3)
    xb_in_data = add_dimension(xb_in, r0, t1, t2, t3, n1, n2, n3, theta1, theta2, theta3)
    xb_out_data = add_dimension(xb_out, r0, t1, t2, t3, n1, n2, n3, theta1, theta2, theta3)
    xif_data = add_dimension(xif, r0, t1, t2, t3, n1, n2, n3, theta1, theta2, theta3)

    sensor_f_p = get_f_p(sensor_location_p)[:, 0]
    sensor_f_n = get_f_n(sensor_location_n)[:, 0]
    sensor_f = np.hstack((sensor_f_p, sensor_f_n))
    sensor_phi_p = get_Phi(sensor_location_p, r0, t1, t2, t3, n1, n2, n3, theta1, theta2, theta3)
    sensor_phi_n = get_Phi(sensor_location_n, r0, t1, t2, t3, n1, n2, n3, theta1, theta2, theta3)
    sensor_phi = np.hstack((sensor_phi_p, sensor_phi_n))

    f_xop = get_f_p(xop)
    f_xon = get_f_n(xon)
    g_b_in = get_g(xb_in)
    g_b_out = get_g(xb_out)
    n = get_normal_vector(xif, r0, t1, t2, t3, n1, n2, n3, theta1, theta2, theta3)
    phi_x_p, phi_y_p, phi_z_p, philaplace_p = get_grad_phi(xop, r0, t1, t2, t3, n1, n2, n3, theta1, theta2, theta3)
    phi_x_if, phi_y_if, phi_z_if, philaplace_if = get_grad_phi(xif, r0, t1, t2, t3, n1, n2, n3, theta1, theta2, theta3)
    a_x_n, dadx_n, dady_n, dadz_n = coff_a_n(xon)
    a_x_if, dadx_if, dady_if, dadz_if = coff_a_n(xif)

    data_xop = np.hstack((sensor_f*np.ones((len(f_xop), len(sensor_f))), sensor_phi*np.ones((len(f_xop), len(sensor_phi))),
                          xop_data, phi_x_p, phi_y_p, phi_z_p, philaplace_p, f_xop))
    data_xon = np.hstack((sensor_f*np.ones((len(f_xon), len(sensor_f))), sensor_phi*np.ones((len(f_xon), len(sensor_phi))),
                          xon_data, a_x_n, dadx_n, dady_n, dadz_n, f_xon))
    data_xb_in = np.hstack((sensor_f*np.ones((len(g_b_in), len(sensor_f))), sensor_phi*np.ones((len(g_b_in), len(sensor_f))),
                            xb_in_data, g_b_in))
    data_xb_out = np.hstack((sensor_f*np.ones((len(g_b_out), len(sensor_f))), sensor_phi*np.ones((len(g_b_out), len(sensor_f))),
                            xb_out_data, g_b_out))
    data_xif = np.hstack((sensor_f*np.ones((interface_num, len(sensor_f))), sensor_phi*np.ones((interface_num, len(sensor_phi))),
                          xif_data, phi_x_if, phi_y_if, phi_z_if, a_x_if, dadx_if, dady_if, dadz_if, n))

    return data_xop, data_xon, data_xb_in, data_xb_out, data_xif

def generate_train_data(sample_num, device):
    r0 = np.random.uniform(0.45, 0.55, sample_num)
    a1 = np.random.uniform(0., 0.2, sample_num)
    a2 = np.random.uniform(-0.2, 0., sample_num)
    a3 = np.random.uniform(0.1, 0.2, sample_num)
    n1 = np.random.uniform(2, 4, sample_num)
    n2 = np.random.uniform(3, 5, sample_num)
    n3 = np.random.uniform(6, 8, sample_num)
    theta1 = np.random.uniform(0.3, 0.7, sample_num)
    theta2 = np.random.uniform(1.6, 2.0, sample_num)
    theta3 = np.random.uniform(-0.2, 0.2, sample_num)

    data_xop, data_xon, data_xb_in, data_xb_out, data_xif = get_data(r0[0], a1[0], a2[0], a3[0], n1[0], n2[0], n3[0], theta1[0], theta2[0], theta3[0])
    for i in range(1, sample_num):
        data_xop_new, data_xon_new, data_xb_in_new, data_xb_out_new, data_xif_new = get_data(r0[i], a1[i], a2[i], a3[i], n1[i], n2[i], n3[i], theta1[i], theta2[i], theta3[i])
        data_xop = np.vstack((data_xop, data_xop_new))
        data_xon = np.vstack((data_xon, data_xon_new))
        data_xb_in = np.vstack((data_xb_in, data_xb_in_new))
        data_xb_out = np.vstack((data_xb_out, data_xb_out_new))
        data_xif = np.vstack((data_xif, data_xif_new))

    data_xb = np.vstack((data_xb_in, data_xb_out))
    train_xop = (data_xop[:, :136], data_xop[:, 136:272], data_xop[:, 272:276], data_xop[:, -5][:, None],
                 data_xop[:, -4][:, None], data_xop[:, -3][:, None], data_xop[:, -2][:, None], data_xop[:, -1][:, None])
    train_xon = (data_xon[:, :136], data_xon[:, 136:272], data_xon[:, 272:276], data_xon[:, -5][:, None],
                 data_xon[:, -4][:, None], data_xon[:, -3][:, None], data_xon[:, -2][:, None], data_xon[:, -1][:, None])
    train_xb = (data_xb[:, :136], data_xb[:, 136:272], data_xb[:, 272:276], data_xb[:, -1][:, None])
    train_xif = (data_xif[:, :136], data_xif[:, 136:272], data_xif[:, 272:276], data_xif[:, -10][:, None],
                 data_xif[:, -9][:, None],data_xif[:, -8][:, None],data_xif[:, -7][:, None],
                 data_xif[:, -6][:, None],data_xif[:, -5][:, None],data_xif[:, -4][:, None], data_xif[:, -3:])
    train_xop = To_tensor(train_xop, device=device)
    train_xon = To_tensor(train_xon, device=device)
    train_xb = To_tensor(train_xb, device=device)
    train_xif = To_tensor(train_xif, device=device)
    return train_xop, train_xon, train_xb, train_xif

def get_w(data):
    x = data[:, 0]
    x = x[:, None]
    y = data[:, 1]
    y = y[:, None]
    z = data[:, 2]
    z = z[:, None]
    w = np.sin(2 * x) * np.cos(2 * y) * np.exp(z)
    return w

def generate_test_data(num, device):
    r0 = 0.483
    t1 = 0.1
    t2 = -0.1
    t3 = 0.15
    n1 = 3.
    n2 = 4.
    n3 = 7.
    theta1 = 0.5
    theta2 = 1.8
    theta3 = 0.
    sensor_resolution = 8 #传感器的分辨率
    sensor_location_p, sensor_location_n = get_sensors_location(sensor_resolution, r0, t1, t2, t3, n1, n2, n3, theta1, theta2, theta3)
    xop, xon = get_omega_points(num, r0, t1, t2, t3, n1, n2, n3, theta1, theta2, theta3)
    x = np.vstack((xop, xon))
    x = add_dimension(x, r0, t1, t2, t3, n1, n2, n3, theta1, theta2, theta3)
    w = get_w(x)
    sensor_f_p = get_f_p(sensor_location_p)[:, 0]
    sensor_f_n = get_f_n(sensor_location_n)[:, 0]
    sensor_f = np.hstack((sensor_f_p, sensor_f_n))
    sensor_phi_p = get_Phi(sensor_location_p, r0, t1, t2, t3, n1, n2, n3, theta1, theta2, theta3)
    sensor_phi_n = get_Phi(sensor_location_n, r0, t1, t2, t3, n1, n2, n3, theta1, theta2, theta3)
    sensor_phi = np.hstack((sensor_phi_p, sensor_phi_n))
    test_data = (sensor_f * np.ones((len(x), len(sensor_f))), sensor_phi * np.ones((len(x), len(sensor_phi))), x, w)
    test_data = To_tensor(test_data, device=device)
    return test_data

