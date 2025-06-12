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
from matplotlib import pyplot as plt
from Tools import To_tensor

def get_u(data):
    x = data[:, 0]
    x = x[:, None]
    y = data[:, 1]
    y = y[:, None]
    z = data[:, 2]
    z = z[:, None]
    w = np.sin(2 * x) * np.cos(2 * y) * np.exp(z)
    return w
def get_theta(x):
    index1 = x[:,0] >= 0
    theta1 = np.arctan(x[:,1]/x[:,0])
    index2 = x[:,0] < 0
    theta2 = np.arctan(x[:,1]/x[:,0]) + np.pi
    index1 = index1.astype(float)
    index2 = index2.astype(float)
    theta = index1 * theta1 + index2 * theta2
    theta = theta[:, None]
    return theta

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

def add_dimension(x, r0, t1, t2, t3, n1, n2, n3, theta1, theta2, theta3):
    lf = level_set_funtion(x, r0, t1, t2, t3, n1, n2, n3, theta1, theta2, theta3)[:, None]
    index = lf >= 0.
    index = index.astype(float)
    x_add = np.hstack((x, np.abs(lf * index)))
    return x_add


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
def get_data(x, r0, t1, t2, t3, n1, n2, n3, theta1, theta2, theta3):
    x = x.detach().numpy()
    r = np.sqrt(x[:, 0] ** 2 + x[:, 1] ** 2 + x[:, 2] ** 2)
    index = np.where((r > 0.151) & (r < 0.911))[0]
    x = x[index, :]
    sensor_resolution = 8  # 传感器的分辨率
    sensor_location_p, sensor_location_n = get_sensors_location(sensor_resolution, r0, t1, t2, t3, n1, n2, n3, theta1, theta2, theta3)
    x = add_dimension(x, r0, t1, t2, t3, n1, n2, n3, theta1, theta2, theta3)
    u = get_u(x)

    sensor_f_p = get_f_p(sensor_location_p)[:, 0]
    sensor_f_n = get_f_n(sensor_location_n)[:, 0]
    sensor_f = np.hstack((sensor_f_p, sensor_f_n))
    sensor_phi_p = get_Phi(sensor_location_p, r0, t1, t2, t3, n1, n2, n3, theta1, theta2, theta3)
    sensor_phi_n = get_Phi(sensor_location_n, r0, t1, t2, t3, n1, n2, n3, theta1, theta2, theta3)
    sensor_phi = np.hstack((sensor_phi_p, sensor_phi_n))

    test_data = (sensor_f * np.ones((len(x), len(sensor_f))), sensor_phi * np.ones((len(x), len(sensor_phi))), x, u)
    test_data = To_tensor(test_data, device=device)
    return test_data

def get_v(x, r0, t1, t2, t3, n1, n2, n3, theta1, theta2, theta3):
    x = x.detach().numpy()
    x1 = x[:, 0]
    x2 = x[:, 1]
    x3 = x[:, 2]
    eu1 = np.sin(2 * x1) * np.cos(2 * x2) * np.exp(x3)
    tp = (x2 - x1) / 3.
    tpsq = tp * tp
    f = ((16 * tpsq - 20) * tpsq + 5) * tp
    g = np.log(x1 + x2 + 3)
    h = np.cos(x3)
    eu2 = f * g * h
    u_p = eu2 - eu1
    lf = level_set_funtion(x, r0, t1, t2, t3, n1, n2, n3, theta1, theta2, theta3)
    index = lf >= 0
    u_n = np.zeros_like(u_p)
    u = np.where(index, u_p, u_n)
    u = u[:, None]
    u = To_tensor(u, device)
    return u

if torch.cuda.is_available():
    device ='cuda'
else:
    device = 'cpu'
model_type ='ionet'
model_1 = XIDeepONet(sensor_dim=136, h_dim=150, in_dim=4, actv=nn.Tanh()).to(device)
model_1 = torch.load('pi_ion_ionet_150_40000.pkl', map_location='cpu')

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
''''-------------------------画图--------------------------------------------'''
with torch.no_grad():
    x1 = torch.linspace(-1, 1, 200)
    x2 = torch.linspace(-1, 1, 200)
    X, Y = torch.meshgrid(x1, x2)
    Z = torch.cat((Y.flatten()[:, None], Y.T.flatten()[:, None]), dim=1)
    Z = torch.cat((Z, torch.zeros(len(Z), 1)), 1)
plt.figure(1)
data = get_data(Z, r0, t1, t2, t3, n1, n2, n3, theta1, theta2, theta3)
pred = model_1(data[1], data[2])
ur = data[-1]
loss_l2 = torch.sqrt(torch.mean(torch.pow(pred - ur,2)))
loss_rel_l2 = loss_l2 / torch.sqrt(torch.mean(torch.pow(ur, 2)))
print('l2相对误差:', loss_rel_l2.item())
X_tsd = data[2]
abserr = torch.abs(pred - ur)
v = get_v(X_tsd, r0, t1, t2, t3, n1, n2, n3, theta1, theta2, theta3)
pred = pred + v
X_tsd = X_tsd.detach().numpy()
pred = pred.detach().numpy()
abserr = abserr.detach().numpy()
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
plt.savefig('pred_u.jpg', bbox_inches='tight', dpi=600)
plt.clf()

ax = fig.add_subplot(1, 1, 1, projection='3d')
surf = ax.scatter(X_tsd[:,0:1], X_tsd[:,1:2], abserr, c=abserr, cmap='plasma')
fig.colorbar(surf, shrink=0.5, aspect=5)
plt.xlabel('x')
plt.ylabel('y')
plt.title('Error distribution')
ax.view_init(elev=30, azim=30)
plt.savefig('error.jpg', bbox_inches='tight', dpi=600)
plt.show()






