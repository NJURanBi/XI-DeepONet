import numpy as np
import os
import time
from sklearn import gaussian_process as gp
from scipy import interpolate
import random

def get_data(inter_a, p, Remark):
    length_scalel = 0.2
    length_scaler = 0.1
    N = 998
    xmin,xmax= 0,1
    h = (xmax-xmin)/(N+1)
    mesh = np.linspace(xmin, xmax, N+2,endpoint=True)
    ### define the interface
    alpha = inter_a

    ### define coef
    k1 = 0.1
    k2 = 0.5
    features = 500
    ## sensor number
    sensors_num = 100
    sample_test_num = 1


    # boundary condition
    def utrue_solution(mesh,alpha):
        u = []
        for i in mesh:
            if i <=alpha:
                u.append(k2*(i-alpha)**3)
            else:
                u.append(k1*(i-alpha)**3)
        return np.array(u)

    ### g_D
    def f_dirich(x):
        return 0.

    ### g_N
    def f_numan(x):
        return 0.


    def weight_order0(alpha,x_array):
        # Taylor expansion
        a = []
        a.append([1, 1, 1])
        a.append(x_array - alpha)
        a.append((x_array - alpha)**2)
        a = np.array(a)
        b = np.array([1, 0, 0]).reshape(-1, 1)
        w = np.dot(np.linalg.inv(a), b)
        return w.reshape(-1,)


    def weight_order1(alpha,x_array):
        a = []
        a.append([1, 1, 1])
        a.append(x_array - alpha)
        a.append((x_array - alpha)**2)
        a = np.array(a)
        b = np.array([0, 1, 0]).reshape(-1, 1)
        w = np.dot(np.linalg.inv(a), b)
        return w.reshape(-1,)


    def fi_and_fiadd1(w01,w02,w11,w12):

        a = np.array([[w02[0], -w01[2]], [k2 * w12[0], -k1 * w11[2]]])  # right
        a_inv = np.linalg.inv(a)
        b = np.array([[w01[0], w01[1], -w02[1], -w02[2], 1, 0], [k1 * w11[0], k1 * w11[1], -k2 * w12[1], -k2 * w12[2], 0, 1]])  # ui-1,ui,ui+2,ui+2,[u],[un]
        return a_inv@b


    def modify_matrix(A,F,mesh,alpha,record):
        """
        alpha in i and i+1
        A_local@[ui-1,ui,ui+2,ui+2,[u],[un]].T=[fi,fi+1].T
        """
        if alpha not in mesh:
            i = record[0] - 1
            x_1 = mesh[record[0] - 1:record[1] + 1]
            x_2 = mesh[record[0]:record[1] + 2]
            w01 = weight_order0(alpha, x_1)
            w02 = weight_order0(alpha, x_2)

            w11 = weight_order1(alpha, x_1)
            w12 = weight_order1(alpha, x_2)

            A_local = fi_and_fiadd1(w01, w02, w11, w12)

            A[i, i + 1] = 0
            A[i, i - 1:i + 3] -= A_local[1, :4] * k1

            A[i + 1, i] = 0
            A[i + 1, i - 1:i + 3] -= A_local[0, :4] * k2
            inter_condi = np.array([f_dirich(alpha), f_numan(alpha)]).reshape(-1, 1)

            F[i] += (A_local[:, 4:] @ inter_condi)[1][0] * k1
            F[i + 1] += (A_local[:, 4:] @ inter_condi)[0][0] * k2

        else:
            index = np.argmin(abs(mesh[1:-1] - alpha))
            F[index + 1] -= k2 * f_dirich(alpha)

            x_1 = mesh[1:-1][index - 1:index + 2]
            x_2 = mesh[1:-1][index:index + 3]

            w11 = weight_order1(alpha, x_1) * k1
            w12 = weight_order1(alpha, x_2) * k2

            cc = np.array([-w11[0], -w11[1] + w12[0], w12[1], w12[2]]) / w11[2] * k1
            A[index, index + 1] = 0
            A[index, index - 1:index + 3] += cc
            F[index] -= f_numan(alpha) / w11[2] * k1
            F[index] -= w12[0] * f_dirich(alpha) / w11[2] * k1

        return A, F


    x = np.linspace(0, alpha, num=features)[:, None]
    A_left = gp.kernels.RBF(length_scale=length_scalel)(x)
    L_left = np.linalg.cholesky(A_left + 1e-10 * np.eye(features))
    gps_left = (L_left @ np.random.randn(features, sample_test_num)).transpose()

    x = np.linspace(alpha, 1, num=features)[:, None]
    A_right = gp.kernels.RBF(length_scale=length_scaler)(x)
    L_right = np.linalg.cholesky(A_right + 1e-10 * np.eye(features))
    gps_right = (L_right @ np.random.randn(features, sample_test_num)).transpose()

    x_sensor = np.linspace(0, 1, num=sensors_num)
    xl_sensor = x_sensor[np.where(x_sensor <= alpha)[0]]
    xr_sensor = x_sensor[np.where(x_sensor > alpha)[0]]


    def deal_with_mesh(mesh=mesh, alpha=alpha, k1=k1,k2=k2):
        """
        input: mesh, alpha, k1, k2
        output:record and coef (two np.array)
        """
        coef = np.zeros(N + 2)
        index = np.argmin(abs(mesh - alpha))
        if mesh[index] <= alpha and mesh[index + 1] > alpha:
            coef[:index + 1] = k1
            coef[index + 1:] = k2
            record = [index, index + 1]

        elif mesh[index] > alpha and mesh[index - 1] <= alpha:
            coef[:index] = k1
            coef[index:] = k2
            record = [index - 1, index]

        return np.array(record), np.array(coef)

    record, coef = deal_with_mesh()

    def get_phi(x):
        phi = x - alpha
        return phi

    def solve(gp_left, gp_right, features=features, alpha=alpha, xl_sensor=xl_sensor, xr_sensor=xr_sensor, mesh=mesh):
        x = np.linspace(0, alpha, num=features)[:, None]
        u_left = interpolate.interp1d(x.reshape(-1, ), gp_left, kind='cubic', copy=False, assume_sorted=True)
        x = np.linspace(alpha, 1, num=features)[:, None]
        u_right = interpolate.interp1d(x.reshape(-1, ), gp_right, kind='cubic', copy=False, assume_sorted=True)

        u_pre = np.zeros(N + 2)
        u_pre[0] = 0.
        u_pre[-1] = 0.

        A = np.zeros((N, N))
        for i in range(N):
            A[i, i] = 2
            if i < N - 1: A[i, i + 1] = -1
            if i > 0:   A[i, i - 1] = -1
        A = np.diag(coef[1:-1]) @ A

        F = np.zeros(N).reshape(-1, 1)
        index = np.where(mesh[1:-1] <= alpha)[0]
        F[index] = (u_left(mesh[1:-1][index]) * h ** 2).reshape(-1, 1)
        index = np.where(mesh[1:-1] > alpha)[0]
        F[index] = (u_right(mesh[1:-1][index]) * h ** 2).reshape(-1, 1)
        F[0] += u_pre[0] * k1
        F[-1] += u_pre[-1] * k2

        A_new, F_new = modify_matrix(A, F, mesh, alpha, record)
        u_pre[1:-1] = np.dot(np.linalg.inv(A_new), F_new).reshape(-1, )

        grad_u_pre = []
        g1 = (u_pre[1] * (-5 / 2) + u_pre[2] * 4 + u_pre[3] * (-3 / 2)) / h
        grad_u_pre.append(g1)
        for i in range(1, N + 1):
            g1 = (u_pre[i + 1] - u_pre[i - 1]) / (2 * h)
            grad_u_pre.append(g1)
        g1 = (u_pre[-2] * (5 / 2) + u_pre[-3] * (-4) + u_pre[-4] * (3 / 2)) / h
        grad_u_pre.append(g1)
        grad_u_pre = np.array(grad_u_pre)

        # test data
        x_left = mesh[np.where(mesh <= alpha)[0]].reshape(-1, 1)
        x_right = mesh[np.where(mesh > alpha)[0]].reshape(-1, 1)
        z_left = np.hstack((x_left, np.abs(get_phi(x_left))))
        z_right = np.hstack((x_right, np.abs(get_phi(x_right))))
        # reference solution
        u_preL = u_pre[np.where(mesh <= alpha)[0]].reshape(-1, 1)
        u_preR = u_pre[np.where(mesh > alpha)[0]].reshape(-1, 1)

        total_sensor = np.append(u_left(xl_sensor), u_right(xr_sensor))
        sensorl = np.tile(total_sensor, (x_left.shape[0], 1))
        sensorr = np.tile(total_sensor, (x_right.shape[0], 1))
        ug_preL = grad_u_pre[np.where(mesh <= alpha)[0]].reshape(-1, 1)
        ug_preR = grad_u_pre[np.where(mesh > alpha)[0]].reshape(-1, 1)

        phi_sensors = get_phi(np.append(xl_sensor, xr_sensor))
        phi_sensors_l = np.tile(phi_sensors, (x_left.shape[0], 1))
        phi_sensors_r = np.tile(phi_sensors, (x_right.shape[0], 1))

        return z_left, z_right, u_preL, u_preR, ug_preL, ug_preR, sensorl, sensorr, F, phi_sensors_l, phi_sensors_r


    test_xleft, test_xright, label_left, label_right, label_grad_left, label_grad_right, sensor_left, sensor_right, F, phi_sensors_l, phi_sensors_r= solve(gps_left[0], gps_right[0])
    nnnl = len(test_xleft)
    nnnr = len(test_xright)
    sample_function = []
    sample_function.append(F)
    t0 = time.time()

    if Remark == 'Train':
        xl_num = len(test_xleft)
        xr_num = len(test_xright)
        p1 = int(p*(xl_num/(xl_num + xr_num)))
        p2 = p - p1
        index1 = random.sample(range(xl_num), p1)
        index2 = random.sample(range(xr_num), p2)
        test_xleft = test_xleft[index1]
        label_left = label_left[index1]
        sensor_left = sensor_left[index1]
        phi_sensors_l = phi_sensors_l[index1]
        test_xright = test_xright[index2]
        label_right = label_right[index2]
        sensor_right = sensor_right[index2]
        phi_sensors_r = phi_sensors_r[index2]


    return test_xleft, test_xright, label_left, label_right, sensor_left, sensor_right, phi_sensors_l, phi_sensors_r