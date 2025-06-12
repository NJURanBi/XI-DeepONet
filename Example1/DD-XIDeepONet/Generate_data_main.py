import numpy as np
import os
import time
from sklearn import gaussian_process as gp
from scipy import interpolate
from Generate_data import get_data

data_file = 'data/'
if not os.path.isdir('./'+data_file): os.makedirs('./'+data_file)
num = 100
p = 100
alpha = np.random.uniform(0.3, 0.7, num)

#Remark = 'Train'
Remark = 'Test'
for i in range(num):
    print(i)
    if i == 0:
        a = alpha[0]
        test_xleft, test_xright, label_left, label_right, sensor_left, sensor_right, phi_sensors_l, phi_sensors_r = get_data(a, p, Remark)
        x_test = np.vstack((test_xleft, test_xright))
        sensors = np.vstack((sensor_left, sensor_right))
        u_label = np.vstack((label_left, label_right))
        phi_sensors = np.vstack((phi_sensors_l, phi_sensors_r))

    else:
        a = alpha[i]
        test_xleft_new, test_xright_new, label_left_new, label_right_new, sensor_left_new, sensor_right_new, phi_sensors_l_new, phi_sensors_r_new = get_data(a, p, Remark)
        x_test_new = np.vstack((test_xleft_new, test_xright_new))
        sensors_new = np.vstack((sensor_left_new, sensor_right_new))
        u_label_new = np.vstack((label_left_new, label_right_new))
        phi_sensors_new = np.vstack((phi_sensors_l_new, phi_sensors_r_new))

        x_test = np.vstack((x_test, x_test_new))
        sensors = np.vstack((sensors, sensors_new))
        u_label = np.vstack((u_label, u_label_new))
        phi_sensors = np.vstack((phi_sensors, phi_sensors_new))


if Remark == 'Train':
    np.savetxt(data_file + 'Train_x.txt', x_test)
    np.savetxt(data_file + 'Train_label.txt', u_label)
    np.savetxt(data_file + 'Train_sensor.txt', sensors)
    np.savetxt(data_file + 'Train_phi_sensor.txt', phi_sensors)

if Remark == 'Test':
    np.savetxt(data_file + 'Test_x.txt', x_test)
    np.savetxt(data_file + 'Test_label.txt', u_label)
    np.savetxt(data_file + 'Test_sensor.txt', sensors)
    np.savetxt(data_file + 'Test_phi_sensor.txt', phi_sensors)