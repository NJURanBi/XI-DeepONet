import numpy as np
import random
from Tools import To_tensor

def get_test_data(data_file, device):
    test_sensor = np.loadtxt(data_file+'Test_sensor.txt')
    test_x = np.loadtxt(data_file+'Test_x.txt')
    test_real_u = np.loadtxt(data_file+'Test_label.txt')
    test_real_u = test_real_u[:, None]
    test_x = test_x[:, 0]
    test_x = test_x[:, None]
    X_test = (test_sensor, test_x, test_real_u)
    X_test = To_tensor(X_test, device)
    return  X_test


def get_train_data(data_file, device):
    train_sensor = np.loadtxt(data_file+'Train_sensor.txt')
    train_x = np.loadtxt(data_file+'Train_x.txt')
    train_real_u = np.loadtxt(data_file+'Train_label.txt')
    train_real_u = train_real_u[:, None]
    train_x = train_x[:, 0]
    train_x = train_x[:, None]
    X_train = (train_sensor, train_x, train_real_u)
    X_train = To_tensor(X_train, device)
    return X_train

