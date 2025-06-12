import sys
sys.path.append('..')
import numpy as np
from sklearn import gaussian_process as gp
from scipy import interpolate
from Tools import To_tensor

length_scalel = 0.2
length_scaler = 0.1
features = 500
sensors_num = 100

def generate_test_data(data_file, device):
    test_sensor = np.loadtxt(data_file+'Test_sensor.txt')
    test_phi_sensor = np.loadtxt(data_file+'Test_phi_sensor.txt')
    test_x = np.loadtxt(data_file+'Test_x.txt')
    test_real_u = np.loadtxt(data_file+'Test_label.txt')
    test_real_u = test_real_u[:, None]
    X_test = (test_sensor, test_phi_sensor, test_x, test_real_u)
    X_test = To_tensor(X_test, device)
    return X_test


def gaussian_process(sample_num, alpha):
    x = np.linspace(0, alpha, num=features)[:, None]
    A_left = gp.kernels.RBF(length_scale=length_scalel)(x)
    L_left = np.linalg.cholesky(A_left + 1e-10 * np.eye(features))
    gps_left=(L_left @ np.random.randn(features, sample_num)).transpose()

    x = np.linspace(alpha,1, num=features)[:, None]
    A_right = gp.kernels.RBF(length_scale=length_scaler)(x)
    L_right = np.linalg.cholesky(A_right+ 1e-10 * np.eye(features))
    gps_right=(L_right @ np.random.randn(features, sample_num)).transpose()
    return gps_left, gps_right

def get_phi(x, alpha):
    phi = x - alpha
    return phi

def add_dimension(x, alpha):
    x = x[:, None]
    phi = np.abs(x - alpha)
    data = np.hstack((x, phi))
    return data
def generate(gp_left, gp_right, alpha, p):
    x = np.linspace(0, alpha, num=gp_left.shape[-1])[:, None]
    u_left = interpolate.interp1d(x.reshape(-1, ), gp_left, kind='cubic', copy=False, assume_sorted=True)
    x = np.linspace(alpha, 1, num=gp_right.shape[-1])[:, None]
    u_right = interpolate.interp1d(x.reshape(-1, ), gp_right, kind='cubic', copy=False, assume_sorted=True)
    xleft = np.sort(np.random.rand(p)) * alpha
    xright = np.sort(np.random.rand(p)) * (1 - alpha) + alpha
    rhsleft = u_left(xleft)[0][:, None]
    rhsright = u_right(xright)[0][:, None]
    # sensor
    x_sensor = np.linspace(0, 1, num=sensors_num)
    xl_sensor = x_sensor[np.where(x_sensor <= alpha)[0]]
    xr_sensor = x_sensor[np.where(x_sensor > alpha)[0]]
    u_sensors_left = u_left(xl_sensor)
    u_sensors_right = u_right(xr_sensor)
    f_sensor = np.hstack((u_sensors_left, u_sensors_right))
    phi_sensor = get_phi(x_sensor, alpha)[None, :]
    interface = np.array([alpha])
    boundary_l = np.array([0.])
    boundary_r = np.array([1.])
    data_left = add_dimension(xleft, alpha)
    data_right = add_dimension(xright, alpha)
    data_if = add_dimension(interface, alpha)
    data_bl = add_dimension(boundary_l, alpha)
    data_br = add_dimension(boundary_r, alpha)
    sample_left = np.hstack((f_sensor * np.ones((p, 1)), phi_sensor * np.ones((p, 1)), data_left, rhsleft))
    sample_right = np.hstack((f_sensor * np.ones((p, 1)), phi_sensor * np.ones((p, 1)), data_right, rhsright))
    sample_interface = np.hstack([f_sensor, phi_sensor, data_if])
    sample_boundaryl = np.hstack([f_sensor, phi_sensor, data_bl])
    sample_boundaryr = np.hstack([f_sensor, phi_sensor, data_br])

    return sample_left, sample_right, sample_interface, sample_boundaryl, sample_boundaryr
def generate_train_data(sample_num, device,  p):
    alpha =  np.random.uniform(0.3, 0.7, sample_num)
    gps1, gps2 = gaussian_process(1, alpha[0])
    sample_left,sample_right, sample_interface, sample_boundaryl, sample_boundaryr = generate(gps1, gps2, alpha[0], p)

    for i in range(1, sample_num):
        gps1, gps2 = gaussian_process(1, alpha[i])
        s1, s2, s3, s4, s5=generate(gps1, gps2, alpha[i], p)
        sample_left=np.vstack([sample_left, s1])
        sample_right=np.vstack([sample_right, s2])
        sample_interface = np.vstack([sample_interface,s3])
        sample_boundaryl = np.vstack([sample_boundaryl,s4])
        sample_boundaryr = np.vstack([sample_boundaryr,s5])


    sample_left = (sample_left[:, :100], sample_left[:, 100:200], sample_left[:, 200:202], sample_left[:, -1][:, None])
    sample_right = (sample_right[:, :100], sample_right[:, 100:200], sample_right[:, 200:202], sample_right[:, -1][:, None])
    sample_interface = (sample_interface[:, :100], sample_interface[:, 100:200], sample_interface[:, 200:])
    sample_boundaryl = (sample_boundaryl[:, :100], sample_boundaryl[:, 100:200], sample_boundaryl[:, 200:])
    sample_boundaryr = (sample_boundaryr[:, :100], sample_boundaryr[:, 100:200], sample_boundaryr[:, 200:])

    sample_left = To_tensor(sample_left, device)
    sample_right = To_tensor(sample_right, device)
    sample_interface = To_tensor(sample_interface, device)
    sample_boundaryl =  To_tensor(sample_boundaryl, device)
    sample_boundaryr = To_tensor(sample_boundaryr, device)

    return sample_left, sample_right, sample_interface, sample_boundaryl, sample_boundaryr
