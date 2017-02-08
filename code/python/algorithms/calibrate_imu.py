import numpy as np
import math
from scipy.interpolate import interp1d
from numba import jit

asdfsadfj
jaskdjf;lkasd
;klaj;ska;df
#@jit
def allan_plot(t, omega, pts=200):
    """
    Compute allan deviation plot
    :param t: time stamp (in seconds)
    :param omega: input signal
    :return: tau, overlapped allan deviation
    """
    num_samples = omega.shape[0]
    n = 2 ** np.arange(0, np.floor(np.log2(num_samples / 2)))
    m = np.unique(np.ceil(np.logspace(0, math.log10(n[-1]), pts))).astype(int)
    tau0 = (t[-1] - t[0]) / t.shape[0]
    print(tau0)
    T = m * tau0
    # integral
    theta = np.cumsum(omega, axis=0) * tau0
    sigma2 = np.zeros([m.shape[0], omega.shape[1]], dtype=float)
    for i in range(m.shape[0]):
        for k in range(num_samples - 2 * int(m[i])):
            sigma2[i, :] = sigma2[i, :] + (theta[k + 2*m[i], :] - 2 * theta[k + m[i], :] + theta[k, :]) ** 2
    sigma2 /= (2 * (T ** 2) * (num_samples - 2 * m))[:, None]
    return T, np.sqrt(sigma2)


def calibrate_imu(time_stamp, samples):
    """
    IMU intrinsic calibration
    :param time_stamp: Nx1 array time stamp of each sample
    :param input_data: Nxm array of samples. Each of the channels will be calibrated independently
    :return: 1xm white noise, 1xm bias instability
    """
    assert time_stamp.shape[0] == samples.shape[0]
    time_stamp /= 1e09

    allan_x, allan_y = allan_plot(time_stamp, samples)


if __name__ == '__main__':
    import argparse
    import matplotlib.pyplot as plt

    parser = argparse.ArgumentParser()
    parser.add_argument('dir')
    parser.add_argument('--skip', default=10000, type=int)

    args = parser.parse_args()

    print('Reading')
    acce_data = np.genfromtxt(args.dir + '/linacce.txt')[args.skip:-args.skip-1]
    acce_data[:, 0] /= 1e09

    print('Computing allan deviation')
    tau, adv = allan_plot(acce_data[:, 0], acce_data[:, 1:])

    allan_curve = interp1d(tau, adv, axis=0)
    print('Random walk:')
    print(allan_curve(1.0))
    print('Bias: ')
    print(np.min(adv, axis=0))

    plt.figure('Allan Plot')
    # plt.plot(tau, adv)
    plt.loglog(tau, adv)
    plt.grid(True)
    plt.show()
