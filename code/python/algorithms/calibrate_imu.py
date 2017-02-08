import numpy as np
import math

def allan(t, omega):
    """
    Compute allan deviation plot
    :param t: time stamp
    :param omega: input signal
    :return: log-x, log-y
    """
    num_samples = omega.shape[0]
    n = 2 ** np.arange(0, np.floor(np.log2(num_samples / 2)))
    max_n = n[-1]


def calibrate_imu(time_stamp, samples):
    """
    IMU intrinsic calibration
    :param time_stamp: Nx1 array time stamp of each sample
    :param input_data: Nxm array of samples. Each of the channels will be calibrated independently
    :return: 1xm white noise, 1xm bias instability
    """
    assert time_stamp.shape[0] == samples.shape[0]