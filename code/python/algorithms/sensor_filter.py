import sys
import os
import math
import numpy as np
import quaternion
import matplotlib.pyplot as plt
from scipy.ndimage.filters import gaussian_filter1d
import filterpy

sys.path.append(os.path.dirname(os.path.abspath(__file__) + '/..'))


def estimate_gravity(acce_data, alpha=0.9):
    """
    Estimate the gravity with a low pass filter
    :param acce_data: Nx3 array containing the accelerometer data
    :param alpha: filter rate
    :return: Nx3 array containing gravity vector at each frame
    """
    gravity = np.zeros(acce_data.shape, dtype=float)
    for i in range(1, acce_data.shape[0]):
        gravity[i] = gravity[i-1] * alpha + acce_data[i-1] * (1.0 - alpha)
    return gravity


def complementary_filter(ts, gyro_data, acce_data, alpha=0.98):
    assert gyro_data.shape[0] == ts.shape[0]
    assert acce_data.shape[0] == ts.shape[0]
    angle = np.zeros(gyro_data.shape, dtype=float)
    for i in range(1, ts.shape[0]):
        angle_from_acce = np.array([math.atan2(acce_data[i, 1], acce_data[i, 2]),
                                    math.atan2(acce_data[i, 0], acce_data[i, 2]),
                                    math.atan2(acce_data[i, 0], acce_data[i, 1])])
        angle[i] = (angle[i-1] + gyro_data[i-1] * (ts[i] - ts[i-1])) * alpha + angle_from_acce * (1.0 - alpha)
    return angle


if __name__ == '__main__':
    import pandas
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('dir', type=str, default=None)
    parser.add_argument('--alpha', default=0.98, type=float)
    parser.add_argument('--skip', default=300, type=int)

    args = parser.parse_args()

    data_all = pandas.read_csv(args.dir + '/processed/data.csv')

    ts = data_all['time'].values / 1e09

    acce_data = data_all[['acce_x', 'acce_y', 'acce_z']].values
    gravity = estimate_gravity(acce_data, alpha=0.9)
    linacce = acce_data[args.skip:] - gravity[args.skip:]
    gravity = gravity[args.skip:]
    ts = ts[args.skip:]
    acce_data = acce_data[args.skip:]

    gyro_data = data_all[['gyro_x', 'gyro_y', 'gyro_z']].values[args.skip:]
    grav_data = data_all[['grav_x', 'grav_y', 'grav_z']].values[args.skip:]
    pose_data = data_all[['pos_x', 'pos_y', 'pos_z']].values[args.skip:]
    orientation_data = data_all[['ori_w', 'ori_x', 'ori_y', 'ori_z']].values[args.skip:]

    # visualize estimated gravity
    plt.figure('Gravity')
    for i in range(gravity.shape[1]):
        plt.subplot(gravity.shape[1] * 100 + 11 + i)
        plt.plot(ts, grav_data[:, i])
        plt.plot(ts, gravity[:, i])
        plt.legend(['sensor', 'filter'])

    # angle_filtered = complementary_filter(ts, gyro_data, acce_data, args.alpha)
    # angle_quat = quaternion.as_quat_array(angle_filtered)
    # from utility.write_trajectory_to_ply import write_ply_to_file

    plt.show()






