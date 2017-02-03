import sys
import os
import numpy as np
import quaternion
import scipy.integrate as integrate
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

sys.path.append(os.path.dirname(os.path.abspath(__file__) + '/..'))
from utility.write_trajectory_to_ply import write_ply_to_file


imu_to_tango_global = np.matrix([[1.0, 0.0, 0.0],
                                [0.0, 0.0, 1.0],
                                [0.0, -1.0, 0.0]])

# imu_to_tango_global = np.matrix([[1.0, 0.0, 0.0],
#                                 [0.0, 1.0, 0.0],
#                                 [0.0, 0.0, 1.0]])


def IMU_double_integration(t, rotation, acceleration):
    """
    Compute position and orientation by integrating angular velocity and double integrating acceleration
    Expect the drift to be as large as hell
    :param t: time sequence, Nx1 array
    :param rotation: device orientation as quaternion, Nx4 array
    :param acceleration: acceleration data, Nx3 array
    :return: position: Nx3 array
    """
    # Sanity check
    assert t.shape[0] == rotation.shape[0]
    assert t.shape[0] == acceleration.shape[0]
    assert rotation.shape[1] == 4

    quats = quaternion.as_quat_array(rotation)
    # convert the acceleration vector to world coordinate frame

    result = np.empty([quats.shape[0], 3], dtype=float)
    for i in range(acceleration.shape[0]):
        result[i, :] = np.dot(quaternion.as_rotation_matrix(quats[i]),
                              np.dot(imu_to_tango_global, acceleration[i, :].reshape([3, 1]))).flatten()
    # double integration with trapz rule
    position = integrate.cumtrapz(integrate.cumtrapz(result, t, axis=0, initial=0), t, axis=0, initial=0)
    return position


if __name__ == '__main__':
    import argparse
    import pandas
    from scipy.ndimage.filters import gaussian_filter1d

    nano_to_sec = 1e09

    parser = argparse.ArgumentParser()
    parser.add_argument('dir', type=str)
    parser.add_argument('--output', type=str)

    args = parser.parse_args()
    data_all = pandas.read_csv(args.dir + '/processed/data.csv')

    time_stamp = data_all['time'].values / nano_to_sec
    rotations = data_all[['ori_w', 'ori_x', 'ori_y', 'ori_z']].values
    positions_gt = data_all[['pos_x', 'pos_y', 'pos_z']].values

    linacce = data_all[['linacce_x', 'linacce_y', 'linacce_z']].values
    linacce = gaussian_filter1d(linacce, axis=0, sigma=50.0)
    # linacce[:, [0, 2]] = 0.0
    linacce[:, 2] = 0.0

    positions = IMU_double_integration(t=time_stamp, rotation=rotations, acceleration=linacce)

    # plt.figure()
    # ax = plt.subplot(111, projection='3d')
    # heading = int(positions.shape[0] / 10)
    # ax.plot(positions[:heading, 0], positions[:heading, 1], positions[:heading, 2], 'r')
    # ax.plot(positions[heading:, 0], positions[heading:, 1], positions[heading:, 2], 'b')
    # ax.plot(positions_gt[:, 0], positions_gt[:, 1], positions_gt[:, 2], 'g')
    # plt.show()
    if args.output is not None:
        write_ply_to_file(path=args.output, position=positions, orientation=rotations)
        print('Write ply to ' + args.output)
