import numpy as np
import quaternion
import scipy.integrate as integrate
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from mpl_toolkits.mplot3d import Axes3D

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
    result = [np.dot(quaternion.as_rotation_matrix(quats[i]), acceleration[i, :])
              for i in range(acceleration.shape[0])]
    # double integration with trapz rule
    result = integrate.cumtrapz(integrate.cumtrapz(result, t, axis=0, initial=0), t, axis=0, initial=0)
    return result


if __name__ == '__main__':
    import argparse
    import pandas

    nano_to_sec = 1e09

    parser = argparse.ArgumentParser()
    parser.add_argument('dir', type=str)

    args = parser.parse_args()
    data_all = pandas.read_csv(args.dir + '/processed/data.csv')

    time_stamp = data_all['time'].values / nano_to_sec
    rotations = data_all[['ori_w', 'ori_x', 'ori_y', 'ori_z']].values
    accelerations = data_all[['linacce_x', 'linacce_y', 'linacce_z']].values
    position = IMU_double_integration(time_stamp, rotations, accelerations)

    plt.figure()
    ax = plt.subplot(111, projection='3d')
    heading = int(position.shape[0] / 10)
    ax.plot(position[:heading, 0], position[:heading, 1], position[:heading, 2], 'r')
    ax.plot(position[heading:, 0], position[heading:, 1], position[heading:, 2], 'b')
    plt.show()
