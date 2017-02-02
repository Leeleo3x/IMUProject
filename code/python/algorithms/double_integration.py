import numpy as np
import quaternion
import scipy.integrate as integrate
import matplotlib.pyplot as plt
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
    positions_gt = data_all[['pos_x', 'pos_y', 'pos_z']].values

    linacce = data_all[['linacce_x', 'linacce_y', 'linacce_z']].values
    positions = IMU_double_integration(t=time_stamp, rotation=rotations, acceleration=linacce)
    plt.figure()

    rot_vec_sample = np.arange(0, positions.shape[0], 100, dtype=np.int)
    quat_array = quaternion.as_quat_array(rotations[rot_vec_sample])

    # array used for visualize oritation
    orientation_sampled = np.empty([rot_vec_sample.shape[0], 3], dtype=np.float)
    for i in range(rot_vec_sample.shape[0]):
        q = quat_array[i]
        rotated = q * quaternion.quaternion(0, 0, 0, -1) * q.conj()
        orientation_sampled[i, :] = rotated.vec
    orientation_position = positions_gt[rot_vec_sample]

    ax = plt.subplot(111, projection='3d')
    heading = int(positions.shape[0] / 10)
    # ax.plot(positions[:heading, 0], positions[:heading, 1], 'r')
    # ax.plot(positions[heading:, 0], positions[heading:, 1], 'b')

    positions_gt_sampled = positions_gt[rot_vec_sample]
    ax.plot(positions_gt[:, 0], positions_gt[:, 1], positions_gt[:, 2], 'g')
    ax.quiver(positions_gt_sampled[:, 0], positions_gt_sampled[:, 1], positions_gt_sampled[:, 2],
              orientation_sampled[:, 0], orientation_sampled[:, 1], orientation_sampled[:, 2],
              length=0.1)
    plt.show()

