import numpy as np
import quaternion
import argparse
import pandas
import matplotlib.pyplot as plt
from scipy.ndimage.filters import gaussian_filter1d

import sys
sys.path.append('/Users/yanhang/Documents/research/IMUProject/code/python')
sys.path.append('/home/yanhang/Documents/research/IMUProject/code/python')

from algorithms import geometry


def compute_gravity_speed(ts, linacce, orientation, gravity):
    directed = np.empty(linacce.shape, dtype=float)
    for i in range(directed.shape[0]):
        q = quaternion.quaternion(*orientation[i])
        directed[i] = (q * quaternion.quaternion(1.0, *linacce[i]) * q.conj()).vec
    speed = np.cumsum(directed[:-1] * (ts[1:, None] - ts[:-1, None]), axis=0)
    speed = np.concatenate([np.array([[0., 0., 0.]]), speed], axis=0)
    local_grav_dir = np.array([0., 1., 0.])
    for i in range(speed.shape[0]):
        q = quaternion.quaternion(*orientation[i])
        rotor = geometry.quaternion_from_two_vectors(gravity[i], local_grav_dir)
        ls = q.conj() * quaternion.quaternion(1.0, *speed[i]) * q
        speed[i] = (rotor * ls * rotor.conj()).vec
    return speed


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('dir', type=str)
    parser.add_argument('--id', type=str, default='regress')
    args = parser.parse_args()

    data_all = pandas.read_csv(args.dir + '/processed/data.csv')

    ts = data_all['time'].values / 1e09
    ts -= ts[0]

    bias_start_id = 10
    bias_end_id = 15

    linacce = gaussian_filter1d(data_all[['linacce_x', 'linacce_y', 'linacce_z']].values, sigma=30.0, axis=0)
    gravity = data_all[['grav_x', 'grav_y', 'grav_z']].values
    orientation = data_all[['ori_w', 'ori_x', 'ori_y', 'ori_z']].values

    speed_raw = compute_gravity_speed(ts, linacce, orientation, gravity)

    constraint = np.genfromtxt(args.dir + '/result_' + args.id + '/sparse_grid_constraint.txt')
    bias = np.genfromtxt(args.dir + '/result_' + args.id + '/sparse_grid_bias.txt')
    corrected_linacce = gaussian_filter1d(np.genfromtxt(args.dir + '/result_' + args.id + '/corrected_linacce.txt'),
                                          sigma=30.0, axis=0)

    speed_corrected = compute_gravity_speed(ts, corrected_linacce[:, 1:], orientation, gravity)

    bias_ind = bias[:, 0].astype(int)
    constraint_ind = constraint[:, 0].astype(int)

    font_config = {'family': 'serif',
                   'size': 18}
    linew = 1.5
    markersize = 15

    plt.rc('font', **font_config)
    fig = plt.figure('Sparse grid', figsize=(20, 5))

    glob_start_id = bias_ind[bias_start_id]
    glob_end_id = bias_ind[bias_end_id]

    bias_end_id = min(bias_ind.shape[0] - 1, bias_end_id + 1)

    # First draw the plot for low-frequency bias.
    plt.subplot(121)
    plt.ylabel('Acceleration (m/s2)')
    plt.xlabel('Time (s)')
    plt.plot(ts[glob_start_id:glob_end_id], linacce[glob_start_id:glob_end_id, 2], color='r', lw=linew)
    plt.plot(ts[glob_start_id:glob_end_id], corrected_linacce[glob_start_id:glob_end_id, 3], color='b', lw=linew)
    plt.plot(ts[bias_ind[bias_start_id:bias_end_id]], bias[bias_start_id:bias_end_id, 3], color=(.5, .8, .0),
             lw=linew, marker='o', markersize=markersize)
    # plt.legend(['Sparse bias', 'Raw acceleration', 'Corrected acceleration'], loc='lower left')

    # Then draw the plot for velocity constraints.
    con_start_id = np.where(constraint_ind > glob_start_id)[0][0] - 1
    con_end_id = np.where(constraint_ind > glob_end_id)[0][0] + 1

    glob_start_id = constraint_ind[con_start_id]
    glob_end_id = constraint_ind[con_end_id]

    plt.subplot(122)
    plt.ylabel('Velocity (m/s)')
    plt.xlabel('Time (s)')
    # plt.xlim(ts[glob_start_id], ts[glob_end_id])
    plt.plot(ts[glob_start_id:glob_end_id], speed_raw[glob_start_id:glob_end_id, 2], color=(0., 0., 0.), lw=linew)
    plt.plot(ts[glob_start_id:glob_end_id], speed_corrected[glob_start_id:glob_end_id, 2], color='b', lw=linew)
    plt.plot(ts[constraint_ind[con_start_id:con_end_id]], constraint[con_start_id:con_end_id, 3], color=(.8, .5, 0),
             lw=0, marker='^', markersize=markersize)

    # plt.legend(['Predicted speed', 'Before correction', 'After correction'], loc='lower left')

    # fig.savefig('/Users/yanhang/Documents/research/paper/iccv2017_supplementary/presentation/alg1.png', bbox_inches='tight')
    plt.show()
