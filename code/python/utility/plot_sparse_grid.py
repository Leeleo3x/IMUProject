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
    args = parser.parse_args()

    data_all = pandas.read_csv(args.dir + '/processed/data.csv')

    ts = data_all['time'].values / 1e09
    ts -= ts[0]

    window = 30

    linacce = gaussian_filter1d(data_all[['linacce_x', 'linacce_y', 'linacce_z']].values, sigma=30.0, axis=0)
    gravity = data_all[['grav_x', 'grav_y', 'grav_z']].values
    orientation = data_all[['ori_w', 'ori_x', 'ori_y', 'ori_z']].values

    speed_raw = compute_gravity_speed(ts, linacce, orientation, gravity)

    constraint = np.genfromtxt(args.dir + '/sparse_grid_constraint.txt')
    bias = np.genfromtxt(args.dir + '/sparse_grid_bias.txt')
    corrected_linacce = gaussian_filter1d(np.genfromtxt(args.dir + '/corrected_linacce.txt'), sigma=30.0, axis=0)

    speed_corrected = compute_gravity_speed(ts, corrected_linacce[:, 1:], orientation, gravity)

    bias_ind = bias[:, 0].astype(int)
    constraint_ind = constraint[:, 0].astype(int)

    font_config = {'family': 'serif',
                   'size': 60}
    linew = 8.0
    markersize = 30

    plt.rc('font', **font_config)
    fig = plt.figure('Sparse grid', figsize=(60, 12))

    end_id = bias_ind[window]

    plt.subplot(121, axis_bgcolor='black')
    plt.ylabel('Acceleration (m/s2)')
    plt.xlabel('Time (s)')
    plt.plot(ts[:end_id], linacce[:end_id, 2], color='r', lw=linew)
    plt.plot(ts[:end_id], corrected_linacce[:end_id, 3], color='b', lw=linew)
    plt.plot(ts[bias_ind[:window]], bias[:window, 3], color=(.5, .8, .0), lw=linew,
             marker='o', markersize=markersize)
    # plt.legend(['Sparse bias', 'Raw acceleration', 'Corrected acceleration'], loc='lower left')

    cw = 0
    for i in range(constraint_ind.shape[0]):
        cw = i
        if constraint_ind[i] > bias_ind[window]:
            break
    plt.subplot(122, axis_bgcolor='black')
    plt.ylabel('Velocity (m/s)')
    plt.xlabel('Time (s)')
    plt.plot(ts[:end_id], speed_raw[:end_id, 2], color='r', lw=linew)
    plt.plot(ts[:end_id], speed_corrected[:end_id, 2], color='b', lw=linew)
    plt.plot(ts[constraint_ind[:cw]], constraint[:cw, 3], color=(.8, .5, 0), lw=linew,
             marker='^', markersize=markersize)

    # plt.legend(['Predicted speed', 'Before correction', 'After correction'], loc='lower left')

    fig.savefig('/Users/yanhang/Documents/research/paper/iccv2017/8514476vhgjvhxssxgf/images/sparse_plot.png', bbox_inches='tight')
    # plt.show()