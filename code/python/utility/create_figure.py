import sys
import os
sys.path.append('/home/yanhang/Documents/research/IMUProject/code/python')
sys.path.append('/Users/yanhang/Documents/research/IMUProject/code/python')

import numpy as np
import matplotlib.pyplot as plt
import pandas
import argparse
import quaternion
from scipy.ndimage.filters import gaussian_filter1d

from speed_regression import training_data as td

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('dir')

    filter_sigma = 20.0

    args = parser.parse_args()
    output_path = args.dir + '/figure'

    if not os.path.exists(output_path):
        os.makedirs(output_path)

    result = pandas.read_csv(args.dir + '/result.csv')

    data_all = pandas.read_csv(args.dir + '/processed/data.csv')

    regression_result = np.genfromtxt(args.dir + '/regression.txt')
    constraint_ind = regression_result[:, 0].astype(int)
    local_speed = regression_result[:, 1:]

    ts = result['time'].values
    ts -= ts[0]

    pos_res = result[['pos_x', 'pos_y', 'pos_z']].values
    speed_res = result[['speed_x', 'speed_y', 'speed_z']].values
    bias_res = result[['bias_x', 'bias_y', 'bias_z']].values

    speed_res = gaussian_filter1d(speed_res, sigma=filter_sigma, axis=0)

    position_gt = data_all[['pos_x', 'pos_y', 'pos_z']].values

    traj_length = sum(np.linalg.norm(position_gt[1:] - position_gt[:-1], axis=1))
    mean_offset = np.average(np.linalg.norm(position_gt - pos_res, axis=1))
    print('Trajectory length: {:.3f}m, {:.3f}s, mean offset: {:.3f}m ({:.3f})'.format(traj_length, ts[-1] - ts[0],
          mean_offset, mean_offset / traj_length))

    orientation_gt = data_all[['ori_w', 'ori_x', 'ori_y', 'ori_z']].values
    gravity = data_all[['grav_x', 'grav_y', 'grav_z']].values

    speed_gt = td.compute_speed(ts, position_gt)
    ls_gt = td.compute_local_speed_with_gravity(ts, position_gt, orientation_gt, gravity)
    ls_gt = gaussian_filter1d(ls_gt, sigma=30.0, axis=0)
    ls_gt = ls_gt[constraint_ind]

    speed_gt = gaussian_filter1d(speed_gt, sigma=filter_sigma, axis=0)

    linacce = data_all[['linacce_x', 'linacce_y', 'linacce_z']].values
    directed_linacce = np.empty(linacce.shape, dtype=float)
    for i in range(linacce.shape[0]):
        q = quaternion.quaternion(*orientation_gt[i])
        directed_linacce[i] = (q * quaternion.quaternion(1.0, *linacce[i]) * q.conj()).vec
    speed_raw = np.cumsum(directed_linacce[:-1] * (ts[1:, None] - ts[:-1, None]), axis=0)
    # speed_raw = np.concatenate([np.zeros(1, 3), speed_raw], axis=0)

    legends = [['IMU-x', 'Tango-x'], ['IMU-y', 'Tango-y']]
    legend_speed = [['IMU-x', 'Before-x', 'After-x'],
                    ['IMU-z', 'Before-z', 'After-z']]
    axis_name_global = ['Xw', 'Yw']
    axis_name_grav = ['Xg', 'Zg']
    axis_name_local = ['Xl', 'Yl', 'Zl']

    axes_glob = [0, 1]
    axes_local = [0, 2]

    lines_imu = []
    lines_raw = []
    lines_tango = []

    fig_gs = plt.figure('Speed', figsize=(10, 8))
    for i in range(0, 2):
        plt.subplot(211 + i)
        lines_imu += plt.plot(ts, speed_res[:, axes_glob[i]])
        lines_raw += plt.plot(ts[1:], speed_raw[:, axes_glob[i]])
        lines_tango += plt.plot(ts, speed_gt[:, axes_glob[i]])
    plt.figlegend([lines_imu[-1], lines_raw[-1], lines_tango[-1]],
                  {'Our method', 'Double integration', 'Tango'},
                  loc='lower center', ncol=3, labelspacing=0.)
    fig_gs.savefig(output_path + '/fig_gs.png')

    fig_ls = plt.figure('Local speed', figsize=(10, 8))
    for i in range(0, 2):
        plt.subplot(211+i)
        lines_imu += plt.plot(ts[constraint_ind], local_speed[:, axes_local[i]])
        lines_tango += plt.plot(ts[constraint_ind], ls_gt[:, axes_local[i]])
    plt.figlegend([lines_imu[-1], lines_raw[-1], lines_tango[-1]],
                  {'Our method', 'Tango (Ground truth)'},
                  loc='lower center', ncol=2, labelspacing=0.)
    fig_ls.savefig(output_path + '/fig_ls.png')

    fig_bias = plt.figure('Bias', figsize=(10, 8))
    for i in range(0, 3):
        plt.subplot(311+i)
        plt.plot(ts, bias_res[:, i])
        plt.legend(axis_name_local[i], loc='lower right')
    fig_bias.savefig(output_path + '/fig_bias.png')
    # plt.show()