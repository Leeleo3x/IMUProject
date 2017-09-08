import numpy as np
import pandas
import argparse
from scipy import interpolate
import matplotlib.pyplot as plt
import quaternion

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/../')

from pre_processing import gen_dataset

nano_to_sec = 1e09

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('dir')

    print('Loading...')
    args = parser.parse_args()
    step_data = np.genfromtxt(args.dir + '/step.txt')
    step_data[:, 0] /= nano_to_sec

    data_all = pandas.read_csv(args.dir + '/processed/data.csv')

    ts = data_all['time'].values / nano_to_sec
    positions = data_all[['pos_x', 'pos_y', 'pos_z']].values
    orientations = data_all[['ori_w', 'ori_x', 'ori_y', 'ori_z']].values
    track_length = np.sum(np.linalg.norm(positions[1:] - positions[:-1], axis=1))
    stride = track_length / step_data[-1][1]
    print('Track length: {:.3f}m, stride length: {:.3f}m'.format(track_length, stride))

    ori_at_step = gen_dataset.interpolate_quaternion_linear(np.concatenate([ts[:, None], orientations], axis=1),
                                                            step_data[:, 0])
    pos_at_step = np.empty([step_data.shape[0], 3], dtype=float)
    pos_at_step[0] = positions[0]
    for i in range(1, pos_at_step.shape[0]):
        q = quaternion.quaternion(*ori_at_step[i, 1:])
        offset = np.array([0, 0, -stride * (step_data[i, 1] - step_data[i - 1, 1])])
        pos_at_step[i] = pos_at_step[i-1] + (q * quaternion.quaternion(1.0, *offset) * q.conj()).vec

    # insert one rwo at the beginning and ending
    step_ts = np.concatenate([[ts[0]-1], step_data[:, 0], [ts[-1]+1]], axis=0)
    pos_at_step = np.concatenate([[pos_at_step[0]], pos_at_step, [pos_at_step[-1]]], axis=0)
    pos_inte = gen_dataset.interpolate_3dvector_linear(np.concatenate([step_ts[:, None], pos_at_step], axis=1), ts)
    pos_inte = pos_inte[:, 1:]
    pos_inte[:, 2] = 0.0

    print('Writing to csv')
    data_mat = np.zeros([ts.shape[0], 10], dtype=float)
    column_list = ['time', 'pos_x', 'pos_y', 'pos_z', 'speed_x', 'speed_y', 'speed_z', 'bias_x', 'bias_y', 'bias_z']
    data_mat[:, 0] = ts
    data_mat[:, 1:4] = pos_inte

    data_pandas = pandas.DataFrame(data_mat, columns=column_list)
    data_pandas.to_csv(args.dir + '/result_step.csv')

    gen_dataset.write_ply_to_file(args.dir + '/result_trajectory_step.ply', pos_inte, orientations)
    print('All done')
