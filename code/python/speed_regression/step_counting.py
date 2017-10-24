import numpy as np
import pandas
import argparse
from scipy import interpolate
import matplotlib.pyplot as plt
import quaternion

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/../')

from algorithms import icp
from pre_processing import gen_dataset
from utility import write_trajectory_to_ply

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
    # orientations = data_all[['ori_w', 'ori_x', 'ori_y', 'ori_z']].values
    orientations = data_all[['rv_w', 'rv_x', 'rv_y', 'rv_z']].values
    ori_tango = data_all[['ori_w', 'ori_x', 'ori_y', 'ori_z']].values
    track_length = np.sum(np.linalg.norm(positions[1:] - positions[:-1], axis=1))
    step_stride = track_length / step_data[-1][1]
    print('Track length: {:.3f}m, stride length: {:.3f}m'.format(track_length, step_stride))

    # Interpolate step counts to Tango's sample rate.
    step_data = np.concatenate([np.array([[ts[0] - 1, 0]]), step_data, np.array([[ts[-1] + 1, step_data[-1][1]]])],
                               axis=0)
    step_func = interpolate.interp1d(step_data[:, 0], step_data[:, 1])
    step_interpolated = step_func(ts)

    # Compute positions by dead-reckoning
    step_length = step_stride * (step_interpolated[1:] - step_interpolated[:-1])
    position_from_step = np.zeros(positions.shape)
    position_from_step[0] = positions[0]
    forward_dir = quaternion.quaternion(0., 0., 0., -1.)
    for i in range(1, positions.shape[0]):
        q = quaternion.quaternion(*orientations[i - 1])
        segment = (q * forward_dir * q.conj()).vec * step_length[i - 1]
        position_from_step[i] = position_from_step[i-1] + segment

    # Find a 2D rotation transformation to align the start portion of estimated track and the ground truth track.
    start_length = 3000
    _, rotation_to_gt, translation_to_gt = icp.fit_transformation(position_from_step, positions)
    position_from_step = (np.dot(rotation_to_gt, position_from_step.T) + translation_to_gt).T

    _, rotation_2d, translation_2d = icp.fit_transformation(position_from_step[:start_length, :2],
                                                            positions[:start_length, 2])
    position_from_step[:, 2] = (np.dot(rotation_2d, position_from_step[:, :2].T) + translation_2d).T

    rotation_combined = np.identity(3)
    rotation_combined[:2, :2] = rotation_2d
    rotatioin_combined = rotation_combined * rotation_to_gt
    output_orientations = orientations[:]
    for i in range(orientations.shape[0]):
        rotor = quaternion.from_rotation_matrix(rotation_combined)
        out_orientation = rotor * quaternion.quaternion(*orientations[i]) * rotor.conj()
        output_orientations[i] = quaternion.as_float_array(rotor)



    print('Writing to csv')
    data_mat = np.zeros([ts.shape[0], 10], dtype=float)
    column_list = ['time', 'pos_x', 'pos_y', 'pos_z', 'speed_x', 'speed_y', 'speed_z', 'bias_x', 'bias_y', 'bias_z']
    data_mat[:, 0] = ts
    data_mat[:, 1:4] = position_from_step

    out_dir = args.dir + '/result_step/'
    if not os.path.isdir(out_dir):
        os.makedirs(out_dir)
    data_pandas = pandas.DataFrame(data_mat, columns=column_list)
    data_pandas.to_csv(out_dir + '/result_step.csv')

    write_trajectory_to_ply.write_ply_to_file(out_dir + '/result_trajectory_step.ply', position_from_step, orientations,
                                              trajectory_color=(0, 150, 150), length=0.8, interval=300, num_axis=1)
    print('All done')
