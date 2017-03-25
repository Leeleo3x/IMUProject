import sys
import os
import argparse
import numpy as np
import quaternion
import matplotlib.pyplot as plt
import plyfile
import pandas
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/..')
from pre_processing import gen_dataset
from algorithms import geometry

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('dir', type=str)
    parser.add_argument('--skip', default=2000, type=int)
    args = parser.parse_args()

    rv_mag = 100

    gyro_input = np.genfromtxt(args.dir + '/gyro.txt')[args.skip:-args.skip]
    acce_input = np.genfromtxt(args.dir + '/acce.txt')[args.skip:-args.skip]
    linacce_input = np.genfromtxt(args.dir + '/linacce.txt')[args.skip:-args.skip]
    gravity_input = np.genfromtxt(args.dir + '/gravity.txt')[args.skip:-args.skip]
    rv_input = np.genfromtxt(args.dir + '/orientation.txt')[args.skip+rv_mag:-args.skip-rv_mag]

    rv_input[:, [1, 2, 3, 4]] = rv_input[:, [4, 1, 2, 3]]

    rv_output = np.copy(rv_input)
    imu_to_tango = geometry.quaternion_from_two_vectors(np.array([0, 0, -1]), np.array([0, 1, 0]))
    for i in range(rv_input.shape[0]):
        rv_tango = imu_to_tango * quaternion.quaternion(*rv_input[i, 1:])
        rv_output[i, 1:] = np.array([rv_tango.w, rv_tango.x, rv_tango.y, rv_tango.z])

    output_dir = args.dir + '/processed'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    output_timestamp = rv_input[:, 0]
    print('Processing gyro...')
    gyro_output = gen_dataset.interpolate_3dvector_linear(gyro_input, output_timestamp)
    print('Processing accelerometer...')
    acce_output = gen_dataset.interpolate_3dvector_linear(acce_input, output_timestamp)
    print('Processing linear acceleration...')
    linacce_output = gen_dataset.interpolate_3dvector_linear(linacce_input, output_timestamp)
    print('Processing gravity...')
    gravity_output = gen_dataset.interpolate_3dvector_linear(gravity_input, output_timestamp)

    fake_pose_data = np.zeros([rv_input.shape[0], 7], dtype=float)
    fake_pose_data[:, -4:] = rv_output[:, 1:]

    column_list = 'time,gyro_x,gyro_y,gyro_z,acce_x'.split(',') + \
                  'acce_y,acce_z,linacce_x,linacce_y,linacce_z,grav_x,grav_y,grav_z'.split(',') + \
                  'pos_x,pos_y,pos_z,ori_w,ori_x,ori_y,ori_z,rv_w,rv_x,rv_y,rv_z'.split(',')

    data_mat = np.concatenate([gyro_output,
                               acce_output[:, 1:],
                               linacce_output[:, 1:],
                               gravity_output[:, 1:],
                               fake_pose_data,
                               rv_input[:, 1:]], axis=1)

    data_csv = pandas.DataFrame(data_mat, columns=column_list)

    print('Writing csv...')
    data_csv.to_csv(output_dir + '/data.csv')

    print('Writing plain txt...')
    with open(output_dir + '/data_plain.txt', 'w') as f:
        f.write('{} {}\n'.format(data_mat.shape[0], data_mat.shape[1]))
        for i in range(data_mat.shape[0]):
            for j in range(data_mat.shape[1]):
                f.write('{}\t'.format(data_mat[i][j]))
            f.write('\n')
