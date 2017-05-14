#pylint: disable=C0103,C0111,C0301

"""
Modification 02/02/17
1. Remove 'adjustAxis'. Now the imu coordinate frame is the same with tango device frame
2. Use Eular angle for gyro data. Interpolate linearly
3. Swap indices of Tango orientation from [x,y,z,w] to [w,x,y,z]
"""
import sys
import os
import math
import argparse
import numpy as np
import scipy.interpolate
import quaternion
import quaternion.quaternion_time_series
import matplotlib.pyplot as plt
import plyfile
import pandas
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/..')
from algorithms import geometry
from utility.write_trajectory_to_ply import write_ply_to_file


def analysis_error(pose_data):
    position = np.linalg.norm(pose_data[-1, 1:4] - pose_data[0, 1:4])
    q_start = quaternion.quaternion(pose_data[0, -4], pose_data[0, -3], pose_data[0, -2], pose_data[0, -1])
    q_end = quaternion.quaternion(pose_data[-1, -4], pose_data[-1, -3], pose_data[-1, -2], pose_data[-1, -1])
    q_diff = q_start.inverse() * q_end
    return position, 2 * math.acos(q_diff.w)


def interpolate_quaternion_spline(gyro_data, output_timestamp):
    # convert angular velocity to quaternion
    N_input = gyro_data.shape[0]
    gyro_quat = np.empty([gyro_data.shape[0], 4], dtype=float)

    for i in range(N_input):
        record = gyro_data[i, 1:4]
        gyro_quat[i] = quaternion.as_float_array(quaternion.from_euler_angles(record[0], record[1], record[2]))

    gyro_interpolated = quaternion.quaternion_time_series.squad(quaternion.as_quat_array(gyro_quat),
                                                                gyro_data[:, 0], output_timestamp)
    return np.concatenate([output_timestamp[:, np.newaxis], quaternion.as_float_array(gyro_interpolated)], axis=1)


def interpolate_quaternion_linear(quat_data, output_timestamp):
    N_input = quat_data.shape[0]
    N_output = output_timestamp.shape[0]

    quat_inter = np.zeros([N_output, 4])
    ptr1 = 0
    ptr2 = 0
    for i in range(N_output):
        if ptr1 >= N_input - 1 or ptr2 >= N_input:
            raise ValueError
        # Forward to the correct interval
        while quat_data[ptr1 + 1, 0] < output_timestamp[i]:
            ptr1 += 1
            if ptr1 == N_input - 1:
                break
        while quat_data[ptr2, 0] < output_timestamp[i]:
            ptr2 += 1
            if ptr2 == N_input:
                break
        if quat_data.shape[1] == 4:
            q1 = quaternion.from_euler_angles(*quat_data[ptr1, 1:])
            q2 = quaternion.from_euler_angles(*quat_data[ptr2, 1:])
        else:
            q1 = quaternion.quaternion(*quat_data[ptr1, 1:])
            q2 = quaternion.quaternion(*quat_data[ptr2, 1:])
        quat_inter[i] = quaternion.as_float_array(quaternion.quaternion_time_series.
                                                  slerp(q1, q2, quat_data[ptr1, 0],
                                                        quat_data[ptr2, 0], output_timestamp[i]))
    return np.concatenate([output_timestamp[:, np.newaxis], quat_inter], axis=1)


def interpolate_3dvector_linear(input, output_timestamp):
    func = scipy.interpolate.interp1d(input[:, 0], input[:, 1:], axis=0)
    interpolated = func(output_timestamp)
    return np.concatenate([output_timestamp[:, np.newaxis], interpolated], axis=1)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('list')
    parser.add_argument('--skip', default=400)
    parser.add_argument('--recompute', action='store_true')
    parser.add_argument('--no_trajectory', action='store_true')

    args = parser.parse_args()

    with open(args.list) as f:
        dataset_list = [s.strip('\n') for s in f.readlines()]

    print(dataset_list)
    root_dir = os.path.dirname(args.list)

    nano_to_sec = 1000000000.0
    total_length = 0.0
    length_dict = {}
    for dataset in dataset_list:
        if len(dataset.strip()) == 0:
            continue
        if dataset[0] == '#':
            continue
        info = dataset.split(',')
        motion_type = 'unknown'
        if len(info) == 2:
            motion_type = info[1]
        data_root = root_dir + '/' + info[0]
        length = 0
        if os.path.exists(data_root + '/processed/data.csv') and not args.recompute:
            data_pandas = pandas.read_csv(data_root + '/processed/data.csv')
        else:
            print('------------------\nProcessing ' + data_root, ', type: ' + motion_type)
            pose_data = np.genfromtxt(data_root+'/pose.txt')[args.skip:-args.skip]
            # swap tango's orientation from [x,y,z,w] to [w,x,y,z]
            pose_data[:, [-4, -3, -2, -1]] = pose_data[:, [-1, -4, -3, -2]]

            # drop the head
            acce_data = np.genfromtxt(data_root+'/acce.txt')[args.skip:]
            print('Acceleration found. Sample rate:{:2f} Hz'
                  .format((acce_data.shape[0] - 1.0) * nano_to_sec / (acce_data[-1, 0] - acce_data[0, 0])))
            gyro_data = np.genfromtxt(data_root+'/gyro.txt')[args.skip:]
            print('Gyroscope found. Sample rate:{:2f} Hz'
                  .format((gyro_data.shape[0] - 1.0) * nano_to_sec / (gyro_data[-1, 0] - gyro_data[0, 0])))
            linacce_data = np.genfromtxt(data_root+'/linacce.txt')[args.skip:]
            print('Linear acceleration found. Sample rate:{:2f} Hz'
                  .format((linacce_data.shape[0] - 1.0) * nano_to_sec / (linacce_data[-1, 0] - linacce_data[0, 0])))
            gravity_data = np.genfromtxt(data_root+'/gravity.txt')[args.skip:]
            print('Gravity found. Sample rate:{:2f} Hz'
                  .format((gravity_data.shape[0] - 1.0) * nano_to_sec / (gravity_data[-1, 0] - gravity_data[0, 0])))

            magnet_data = np.genfromtxt(data_root + '/magnet.txt')
            print('Magnetometer: {:.2f}Hz'.
                  format((magnet_data.shape[0] - 1.0) * nano_to_sec / (magnet_data[-1, 0] - magnet_data[0, 0])))

            orientation_data = np.genfromtxt(data_root + '/orientation.txt')[args.skip:]
            print('Orientation found. Sample rate:{:2f}'
                  .format((orientation_data.shape[0] - 1.0) * nano_to_sec /
                          (orientation_data[-1, 0] - orientation_data[0, 0])))

            output_folder = data_root + '/processed'
            if not os.path.isdir(output_folder):
                os.makedirs(output_folder)

            # Generate dataset
            output_timestamp = pose_data[:, 0]

            # output_gyro_linear = interpolateAngularRateLinear(gyro_data, output_timestamp)
            output_gyro_linear = interpolate_3dvector_linear(gyro_data, output_timestamp)
            output_accelerometer_linear = interpolate_3dvector_linear(acce_data, output_timestamp)
            output_linacce_linear = interpolate_3dvector_linear(linacce_data, output_timestamp)
            output_gravity_linear = interpolate_3dvector_linear(gravity_data, output_timestamp)
            output_magnet_linear = interpolate_3dvector_linear(magnet_data, output_timestamp)

            # convert gyro, accelerometer and linear acceleration to stablized IMU frame
            gyro_stab = geometry.align_eular_rotation_with_gravity(output_gyro_linear[:, 1:],
                                                                   output_gravity_linear[:, 1:])
            acce_stab = geometry.align_3dvector_with_gravity(output_accelerometer_linear[:, 1:],
                                                             output_gravity_linear[:, 1:])
            linacce_stab = geometry.align_3dvector_with_gravity(output_linacce_linear[:, 1:],
                                                                output_gravity_linear[:, 1:])

            # swap from x,y,z,w to w,x,y,z
            orientation_data[:, [1, 2, 3, 4]] = orientation_data[:, [4, 1, 2, 3]]
            # Convert rotation vector to quaternion
            output_orientation = interpolate_quaternion_linear(orientation_data, output_timestamp)

            # construct a Pandas DataFrame
            column_list = 'time,gyro_x,gyro_y,gyro_z,acce_x'.split(',') + \
                          'acce_y,acce_z,linacce_x,linacce_y,linacce_z,grav_x,grav_y,grav_z'.split(',') + \
                          'magnet_x,magnet_y,magnet_z'.split(',') + \
                          'pos_x,pos_y,pos_z,ori_w,ori_x,ori_y,ori_z,rv_w,rv_x,rv_y,rv_z'.split(',') + \
                          'gyro_stab_x,gyro_stab_y,gyro_stab_z,acce_stab_x,acce_stab_y,acce_stab_z'.split(',') + \
                          'linacce_stab_x,linacce_stab_y,linacce_stab_z'.split(',')

            data_mat = np.concatenate([output_gyro_linear,
                                       output_accelerometer_linear[:, 1:],
                                       output_linacce_linear[:, 1:],
                                       output_gravity_linear[:, 1:],
                                       output_magnet_linear[:, 1:],
                                       pose_data[:, 1:4],
                                       pose_data[:, -4:],
                                       output_orientation[:, 1:],
                                       gyro_stab, acce_stab, linacce_stab], axis=1)

            # write individual files for convenience

            # if the dataset comes with rotation vector, include it
            data_pandas = pandas.DataFrame(data_mat, columns=column_list)

            data_pandas.to_csv(output_folder + '/data.csv')
            print('Dataset written to ' + output_folder + '/data.txt')

            # write data in plain text file for C++
            with open(output_folder + '/data_plain.txt', 'w') as f:
                f.write('{} {}\n'.format(data_mat.shape[0], data_mat.shape[1]))
                for i in range(data_mat.shape[0]):
                    for j in range(data_mat.shape[1]):
                        f.write('{}\t'.format(data_mat[i][j]))
                    f.write('\n')

            if not args.no_trajectory:
                print("Writing trajectory to ply file")
                viewing_dir = np.zeros([data_mat.shape[0], 3], dtype=float)
                viewing_dir[:, 2] = -1.0
                write_ply_to_file(path=output_folder + '/trajectory.ply', position=pose_data[:, 1:4],
                                  orientation=pose_data[:, -4:], acceleration=output_gravity_linear[:, 1:],
                                  length=5.0, kpoints=100, interval=200)

                q_device_tango = quaternion.quaternion(*pose_data[0, -4:])
                # q_device_tango = quaternion.quaternion(1., 0., 0., 0.)
                q_rv_tango = q_device_tango * quaternion.quaternion(*output_orientation[0, 1:]).inverse()
                orientation_tango_frame = np.empty([output_orientation.shape[0], 4], dtype=float)
                for i in range(orientation_tango_frame.shape[0]):
                    orientation_tango_frame[i] = quaternion.as_float_array(q_rv_tango *
                                                                           quaternion.quaternion(*output_orientation[i, 1:]))
                write_ply_to_file(output_folder + '/trajectory_rv.ply', position=pose_data[:, -7:-4],
                                  acceleration=output_gravity_linear[:, 1:],
                                  orientation=orientation_tango_frame, length=5.0, kpoints=100, interval=200)

        length = (data_pandas['time'].values[-1] - data_pandas['time'].values[0]) / nano_to_sec
        hertz = data_pandas.shape[0] / length
        print(info[0] + ', length: {}s, sample rate: {:.2f}Hz'.format(length, hertz))
        if motion_type not in length_dict:
            length_dict[motion_type] = length
        else:
            length_dict[motion_type] += length
        total_length += length

    print('All done. Total length: {:.2f}s ({:.2f}min)'.format(total_length, total_length / 60.0))
    for k, v in length_dict.items():
        print(k + ': {:.2f}s ({:.2f}min)'.format(v, v / 60.0))
