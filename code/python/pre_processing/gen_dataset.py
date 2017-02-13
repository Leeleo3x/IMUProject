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
from utility.write_trajectory_to_ply import write_ply_to_file


def analysis_error(pose_data):
    position = np.linalg.norm(pose_data[-1, 1:4] - pose_data[0, 1:4])
    q_start = quaternion.quaternion(pose_data[0, -4], pose_data[0, -3], pose_data[0, -2], pose_data[0, -1])
    q_end = quaternion.quaternion(pose_data[-1, -4], pose_data[-1, -3], pose_data[-1, -2], pose_data[-1, -1])
    q_diff = q_start.inverse() * q_end
    return position, 2 * math.acos(q_diff.w)


def interpolate_quaternion_spline(gyro_data, output_timestamp):
    # convert angular velocity to quaternion
    print('Using spline interpolation')
    N_input = gyro_data.shape[0]
    gyro_quat = np.empty([gyro_data.shape[0], 4], dtype=float)

    for i in range(N_input):
        record = gyro_data[i, 1:4]
        gyro_quat[i] = quaternion.as_float_array(quaternion.from_euler_angles(record[0], record[1], record[2]))

    gyro_interpolated = quaternion.quaternion_time_series.squad(quaternion.as_quat_array(gyro_quat),
                                                                gyro_data[:, 0], output_timestamp)
    return np.concatenate([output_timestamp[:, np.newaxis], quaternion.as_float_array(gyro_interpolated)], axis=1)


def interpolate_quaternion_linear(gyro_data, output_timestamp):
    print('Using linear interpolation')
    N_input = gyro_data.shape[0]
    N_output = output_timestamp.shape[0]

    quat_inter = np.zeros([N_output, 4])
    ptr1 = 0
    ptr2 = 0
    for i in range(N_output):
        if ptr1 >= N_input - 1 or ptr2 >= N_input:
            raise ValueError
        # Forward to the correct interval
        while gyro_data[ptr1 + 1, 0] <= output_timestamp[i]:
            ptr1 += 1
        while gyro_data[ptr2, 0] <= output_timestamp[i]:
            ptr2 += 1
        # assert gyro_data[ptr1, 0] <= output_timestamp[i] <= gyro_data[ptr2 , 0]
        # assert ptr2 - ptr1 <= 2
        q1 = quaternion.from_euler_angles(gyro_data[ptr1, 1], gyro_data[ptr1, 2], gyro_data[ptr1, 3])
        q2 = quaternion.from_euler_angles(gyro_data[ptr2, 1], gyro_data[ptr2, 2], gyro_data[ptr2, 3])
        quat_inter[i] = quaternion.as_float_array(quaternion.quaternion_time_series.
                                                  slerp(q1, q2, gyro_data[ptr1, 0],
                                                        gyro_data[ptr2, 0], output_timestamp[i]))
    return np.concatenate([output_timestamp[:, np.newaxis], quat_inter], axis=1)


def interpolate_rotation_vector_linear(rv_data, output_timestamp):
    rv_quaternion = np.empty([rv_data.shape[0], 4], dtype=float)
    for i in range(rv_data.shape[0]):
        mag = np.linalg.norm(rv_data[i, 1:])
        rv_quaternion[i, 1:] = rv_data[i, 1:] / mag
        rv_quaternion[i, 0] = math.sqrt(math.fabs(1 - mag ** 2))
    rv_quaternion = np.concatenate([rv_data[:, 0][:, None], rv_quaternion], axis=1)
    return interpolate_quaternion_linear(rv_quaternion, output_timestamp)


def interpolate_3dvector_linear(input, output_timestamp):
    func = scipy.interpolate.interp1d(input[:, 0], input[:, 1:], axis=0)
    interpolated = func(output_timestamp)
    return np.concatenate([output_timestamp[:, np.newaxis], interpolated], axis=1)


def write_file(path, data, header=''):
    with open(path, 'w') as f:
        if len(header) > 0:
            f.write(header + '\n')
        for row in data:
            f.write('{:.0f}'.format(row[0]))
            for j in range(data.shape[1] - 1):
                f.write(' {:.6f}'.format(row[j+1]))
            f.write('\n')


def test_eular_to_quaternion(eular_input):
    eular_ret = np.empty(eular_input.shape)
    for i in range(eular_input.shape[0]):
        q = quaternion.from_euler_angles(eular_input[i, 0], eular_input[i, 1], eular_input[i, 2])
        eular_ret[i, :] = quaternion.as_euler_angles(q)
    return eular_ret


def rotation_matrix_from_two_vectors(v1, v2):
    """
    Using Rodrigues rotationformula
    https://en.wikipedia.org/wiki/Rodrigues%27_rotation_formula
    :param v1: starting vector
    :param v2: ending vector
    :return 3x3 rotation matrix
    """
    v1 /= np.linalg.norm(v1)
    v2 /= np.linalg.norm(v2)
    theta = np.dot(v1, v2)
    if theta == 1:
        return np.identity(3)
    if theta == -1:
        raise ValueError
    k = np.cross(v1, v2)
    k /= np.linalg.norm(k)
    K = np.matrix([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.identity(3) + math.sqrt(1 - theta * theta) * K + np.dot((1 - theta) * K * K, v1)


def quaternion_from_two_vectors(v1, v2):
    """
    Compute quaternion from two vectors
    :param v1:
    :param v2:
    :return Quaternion representation of rotation between v1 and v2
    """
    v1n = v1 / np.linalg.norm(v1)
    v2n = v2 / np.linalg.norm(v2)
    w = np.cross(v1n, v2n)
    q = np.array([1.0 + np.dot(v1n, v2n), *w])
    q /= np.linalg.norm(q)
    return quaternion.quaternion(*q)


def align_with_gravity(poses, gravity, local_g_direction=np.array([0, 0, -1])):
    """
    Adjust pose such that the gravity is at $target$ direction
    @:param poses: N x 7 array, each row is position + orientation (quaternion). The array will be modified in place.
    @:param gravity: real gravity direction
    @:param local_g_direction: z direction before alignment
    @:return None.
    """
    assert poses.ndim == 2, 'Expect 2 dimensional array input'
    assert poses.shape[1] == 7, 'Expect Nx7 array input'
    rotor = quaternion_from_two_vectors(local_g_direction, gravity)
    for pose in poses:
        distance = np.linalg.norm(pose[0:3])
        position_n = pose[0:3] / distance
        pose[0:3] = distance * (rotor * quaternion.quaternion(0.0, *position_n) * rotor.conjugate()).vec
        pose[-4:] = quaternion.as_float_array(rotor * quaternion.quaternion(*pose[-4:]) * rotor.conjugate())


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('list')
    parser.add_argument('--skip', default=100)
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
            pose_data = np.genfromtxt(data_root+'/pose.txt')
            # swap tango's orientation from [x,y,z,w] to [w,x,y,z]
            pose_data[:, [-4, -3, -2, -1]] = pose_data[:, [-1, -4, -3, -2]]

            # drop the head
            acce_data = np.genfromtxt(data_root+'/acce.txt')[args.skip:]
            print('Acceleration found. Sample rate:{:2f} Hz'
                  .format((acce_data.shape[0] - 1.0) * nano_to_sec / (acce_data[-1, 0] - acce_data[0, 0])))
            gyro_data = np.genfromtxt(data_root+'/gyro.txt')[args.skip:]
            print('Gyroscope found. Sample rate:{:2f} Hz'
                  .format((gyro_data.shape[0] - 1.0) * nano_to_sec/ (gyro_data[-1, 0] - gyro_data[0, 0])))
            linacce_data = np.genfromtxt(data_root+'/linacce.txt')[args.skip:]
            print('Linear acceleration found. Sample rate:{:2f} Hz'
                  .format((linacce_data.shape[0] - 1.0) * nano_to_sec / (linacce_data[-1, 0] - linacce_data[0, 0])))
            gravity_data = np.genfromtxt(data_root+'/gravity.txt')[args.skip:]
            print('Gravity found. Sample rate:{:2f} Hz'
                  .format((gravity_data.shape[0] - 1.0) * nano_to_sec / (gravity_data[-1, 0] - gravity_data[0, 0])))

            # Error analysis
            position_error, angular_error = analysis_error(pose_data)
            print('Positional error: {:.6f}(m), angular error: {:.6f}(rad)'.format(position_error, angular_error))

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

            # construct a Pandas DataFrame
            column_list = 'time,gyro_x,gyro_y,gyro_z,acce_x'.split(',') + \
                          'acce_y,acce_z,linacce_x,linacce_y,linacce_z,grav_x,grav_y,grav_z'.split(',') + \
                          'pos_x,pos_y,pos_z,ori_w,ori_x,ori_y,ori_z'.split(',')

            data_mat = np.concatenate([output_gyro_linear,
                                       output_accelerometer_linear[:, 1:],
                                       output_linacce_linear[:, 1:],
                                       output_gravity_linear[:, 1:],
                                       pose_data[:, 1:4],
                                       pose_data[:, -4:]], axis=1)

            # write individual files for convenience
            write_file(output_folder + '/output_gyro_linear.txt', output_gyro_linear)
            write_file(output_folder + '/linacce_linear.txt', output_linacce_linear)
            write_file(output_folder + '/gravity_linear.txt', output_gravity_linear)
            write_file(output_folder + '/acce_linear.txt', output_accelerometer_linear)

            # if the dataset comes with rotation vector, include it
            output_orientation = None
            if os.path.exists(data_root + '/orientation.txt'):
                orientation_data = np.genfromtxt(data_root + '/orientation.txt')[args.skip:]
                print('Orientation found. Sample rate:{:2f}'
                      .format((orientation_data.shape[0] - 1.0) * nano_to_sec / (orientation_data[-1, 0] - orientation_data[0, 0])))
                # Convert rotation vector to quaternion
                output_orientation = interpolate_rotation_vector_linear(orientation_data, output_timestamp)
                write_file(output_folder + '/output_orientation_linear.txt', output_orientation)
                data_mat = np.concatenate([data_mat, output_orientation[:, 1:]], axis=1)
                column_list += 'rv_w,rv_x,rv_y,rv_z'.split(',')

            data_pandas = pandas.DataFrame(data_mat, columns=column_list)

            data_pandas.to_csv(output_folder + '/data.csv')
            print('Dataset written to ' + output_folder + '/data.txt')

            if not args.no_trajectory:
                print("Writing trajectory to ply file")
                # write_ply_to_file(path=output_folder + '/trajectory.ply', position=pose_data[:, 1:4],
                #                   orientation=pose_data[:, -4:])
                write_ply_to_file(path=output_folder + '/trajectory_rv.ply', position=pose_data[:, 1:4],
                                  orientation=output_orientation[:, 1:])

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
