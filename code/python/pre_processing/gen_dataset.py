#pylint: disable=C0103,C0111,C0301

import math
import argparse
import os
from datetime import datetime

import numpy as np
import scipy.interpolate
import quaternion
import quaternion.quaternion_time_series
import matplotlib.pyplot as plt
import plyfile
import pandas


def computeIntervalVariance(input, N=100, step=10):
    range_list = [np.var(input[s:s+N, :], axis=0) for s in range(0, input.shape[0]-N, step)]
    plt.plot(np.linalg.norm(range_list))


# for Tango development kit
def adjustAxis(input_data):
    # first swap y and z
    input_data[:, [2, 3]] = input_data[:, [3, 2]]
    # invert x, z axis
    input_data[:, [1, 3]] *= -1


def extractGravity(acce):
    """Extract gravity from accelerometer"""
    pass


def analysisError(pose_data):
    position = np.linalg.norm(pose_data[-1, 1:4] - pose_data[0, 1:4])
    q_start = quaternion.quaternion(pose_data[0, -4], pose_data[0, -3], pose_data[0, -2], pose_data[0, -1])
    q_end = quaternion.quaternion(pose_data[-1, -4], pose_data[-1, -3], pose_data[-1, -2], pose_data[-1, -1])
    q_diff = q_start.inverse() * q_end
    return position, 2 * math.acos(q_diff.w)


def interpolateAngularRateSpline(gyro_data, output_timestamp):
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


def interpolateAngularRateLinear(gyro_data, output_timestamp):
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


def interpolate3DVectorLinear(input, output_timestamp):
    func = scipy.interpolate.interp1d(input[:, 0], input[:, 1:], axis=0)
    interpolated = func(output_timestamp)
    return np.concatenate([output_timestamp[:, np.newaxis], interpolated], axis=1)


def writeFile(path, data, header=''):
    with open(path, 'w') as f:
        if len(header) > 0:
            f.write(header + '\n')
        for row in data:
            f.write('{:.0f}'.format(row[0]))
            for j in range(data.shape[1] - 1):
                f.write(' {:.6f}'.format(row[j+1]))
            f.write('\n')


def writeTrajectoryToPly(path, positions, gravity=np.array([0, 0, 0])):
    """
    Write camera poses to ply file.
    :param path: File path
    :param positions: N x 3 array of positions
    :param gravity: gravity direction
    :return: None
    """
    assert gravity.ndim == 1, 'Gravity should be a 3d vector'
    assert gravity.shape[0] == 3, 'Gravity should be a 3d vector'
    gravity_length = 1
    vertex_type = [('x', 'f4'), ('y', 'f4'), ('z', 'f4'), ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')]
    positions_data = np.empty((positions.shape[0],), dtype=vertex_type)
    positions_data[:] = [tuple([*i, 0, 255, 0]) for i in positions]
    if np.linalg.norm(gravity) >= 0.1:
        gravity_sample = np.linspace(0.0, gravity_length, 1000)
        gravity_points = np.empty((gravity_sample.shape[0],), dtype=vertex_type)
        gravity_points[:] = [tuple([*(gravity * dis), 255, 0, 0]) for dis in np.nditer(gravity_sample)]
        positions_data = np.concatenate([positions_data, gravity_points], axis=0)

    vertex_element = plyfile.PlyElement.describe(positions_data, 'vertex')
    plyfile.PlyData([vertex_element], text=True).write(path)

def testEularToQuaternion(eular_input):
    eular_ret = np.empty(eular_input.shape)
    for i in range(eular_input.shape[0]):
        q = quaternion.from_euler_angles(eular_input[i, 0], eular_input[i, 1], eular_input[i, 2])
        eular_ret[i, :] = quaternion.as_euler_angles(q)
    return eular_ret


def rotationMatrixFromTwoVectors(v1, v2):
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


def quaternionFromTwoVectors(v1, v2):
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


def alignWithGravity(poses, gravity, local_g_direction=np.array([0, 0, -1])):
    """
    Adjust pose such that the gravity is at $target$ direction
    @:param poses: N x 7 array, each row is position + orientation (quaternion). The array will be modified in place.
    @:param gravity: real gravity direction
    @:param local_g_direction: z direction before alignment
    @:return None.
    """
    assert poses.ndim == 2, 'Expect 2 dimensional array input'
    assert poses.shape[1] == 7, 'Expect Nx7 array input'
    rotor = quaternionFromTwoVectors(local_g_direction, gravity)
    for pose in poses:
        distance = np.linalg.norm(pose[0:3])
        position_n = pose[0:3] / distance
        pose[0:3] = distance * (rotor * quaternion.quaternion(0.0, *position_n) * rotor.conjugate()).vec
        pose[-4:] = quaternion.as_float_array(rotor * quaternion.quaternion(*pose[-4:]) * rotor.conjugate())


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('list')
    parser.add_argument('--skip', default=100)

    args = parser.parse_args()

    with open(args.list) as f:
        dataset_list = [s.strip('\n') for s in f.readlines()]

    print(dataset_list)
    root_dir = os.path.dirname(args.list)

    nano_to_sec = 1000000000.0
    total_length = 0.0
    length_dict = {}
    for dataset in dataset_list:
        info = dataset.split(',')
        motion_type = 'unknown'
        if len(info) == 2:
            motion_type = info[1]
        data_root = root_dir + '/' + info[0]
        length = 0
        if os.path.exists(data_root + '/processed/data.csv'):
            data_pandas = pandas.read_csv(data_root + '/processed/data.csv')
        else:
            print('------------------\nProcessing ' + data_root, ', type: ' + motion_type)
            pose_data = np.genfromtxt(data_root+'/pose.txt')
            acce_data = np.genfromtxt(data_root+'/acce.txt')
            gyro_data = np.genfromtxt(data_root+'/gyro.txt')
            linacce_data = np.genfromtxt(data_root+'/linacce.txt')
            gravity_data = np.genfromtxt(data_root+'/gravity.txt')

            # Error analysis
            position_error, angular_error = analysisError(pose_data)
            print('Positional error: {:.6f}(m), angular error: {:.6f}(rad)'.format(position_error, angular_error))

            output_folder = data_root + '/processed'
            if not os.path.isdir(output_folder):
                os.makedirs(output_folder)

            # adjust axis
            adjustAxis(acce_data)
            adjustAxis(gyro_data)
            adjustAxis(linacce_data)
            adjustAxis(gravity_data)

            linacce_offset = np.average(linacce_data[200:300, 1:4], axis=0)
            print('Linear acceleration offset: ', linacce_offset)
            linacce_data[:, 1:] -= linacce_offset
            gravity_data[:, 1:] += linacce_offset
            print("Writing trajectory to ply file")
            writeTrajectoryToPly(output_folder + '/trajectory.ply', pose_data[:, 1:4])

            # alignWithGravity(pose_data[:, 1:], initial_gravity)
            # print('Writing aligned trajectory to ply file')
            # writeTrajectoryToPly(output_folder + '/trajectory_aligned.ply', pose_data[:, 1:4], initial_gravity)

            # test_gyro = testEularToQuaternion(gyro_data[:, 1:])
            # np.savetxt(data_root+'/output_test.txt', test_gyro, '%.6f')

            # convert the gyro data to quaternion
            # gyro_quat = np.empty([gyro_data.shape[0], 5])
            # gyro_quat[:, 0] = gyro_data[:, 0]
            # for i in range(gyro_data.shape[0]):
            #     gyro_quat[i, 1:] = quaternion.as_float_array(quaternion.from_euler_angles(gyro_data[i, 1],
            #                                                                           gyro_data[i, 2], gyro_data[i, 3]))
            # writeFile(data_root + '/gyro_quat.txt', gyro_quat)
            #
            # Generate dataset
            output_timestamp = pose_data[:, 0]
            print('Interpolate gyro data.')
            output_gyro = interpolateAngularRateSpline(gyro_data, output_timestamp)
            writeFile(data_root + '/output_gyro.txt', output_gyro)

            output_gyro_linear = interpolateAngularRateLinear(gyro_data, output_timestamp)
            print('Interpolate the acceleration data')
            output_accelerometer_linear = interpolate3DVectorLinear(acce_data, output_timestamp)
            output_linacce_linear = interpolate3DVectorLinear(linacce_data, output_timestamp)
            output_gravity_linear = interpolate3DVectorLinear(gravity_data, output_timestamp)

            # write individual files for convenience
            writeFile(output_folder + '/output_gyro_linear.txt', output_gyro_linear)
            writeFile(output_folder + '/linacce_linear.txt', output_linacce_linear)
            writeFile(output_folder + '/gravity_linear.txt', output_gravity_linear)
            writeFile(output_folder + '/acce_linear.txt', output_accelerometer_linear)

            # construct a Pandas DataFrame
            column_list = 'time,gyro_w,gyro_x,gyro_y,gyro_z,acce_x'.split(',') + \
                          'acce_y,acce_z,linacce_x,linacce_y,linacce_z,grav_x,grav_y,grav_z'.split(',') +\
                          'pos_x,pos_y,pos_z,ori_w,ori_x,ori_y,ori_z'.split(',')

            data_pandas = pandas.DataFrame(np.concatenate([output_gyro_linear,
                                                           output_accelerometer_linear[:, 1:],
                                                           output_linacce_linear[:, 1:],
                                                           output_gravity_linear[:, 1:],
                                                           pose_data[:, 1:4],
                                                           pose_data[:, -4:]], axis=1),
                                           columns=column_list)

            data_pandas.to_csv(output_folder + '/data.csv')
            print('Dataset written to ' + output_folder + '/data.txt')

        length = (data_pandas['time'].values[-1] - data_pandas['time'].values[0]) / nano_to_sec
        print(info[0] + ', length: ', length)
        if motion_type not in length_dict:
            length_dict[motion_type] = length
        else:
            length_dict[motion_type] += length
        total_length += length

    print('All done. Total length: {:.2f}s ({:.2f}min)'.format(total_length, total_length / 60.0))
    for k, v in length_dict.items():
        print(k + ': {:.2f}s ({:.2f}min)'.format(v, v / 60.0))
