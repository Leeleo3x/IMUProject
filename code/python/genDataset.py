#pylint: disable=C0103,C0111,C0301

import argparse
import os
from datetime import datetime

import numpy as np
import scipy.interpolate
import quaternion
import quaternion.quaternion_time_series
import matplotlib.pyplot as plt


def computeIntervalVariance(input, N=100, step=10):
    range_list = [np.var(input[s:s+N, :], axis=0) for s in range(0, input.shape[0]-N, step)]
    plt.plot(np.linalg.norm(range_list))


def estimateOffset(input_data, N=200, K=100):
    # assert input_data.shape[0] >= K
    # sample_point = np.arange(0, min(400, input_data.shape[0]), 10)
    # range_list = [np.linalg.norm(np.var(input_data[s:s+K, :], axis=0), axis=0) for s in sample_point]
    # min_var = min(range_list)
    # min_ind = sample_point[range_list.index(min_var)]
    # return np.average(input_data[min_ind:min_ind+K], axis=0)

    # Assume the device is static from 2s to 3s
    return np.average(input_data[N:N+K], axis=0)


# for Tango development kit
def adjustAxis(input_data):
    # first swap y and z
    input_data[:, [2, 3]] = input_data[:, [3, 2]]
    # invert x, y axis
    input_data[:, 1:2] *= -1

def extractGravity(acce):
    """Extract gravity from accelerometer"""
    

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

def testEularToQuaternion(eular_input):
    eular_ret = np.empty(eular_input.shape)
    for i in range(eular_input.shape[0]):
        q = quaternion.from_euler_angles(eular_input[i, 0], eular_input[i, 1], eular_input[i, 2])
        eular_ret[i, :] = quaternion.as_euler_angles(q)
    return eular_ret


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('dir')
    parser.add_argument('--skip', default=100)

    args = parser.parse_args()

    pose_data = np.genfromtxt(args.dir+'/pose.txt')
    acce_data = np.genfromtxt(args.dir+'/acce.txt')
    gyro_data = np.genfromtxt(args.dir+'/gyro.txt')
    linacce_data = np.genfromtxt(args.dir+'/linacce.txt')
    gravity_data = np.genfromtxt(args.dir+'/gravity.txt')

    output_folder = args.dir + '/processed'
    if not os.path.isdir(output_folder):
        os.makedirs(output_folder)

    linacce_offset = estimateOffset(linacce_data[:, 1:])
    print('Linear acceleration offset: ', linacce_offset)
    linacce_data[:, 1:] -= linacce_offset
    gravity_data[:, 1:] += linacce_offset

    # test_gyro = testEularToQuaternion(gyro_data[:, 1:])
    # np.savetxt(args.dir+'/output_test.txt', test_gyro, '%.6f')

    # adjust axis
    # adjustAxis(acce_data)
    # adjustAxis(gyro_data)

    # convert the gyro data to quaternion
    # gyro_quat = np.empty([gyro_data.shape[0], 5])
    # gyro_quat[:, 0] = gyro_data[:, 0]
    # for i in range(gyro_data.shape[0]):
    #     gyro_quat[i, 1:] = quaternion.as_float_array(quaternion.from_euler_angles(gyro_data[i, 1],
    #                                                                           gyro_data[i, 2], gyro_data[i, 3]))
    # writeFile(args.dir + '/gyro_quat.txt', gyro_quat)
    #
    # Generate dataset
    output_timestamp = pose_data[:, 0]
    # print('Interpolate gyro data.')
    # output_gyro = interpolateAngularRateSpline(gyro_data, output_timestamp)
    # writeFile(args.dir + '/output_gyro.txt', output_gyro)
    #
    output_gyro_linear = interpolateAngularRateLinear(gyro_data, output_timestamp)
    print('Interpolate the acceleration data')
    output_accelerometer_linear = interpolate3DVectorLinear(acce_data, output_timestamp)
    output_linacce_linear = interpolate3DVectorLinear(linacce_data, output_timestamp)
    output_gravity_linear = interpolate3DVectorLinear(gravity_data, output_timestamp)
    output_acce_combined = np.concatenate([output_timestamp[:, np.newaxis],
                                          output_linacce_linear[:, 1:] + output_gravity_linear[:, 1:]], axis=1)

    writeFile(output_folder + '/output_gyro_linear.txt', output_gyro_linear)
    writeFile(output_folder + '/linacce_linear.txt', output_linacce_linear)
    writeFile(output_folder + '/gravity_linear.txt', output_gravity_linear)
    writeFile(output_folder + '/combined_linear.txt', output_acce_combined)
    writeFile(output_folder + '/acce_linear.txt', output_accelerometer_linear)

    dataset_all = np.concatenate([output_gyro_linear, output_linacce_linear[:, 1:],
                                  output_gravity_linear[:, 1:]], axis=1)
    writeFile(output_folder + '/data.txt', dataset_all,
              '# {}, timestamp, gyro (quaternion), linear acceleration, gravity'.format(datetime.now()))
    print('Dataset written to ' + output_folder + '/data.txt')
