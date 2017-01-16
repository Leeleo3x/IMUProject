#pylint: disable=C0103,C0111,C0301

import math
import argparse
import numpy as np
import quaternion
import quaternion.quaternion_time_series


def estimateOffset(input_data, K=100):
    assert input_data.shape[0] >= K
    return np.average(input_data[:K, 1:], axis=0)

# for Tango development kit
def adjustAxis(input_data):
    # first swap y and z
    input_data[:, [2, 3]] = input_data[:, [3, 2]]
    # invert x, y axis
    input_data[:, 1:2] *= -1


def normalizeEularAngle(input):
    """Normalize eular angle to -pi/2 to pi/2"""
    input[input < -math.pi / 2] += math.pi
    input[input > math.pi / 2] -= math.pi

def interpolateAngularRateSpline(gyro_data, output_timestamp):
    # convert angular velocity to quaternion
    print('Using spline interpolation')
    N_input = gyro_data.shape[0]
    N_output = output_timestamp.shape[0]
    gyro_quat = np.empty([gyro_data.shape[0], 4], dtype=float)
    output_eular = np.zeros([N_output, 4])
    output_eular[:, 0] = output_timestamp

    for i in range(N_input):
        record = gyro_data[i, 1:4]
        gyro_quat[i] = quaternion.as_float_array(quaternion.from_euler_angles(record[0], record[1], record[2]))

    gyro_interpolated = quaternion.quaternion_time_series.squad(quaternion.as_quat_array(gyro_quat),
                                                                gyro_data[:, 0], output_timestamp)

    for i in range(N_output):
        output_eular[i, 1:] = quaternion.as_euler_angles(gyro_interpolated[i])

    normalizeEularAngle(output_eular[:, 1:])
    return output_eular


def interpolateAngularRateLinear(gyro_data, output_timestamp):
    print('Using linear interpolation')
    N_input = gyro_data.shape[0]
    N_output = output_timestamp.shape[0]

    result = np.zeros([output_timestamp.shape[0], 4])
    result[:, 0] = output_timestamp

    ptr = 0
    for i in range(N_output):
        if ptr >= N_input - 1:
            raise ValueError
        if gyro_data[ptr, 0] <= output_timestamp[i] <= gyro_data[ptr+1, 0]:
            # The correct interval
            q1 = quaternion.from_euler_angles(gyro_data[ptr, 1], gyro_data[ptr, 2], gyro_data[ptr, 3])
            q2 = quaternion.from_euler_angles(gyro_data[ptr+1, 1], gyro_data[ptr+1, 2], gyro_data[ptr+1, 3])

            q_inter = quaternion.quaternion_time_series.slerp(q1, q2, gyro_data[ptr, 0], gyro_data[ptr+1, 0],
                                                              output_timestamp[i])
            result[i, 1:] = quaternion.as_euler_angles(q_inter)
        else:
            ptr += 1
    return result


def interpolateAcceleration(acce_data, output_timestamp):
    offset = estimateOffset(acce_data)
    for record in gyro_data:
        acce_data[:,1:] -= offset
    return gyro_data


def writeFile(path, data):
    with open(path, 'w') as f:
        for row in data:
            f.write('{:.0f} {:.6f} {:.6f} {:.6f}\n'.format(row[0], row[1], row[2], row[3]))

def testEularToQuaternion(eular_input):
    eular_ret = np.empty(eular_input.shape)
    for i in range(eular_input.shape[0]):
        q = quaternion.from_euler_angles(eular_input[i, 0], eular_input[i, 1], eular_input[i, 2])
        eular_ret[i, :] = quaternion.as_euler_angles(q)
    normalizeEularAngle(eular_ret)
    return eular_ret


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('dir')
    parser.add_argument('--skip', default=100)

    args = parser.parse_args()

    pose_data = np.genfromtxt(args.dir+'/pose.txt')
    acce_data = np.genfromtxt(args.dir+'/acce.txt')
    gyro_data = np.genfromtxt(args.dir+'/gyro.txt')

    # test_gyro = testEularToQuaternion(gyro_data[:, 1:])
    # np.savetxt(args.dir+'/output_test.txt', test_gyro, '%.6f')

    # adjust axis
    adjustAxis(acce_data)
    adjustAxis(gyro_data)

    # Generate dataset
    output_timestamp = pose_data[:, 0]
    print('Interpolate gyro data.')
    output_gyro = interpolateAngularRateSpline(gyro_data, output_timestamp)
    output_gyro_linear = interpolateAngularRateLinear(gyro_data, output_timestamp)
    writeFile(args.dir + '/output_gyro.txt', output_gyro)
    writeFile(args.dir + '/output_gyro_linear.txt', output_gyro_linear)
