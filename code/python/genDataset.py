import numpy as np
import quaternion
import quaternion.quaternion_time_series as quat_time
import argparse
import math

def estimateOffset(input_data, K = 100):
    assert(input_data.shape[0] >= K)
    return np.average(input_data[:K, 1:], axis=0)

# for Tango development kit
def adjustAxis(input_data):
    # first swap y and z
    input_data[:, [2, 3]] = input_data[:, [3, 2]]
    # invert x, y axis
    input_data[:, 0:1] *= -1

def interpolateAngularRate(gyro_data, output_timestamp):
    # offset = estimateOffset(gyro_data)
    # print('Offset in gyro data: {}'.format(offset))
    # for record in gyro_data:
    #     gyro_data[:,1:] -= offset

    # convert angular velocity to quaternion
    N_input = gyro_data.shape[0]
    N_output = output_timestamp.shape[0]
    gyro_quat = np.empty([gyro_data.shape[0], 4], dtype=float)
    output_eular = np.zeros([N_output, 4])
    output_eular[:, 0] = output_timestamp

    for i in range(N_input):
        record = gyro_data[i, 1:4]
        q = quaternion.from_euler_angles(record[0], record[1], record[2])
        gyro_quat[i] = quaternion.as_float_array()

    np.savetxt(args.dir + '/quat.txt', gyro_quat, '%.6f')

    gyro_interpolated = quaternion.quaternion_time_series.squad(quaternion.as_quat_array(gyro_quat), gyro_data[:, 0], output_timestamp)

    np.savetxt(args.dir + '/quat_inter.txt', quaternion.as_float_array(gyro_interpolated), '%.6f')
    last_eular = gyro_data[0, 1:]

    for i in range(N_output):
        output_eular[i, 1:] = quaternion.as_euler_angles(gyro_interpolated[i])
        for j in range(3):
            if abs((output_eular[i, j+1] - last_eular[0])) > math.pi / 2:
                output_eular[i, j+1] = math.pi - output_eular[i, j+1]
        last_eular = output_eular[i, 1:]

    return output_eular
    

def interpolateAcceleration(acce_data, output_timestamp):
    offset = estimateOffset(acce_data)
    print('Offset in accleration data: {}'.format(offset))
    for record in gyro_data:
        acce_data[:,1:] -= offset

    return gyro_data


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

    #test_gyro = testEularToQuaternion(gyro_data[:, 1:])
    #np.savetxt(args.dir+'/output_test.txt', test_gyro, '%.6f')

    # adjust axis
    adjustAxis(acce_data)
    adjustAxis(gyro_data)

    # Generate dataset
    output_timestamp = pose_data[:, 0]

    print('Interpolate gyro data.')
    output_gyro = interpolateAngularRate(gyro_data, output_timestamp)
    np.savetxt(args.dir+'/output_gyro.txt', output_gyro[:, 1:], '%.6f')



    #output_acce = interpolateAcceleration(acce_data, output_timestamp)
