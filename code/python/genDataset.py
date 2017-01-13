import numpy as np
import quaternion
import quaternion.quaternion_time_series as quat_time
import argparse

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
    gyro_quat = quaternion.as_quat_array(np.asarray([quaternion.from_euler_angles(angle[0], angle[1], angle[2])
                                                     for angle in gyro_data[:, 1:]]))
    output_quat = quaternion.quaternion_time_series.squad(gyro_quat, gyro_data[:, 0], output_timestamp)
    output_eular = [quaternion.as_euler_angles(q) for q in output_quat]
    return output_eular
    

def interpolateAcceleration(acce_data, output_timestamp):
    offset = estimateOffset(acce_data)
    print('Offset in accleration data: {}'.format(offset))
    for record in gyro_data:
        acce_data[:,1:] -= offset

    return gyro_data

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('dir')
    parser.add_argument('--skip', default=100)

    args = parser.parse_args()

    pose_data = np.genfromtxt(args.dir+'/pose.txt')
    acce_data = np.genfromtxt(args.dir+'/acce.txt')
    gyro_data = np.genfromtxt(args.dir+'/gyro.txt')

    # adjust axis
    adjustAxis(acce_data)
    adjustAxis(gyro_data)

    # Generate dataset
    output_timestamp = pose_data[:, 0]
    output_gyro = interpolateAngularRate(gyro_data, output_timestamp)
    output_acce = interpolateAcceleration(acce_data, output_timestamp)
