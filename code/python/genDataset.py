import numpy as np
import quaternion
import argparse

def estimateOffset(input_data, K = 100):
    assert(input_data.shape[0] >= K)
    return np.average(input_data[:K, 1:], axis=0)

    

def interpolateAngularRate(gyro_data, output_timestamp):
    offset = estimateOffset(gyro_data)
    print('Offset in gyro data: {}'.format(offset))
    for record in gyro_data:
        gyro_data[:,1:] -= offset

    return gyro_data
    

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

    output_timestamp = pose_data[:, 0]

    # Generate dataset
    output_gyro = interpolateAngularRate(gyro_data, output_timestamp)
    output_acce = interpolateAcceleration(acce_data, output_timestamp)
    
