import numpy as np
import pandas
import quaternion
import pre_processing.gen_dataset as gen_dataset
from scipy.ndimage.filters import gaussian_filter1d

nano_to_sec = 1e09

def compute_time_offset(source, target, search_range=200):
    """
    Synchronize the timestamp of source to the target
    :param source:
    :param target:
    :param range:
    :return: synchonized timestamp
    """
    assert source.shape[1] == target.shape[1]
    best_score = np.inf
    best_offset = 0
    for offset in range(0, search_range):
        length = min(source.shape[0] - offset, target.shape[0])
        diff = np.sum(np.fabs(source[offset:offset+length, 1:] - target[:length, 1:])) / float(length)
        if diff < best_score:
            best_score = diff
            best_offset = -offset
        length = min(target.shape[0] - offset, source.shape[0])
        diff = np.sum(np.fabs((source[:length, 1:] - target[offset:offset+length, 1:]))) / float(length)
        if diff < best_score:
            best_score = diff
            best_offset = offset
    time_offset = 0
    if best_offset > 0:
        time_offset = target[best_offset, 0] - source[0, 0]
    elif best_offset < 0:
        time_offset = target[0, 0] - source[best_offset, 0]



    print('Best offset: {}, time_offset: {}'.format(best_offset, time_offset / nano_to_sec))
    return time_offset



if __name__ == '__main__':
    import argparse
    import matplotlib.pyplot as plt

    parser = argparse.ArgumentParser()
    parser.add_argument('dir', type=str, default=None)
    parser.add_argument('--margin', type=int, default=500)
    args = parser.parse_args()

    # read raw data from two devices
    print('Reading')
    gyro_nexus = np.genfromtxt(args.dir + '/nexus/gyro.txt')[args.margin:-args.margin]
    acce_nexus = np.genfromtxt(args.dir + '/nexus/acce.txt')[args.margin:-args.margin]
    linacce_nexus = np.genfromtxt(args.dir + '/nexus/linacce.txt')[args.margin:-args.margin]
    rv_nexus = np.genfromtxt(args.dir + '/nexus/orientation.txt')[args.margin:-args.margin]

    gyro_tango = np.genfromtxt(args.dir + '/tango/gyro.txt')[args.margin:-args.margin]
    acce_tango = np.genfromtxt(args.dir + '/tango/acce.txt')[args.margin:-args.margin]

    pose_data = np.genfromtxt(args.dir + '/tango/pose.txt')[args.margin:-args.margin]

    # Synchronize using accelerometer data
    print('Synchronizing')
    print('---------------\nUsing accelerometer')
    # For accelerometer, pass a low pass filter
    filter_size = 30.0
    sync_source = np.copy(acce_nexus)
    sync_source[:, 1:] -= np.average(sync_source[:, 1:], axis=0)
    # sync_source[:, 1:] = gaussian_filter1d(sync_source[:, 1:], sigma=filter_size, axis=0)
    sync_target = np.copy(acce_tango)
    sync_target[:, 1:] -= np.average(sync_target[:, 1:], axis=0)
    # sync_target[:, 1:] = gaussian_filter1d(sync_target[:, 1:], sigma=filter_size, axis=0)

    plt.figure('Accelerometer')
    for i in range(3):
        plt.subplot(311 + i)
        plt.plot((sync_target[:, 0] - sync_target[0, 0]) / nano_to_sec, sync_target[:, i + 1])
        plt.plot((sync_source[:, 0] - sync_source[0, 0]) / nano_to_sec, sync_source[:, i + 1])
        plt.legend(['Tango', 'Nexus'])

    time_offset = compute_time_offset(sync_source, sync_target)

    print('Time offset between two devices: ', time_offset)
    print('original: {}, corrected: {}, target: {}'.format(gyro_nexus[0, 0], gyro_nexus[0, 0] + time_offset, gyro_tango[0, 0]))
    print('Time difference between two device: {:.2f}s'.format((gyro_tango[0, 0] - gyro_nexus[0, 0] - time_offset) / nano_to_sec))

    print('---------------\nUsing gyroscope')
    sync_source = np.copy(gyro_nexus)
    sync_source[:, 1:] -= np.average(sync_source[:, 1:], axis=0)
    sync_target = np.copy(gyro_tango)
    sync_target[:, 1:] -= np.average(sync_target[:, 1:], axis=0)

    time_offset = compute_time_offset(sync_source, sync_target)

    print('Time offset between two devices: ', time_offset)
    print('original: {}, corrected: {}, target: {}'.format(gyro_nexus[0, 0], gyro_nexus[0, 0] + time_offset,
                                                           gyro_tango[0, 0]))
    print('Time difference between two device: {:.2f}s'.format(
        (gyro_tango[0, 0] - gyro_nexus[0, 0] - time_offset) / nano_to_sec))

    plt.figure('Gyroscope')
    for i in range(3):
        plt.subplot(311 + i)
        plt.plot((sync_target[:, 0] - sync_target[0, 0]) / nano_to_sec, sync_target[:, i + 1])
        plt.plot((sync_source[:, 0] - sync_source[0, 0]) / nano_to_sec, sync_source[:, i + 1])
        plt.legend(['Tango', 'Nexus'])

    plt.show()
    # Interpolation
