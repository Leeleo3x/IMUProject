import os
import numpy as np
import pandas
import quaternion
import pre_processing.gen_dataset as gen_dataset
from utility.write_trajectory_to_ply import write_ply_to_file
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
    # for offset in range(0, search_range):
    #     length = min(source.shape[0] - offset, target.shape[0])
    #     diff = np.sum(np.fabs(source[offset:offset+length, 1:] - target[:length, 1:])) / float(length)
    #     if diff < best_score:
    #         best_score = diff
    #         best_offset = -offset
    #     length = min(target.shape[0] - offset, source.shape[0])
    #     diff = np.sum(np.fabs((source[:length, 1:] - target[offset:offset+length, 1:]))) / float(length)
    #     if diff < best_score:
    #         best_score = diff
    #         best_offset = offset
    best_offset = -113
    time_offset = 0
    if best_offset > 0:
        time_offset = target[best_offset, 0] - source[0, 0]
    elif best_offset < 0:
        time_offset = target[0, 0] - source[-best_offset, 0]
    print('Best offset: {}, time_offset: {}'.format(best_offset, time_offset / nano_to_sec))
    return time_offset

if __name__ == '__main__':
    import argparse
    import matplotlib.pyplot as plt

    parser = argparse.ArgumentParser()
    parser.add_argument('dir', type=str, default=None)
    parser.add_argument('--margin', type=int, default=500)
    parser.add_argument('--no_trajectory', action='store_true')
    args = parser.parse_args()

    # read raw data from two devices
    print('Reading')
    gyro_nexus = np.genfromtxt(args.dir + '/nexus/gyro.txt')[args.margin:-args.margin]

    gyro_tango = np.genfromtxt(args.dir + '/tango/gyro.txt')[args.margin:-args.margin]
    acce_tango = np.genfromtxt(args.dir + '/tango/acce.txt')[args.margin:-args.margin]

    print('---------------\nUsing gyroscope')
    sync_source = np.copy(gyro_nexus)
    # sync_source[:, 1:] -= np.average(sync_source[:, 1:], axis=0)
    sync_target = np.copy(gyro_tango)
    # sync_target[:, 1:] -= np.average(sync_target[:, 1:], axis=0)

    time_offset = compute_time_offset(sync_source, sync_target)

    print('Time offset between two devices: ', time_offset)
    print('original: {}, corrected: {}, target: {}'.format(gyro_nexus[0, 0], gyro_nexus[0, 0] + time_offset,
                                                           gyro_tango[0, 0]))
    print('Time difference between two device: {:.2f}s'.format(
        (gyro_tango[0, 0] - gyro_nexus[0, 0] - time_offset) / nano_to_sec))

    sync_source[:, 0] = (sync_source[:, 0] + time_offset - sync_target[0, 0]) / nano_to_sec
    sync_target[:, 0] = (sync_target[:, 0] - sync_target[0, 0]) / nano_to_sec
    plt.figure('Gyroscope')
    for i in range(3):
        plt.subplot(311 + i)
        plt.plot(sync_target[:, 0], sync_target[:, i + 1])
        plt.plot(sync_source[:, 0], sync_source[:, i + 1])
        plt.legend(['Tango', 'Nexus'])

    # plt.show()

    # Interpolation
    sample_rate = lambda x: x.shape[0] / (x[-1, 0] - x[0, 0]) * nano_to_sec
    print('Gyroscope: {:.2f}Hz'.format(sample_rate(gyro_nexus)))
    acce_nexus = np.genfromtxt(args.dir + '/nexus/acce.txt')[args.margin:-args.margin]
    print('Accelerometer: {:.2f}Hz'.format(sample_rate(acce_nexus)))
    linacce_nexus = np.genfromtxt(args.dir + '/nexus/linacce.txt')[args.margin:-args.margin]
    print('Linear acceleration: {:.2f}Hz'.format(sample_rate(linacce_nexus)))
    gravity_nexus = np.genfromtxt(args.dir + '/nexus/gravity.txt')[args.margin:-args.margin]
    print('Gravity: {:.2f}Hz'.format(sample_rate(gravity_nexus)))
    magnet_nexus = np.genfromtxt(args.dir + '/nexus/magnet.txt')[args.margin:-args.margin]
    print('Magnetometer: {:.2f}Hz'.format(sample_rate(magnet_nexus)))
    rv_nexus = np.genfromtxt(args.dir + '/nexus/orientation.txt')[args.margin:-args.margin]
    print('Rotation vector: {:.2f}Hz'.format(sample_rate(rv_nexus)))

    pose_data = np.genfromtxt(args.dir + '/tango/pose.txt')[args.margin:-args.margin]
    # reorder the quaternion representation
    pose_data[:, [-4, -3, -2, -1]] = pose_data[:, [-1, -4, -3, -2]]

    gyro_nexus[:, 0] += time_offset
    acce_nexus[:, 0] += time_offset
    linacce_nexus[:, 0] += time_offset
    gravity_nexus[:, 0] += time_offset
    magnet_nexus[:, 0] += time_offset
    rv_nexus[:, 0] += time_offset
    # reorder the quaternion representation from [x,y,z,w] to [w,x,y,z]
    rv_nexus[:, [1, 2, 3, 4]] = rv_nexus[:, [4, 1, 2, 3]]

    pose_truncate_ind = pose_data.shape[0] - 1
    min_timestamp = min([gyro_nexus[-1, 0], acce_nexus[-1, 0], linacce_nexus[-1, 0],
                         gravity_nexus[-1, 0], magnet_nexus[-1, 0], rv_nexus[-1, 0]])
    while pose_data[pose_truncate_ind, 0] >= min_timestamp:
        pose_truncate_ind -= 1
    pose_data = pose_data[:pose_truncate_ind + 1]

    output_timestamp = pose_data[:, 0]
    output_gyro = gen_dataset.interpolate_3dvector_linear(gyro_nexus, output_timestamp)
    output_acce = gen_dataset.interpolate_3dvector_linear(acce_nexus, output_timestamp)
    output_linacce = gen_dataset.interpolate_3dvector_linear(linacce_nexus, output_timestamp)
    output_gravity = gen_dataset.interpolate_3dvector_linear(gravity_nexus, output_timestamp)
    output_magnet = gen_dataset.interpolate_3dvector_linear(magnet_nexus, output_timestamp)

    output_rv = gen_dataset.interpolate_quaternion_linear(rv_nexus, output_timestamp)

    column_list = 'time,gyro_x,gyro_y,gyro_z,acce_x'.split(',') + \
                  'acce_y,acce_z,linacce_x,linacce_y,linacce_z,grav_x,grav_y,grav_z,mag_x,mag_y,mag_z'.split(',') + \
                  'pos_x,pos_y,pos_z,ori_w,ori_x,ori_y,ori_z,rv_w,rv_x,rv_y,rv_z'.split(',')
    data_mat = np.concatenate([output_timestamp[:, None],
                               output_gyro[:, 1:], output_acce[:, 1:], output_linacce[:, 1:],
                               output_gravity[:, 1:], output_magnet[:, 1:],
                               pose_data[:, -7:-4], pose_data[:, -4:], output_rv[:, 1:]], axis=1)
    output_data = pandas.DataFrame(data=data_mat, columns=column_list, dtype=float)
    print('Writing csv...')
    output_dir = args.dir + '/processed'
    os.makedirs(output_dir, exist_ok=True)

    output_data.to_csv(output_dir + '/data.csv')

    if not args.no_trajectory:
        print('Writing ply...')
        write_ply_to_file(output_dir + '/trajectory.ply', position=pose_data[:, -7:-4],
                          orientation=pose_data[:, -4:])
        write_ply_to_file(output_dir + '/trajectory_rv.ply', position=pose_data[:, -7:-4],
                          orientation=output_rv[:, 1:])
