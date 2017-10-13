import os
import sys
import numpy as np
import pandas
import quaternion
import cv2
from scipy.ndimage.filters import gaussian_filter1d
from scipy.interpolate import interp1d

import pre_processing.gen_dataset as gen_dataset
from utility.write_trajectory_to_ply import write_ply_to_file

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
    best_offset = -134
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
    parser.add_argument('source', type=str)
    parser.add_argument('target', type=str)
    parser.add_argument('--margin', type=int, default=100)
    parser.add_argument('--no_trajectory', action='store_true')
    parser.add_argument('--device', type=str, default='pixel')
    parser.add_argument('--sync', action='store_true')
    args = parser.parse_args()

    # read raw data from two devices
    print('Reading')
    gyro_source = np.genfromtxt(args.source + '/gyro.txt')[args.margin:-args.margin]
    gyro_target = np.genfromtxt(args.target + '/phab/gyro.txt')[args.margin:-args.margin]
    acce_target = np.genfromtxt(args.target + '/phab/acce.txt')[args.margin:-args.margin]

    print('---------------\nUsing gyroscope')
    sync_source = np.copy(gyro_source)
    sync_target = np.copy(gyro_target)

    time_offset = compute_time_offset(sync_source, sync_target)

    print('Time offset between two devices: ', time_offset)
    print('original: {}, corrected: {}, target: {}'.format(gyro_source[0, 0], gyro_source[0, 0] + time_offset,
                                                           gyro_target[0, 0]))
    print('Time difference between two device: {:.2f}s'.format(
        (gyro_target[0, 0] - gyro_source[0, 0] - time_offset) / nano_to_sec))

    sync_source[:, 0] = (sync_source[:, 0] + time_offset - sync_target[0, 0]) / nano_to_sec
    sync_target[:, 0] = (sync_target[:, 0] - sync_target[0, 0]) / nano_to_sec

    if args.sync:
        plt.figure('Gyroscope')
        for i in range(3):
            plt.subplot(311 + i)
            plt.plot(sync_target[:, 0], sync_target[:, i + 1])
            plt.plot(sync_source[:, 0], sync_source[:, i + 1])
            plt.legend(['Tango', args.device])
        plt.show()
    else:
        # Interpolation
        sample_rate = lambda x: x.shape[0] / (x[-1, 0] - x[0, 0]) * nano_to_sec
        print('Gyroscope: {:.2f}Hz'.format(sample_rate(gyro_source)))
        acce_source = np.genfromtxt(args.source + '/acce.txt')[args.margin:-args.margin]
        print('Accelerometer: {:.2f}Hz'.format(sample_rate(acce_source)))
        linacce_source = np.genfromtxt(args.source + '/linacce.txt')[args.margin:-args.margin]
        print('Linear acceleration: {:.2f}Hz'.format(sample_rate(linacce_source)))
        gravity_source = np.genfromtxt(args.source + '/gravity.txt')[args.margin:-args.margin]
        print('Gravity: {:.2f}Hz'.format(sample_rate(gravity_source)))
        magnet_source = np.genfromtxt(args.source + '/magnet.txt')[args.margin:-args.margin]
        print('Magnetometer: {:.2f}Hz'.format(sample_rate(magnet_source)))
        rv_source = np.genfromtxt(args.source + '/orientation.txt')[args.margin:-args.margin]
        print('Rotation vector: {:.2f}Hz'.format(sample_rate(rv_source)))

        pose_data = np.genfromtxt(args.target + '/pose.txt')[args.margin:-args.margin]
        # reorder the quaternion representation
        pose_data[:, [-4, -3, -2, -1]] = pose_data[:, [-1, -4, -3, -2]]

        gyro_source[:, 0] += time_offset
        acce_source[:, 0] += time_offset
        linacce_source[:, 0] += time_offset
        gravity_source[:, 0] += time_offset
        magnet_source[:, 0] += time_offset
        rv_source[:, 0] += time_offset

        pose_truncate_ind = pose_data.shape[0] - 1
        min_timestamp = min([gyro_source[-1, 0], acce_source[-1, 0], linacce_source[-1, 0],
                             gravity_source[-1, 0], magnet_source[-1, 0], rv_source[-1, 0]])
        while pose_data[pose_truncate_ind, 0] >= min_timestamp:
            pose_truncate_ind -= 1
        pose_data = pose_data[:pose_truncate_ind + 1]

        output_timestamp = pose_data[:, 0]
        output_gyro = gen_dataset.interpolate_3dvector_linear(gyro_source[:, 1:], gyro_source[:, 0], output_timestamp)
        output_acce = gen_dataset.interpolate_3dvector_linear(acce_source[:, 1:], acce_source[:, 0], output_timestamp)
        output_linacce = gen_dataset.interpolate_3dvector_linear(linacce_source[:, 1:], linacce_source[:, 0],
                                                                 output_timestamp)
        output_gravity = gen_dataset.interpolate_3dvector_linear(gravity_source[:, 1:], gravity_source[:, 0],
                                                                 output_timestamp)
        output_magnet = gen_dataset.interpolate_3dvector_linear(magnet_source[:, 1:], magnet_source[:, 0],
                                                                output_timestamp)
        output_rv = gen_dataset.interpolate_quaternion_linear(rv_source[:, 1:], rv_source[:, 0], output_timestamp)

        column_list = 'time,gyro_x,gyro_y,gyro_z,acce_x'.split(',') + \
                      'acce_y,acce_z,linacce_x,linacce_y,linacce_z,grav_x,grav_y,grav_z'.split(',') + \
                      'magnet_x,magnet_y,magnet_z'.split(',') + \
                      'pos_x,pos_y,pos_z,ori_w,ori_x,ori_y,ori_z,rv_w,rv_x,rv_y,rv_z'.split(',')
        data_mat = np.concatenate([output_timestamp[:, None], output_gyro, output_acce, output_linacce,
                                   output_gravity, output_magnet, pose_data[:, -7:-4], pose_data[:, -4:],
                                   output_rv], axis=1)
        output_data = pandas.DataFrame(data=data_mat, columns=column_list, dtype=float)
        print('Writing csv...')
        output_dir = args.source + '/processed'
        os.makedirs(output_dir, exist_ok=True)

        output_data.to_csv(output_dir + '/data.csv')

        # write data in plain text file for C++
        with open(output_dir + '/data_plain.txt', 'w') as f:
            f.write('{} {}\n'.format(data_mat.shape[0], data_mat.shape[1]))
            for i in range(data_mat.shape[0]):
                for j in range(data_mat.shape[1]):
                    f.write('{}\t'.format(data_mat[i][j]))
                f.write('\n')

        if not args.no_trajectory:
            import quaternion
            print('Writing ply...')
            write_ply_to_file(output_dir + '/trajectory.ply', position=pose_data[:, -7:-4],
                              orientation=pose_data[:, -4:])

            # q_device_tango = quaternion.quaternion(1.0 / math.sqrt(2.0), 1.0 / math.sqrt(2.0), 0., 0.)
            q_device_tango = quaternion.quaternion(*pose_data[0, -4:])
            # q_device_tango = quaternion.quaternion(1., 0., 0., 0.)
            q_rv_tango = q_device_tango * quaternion.quaternion(*output_rv[0, 1:]).inverse()
            orientation_tango_frame = np.empty([output_rv.shape[0], 4], dtype=float)
            for i in range(orientation_tango_frame.shape[0]):
                orientation_tango_frame[i] = quaternion.as_float_array(q_rv_tango *
                                                                       quaternion.quaternion(*output_rv[i, 1:]))
            write_ply_to_file(output_dir + '/trajectory_rv.ply', position=pose_data[:, -7:-4],
                              orientation=orientation_tango_frame)
