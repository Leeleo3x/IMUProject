import os
import sys
import math
import argparse
import time
from numba import jit
import numpy as np
import quaternion
import pandas
from scipy.fftpack import fft
from scipy.ndimage.filters import gaussian_filter1d
import matplotlib.pyplot as plt


class TrainingDataOption:
    def __init__(self, sample_step=10, window_size=200, feature='fourier', frq_threshold=10, discard_direct=False,
                 feature_smooth_sigma=None, speed_smooth_sigma=50.0):
        self.sample_step_ = sample_step
        self.window_size_ = window_size
        self.frq_threshold_ = frq_threshold
        self.feature_ = feature
        self.nanoToSec = 1000000000.0
        self.discard_direct_ = discard_direct
        self.feature_smooth_sigma_ = feature_smooth_sigma
        self.speed_smooth_sigma_ = speed_smooth_sigma


#@jit
def compute_fourier_features(data, samples, window_size, threshold, discard_direct=False):
    """
    Compute fourier coefficients as feature vector
    :param data: NxM array for N samples with M dimensions
    :return: Nxk array
    """
    skip = 0
    if discard_direct:
        skip = 1
    features = np.empty([samples.shape[0], data.shape[1] * (threshold - skip)], dtype=np.float)
    for i in range(samples.shape[0]):
        features[i, :] = np.abs(fft(data[samples[i]-window_size:samples[i]], axis=0)[skip:threshold]).flatten()
    return features


#@jit
def compute_direct_features(data, samples, window_size):
    features = np.empty([samples.shape[0], data.shape[1] * window_size], dtype=np.float)
    for i in range(samples.shape[0]):
        features[i, :] = data[samples[i] - window_size:samples[i]].flatten()
    return features
    # return [list(data[ind - option.window_size_:ind].values.flatten())
    #         for ind in samples]


def compute_speed(time_stamp, position, sample_points=None):
    """
    Compute speed vector giving position and time_stamp
    :param time_stamp:
    :param position:
    :param sample_points:
    :return:
    """
    if sample_points is None:
        sample_points = np.arange(1, time_stamp.shape[0] - 1, dtype=int)
    sample_points[-1] = min(sample_points[-1], time_stamp.shape[0] - 2)
    speed = (position[sample_points+1] - position[sample_points]) \
            / (time_stamp[sample_points+1] - time_stamp[sample_points])[:, None]
    return speed


def compute_delta_angle(time_stamp, position, orientation,
                        sample_points, local_axis=quaternion.quaternion(1.0, 0., 0., -1.)):
    """
    Compute the cosine between the moving direction and viewing direction
    :param time_stamp: Time stamp
    :param position: Position. When passing Nx2 array, compute ignore z direction
    :param orientation: Orientation as quaternion
    :param local_axis: the viewing direction in the device frame. Default is set w.r.t. to android coord frame
    :return:
    """
    epsilon = 1e-10
    speed_dir = compute_speed(time_stamp, position)
    speed_dir = np.concatenate([np.zeros([1, position.shape[1]]), speed_dir], axis=0)
    speed_mag = np.linalg.norm(speed_dir, axis=1)
    cos_array = np.zeros(sample_points.shape[0], dtype=float)
    valid_array = np.empty(sample_points.shape[0], dtype=bool)
    for i in range(sample_points.shape[0]):
        if speed_mag[sample_points[i]] <= epsilon:
            valid_array[i] = False
        else:
            q = quaternion.quaternion(*orientation[sample_points[i]])
            camera_axis = (q * local_axis * q.conj()).vec[:position.shape[1]]
            cos_array[i] = min(np.dot(speed_dir[sample_points[i]], camera_axis) / speed_mag[sample_points[i]], 1.0)
            valid_array[i] = True
    return cos_array, valid_array


def get_training_data(data_all, imu_columns, option, sample_points=None):
    """
    Create training data.
    :param data_all: The whole dataset. Must include 'time' column and all columns inside imu_columns
    :param imu_columns: Columns used for constructing feature vectors. Fields must exist in the dataset
    :return: [Nx(d+1)] array. Target value is appended at back
    """
    N = data_all.shape[0]
    if sample_points is None:
        sample_points = np.arange(option.window_size_,
                                  N - 1,
                                  option.sample_step_,
                                  dtype=int)
    assert sample_points[-1] < N

    pose_data = data_all[['pos_x', 'pos_y', 'pos_z']].values
    data_used = data_all[imu_columns].values
    time_stamp = data_all['time'].values / 1e09

    speed_all = np.zeros(N, dtype=float)
    speed_all[1:-1] = np.divide(np.linalg.norm(pose_data[2:] - pose_data[:-2], axis=1),
                                time_stamp[2:] - time_stamp[:-2])
    # filter the speed with gaussian filter
    if option.speed_smooth_sigma_ > 0:
        speed_all = gaussian_filter1d(speed_all, option.speed_smooth_sigma_, axis=0)

    if option.feature_ == 'direct':
        # local_imu_list = compute_direct_features(data_used, sample_points, option.window_size_)
        local_imu_list = [data_used[ind - option.window_size_:ind].flatten() for ind in sample_points]
    elif option.feature_ == 'fourier':
        local_imu_list = compute_fourier_features(data_used, sample_points, option.window_size_, option.frq_threshold_,
                                                  option.discard_direct_)
        # local_imu_list = [compute_fourier_feature(data_used[ind-option.window_size_:ind].values, option.frq_threshold_)
        #                   .flatten('F') for ind in sample_points]
    else:
        print('Feature type not supported: ' + option.feature_)
        raise ValueError

    return np.concatenate([local_imu_list, speed_all[sample_points, None]], axis=1)


def get_orientation_training_data(data_all, camera_orientation, imu_columns, options, sample_points=None,
                                  no_feature=False):
    """
    Get the 2D view angle feature.
    :param data_all:
    :param imu_columns:
    :param options:
    :param sample_points:
    :return:
    """
    assert camera_orientation.shape[0] == data_all.shape[0]
    if sample_points is None:
        sample_points = np.arange(options.window_size_,
                                  data_all.shape[0] - 1,
                                  options.sample_step_,
                                  dtype=int)
    # Avoid out of bound error
    if sample_points[-1] == data_all.shape[0] - 1:
        sample_points[-1] -= 1
    if sample_points[0] == 0:
        sample_points[0] = 1
    if not no_feature:
        # Use the same algorithm to compute feature vectors
        feature_mat = get_training_data(data_all, imu_columns, option=options, sample_points=sample_points)
    else:
        feature_mat = np.zeros([sample_points.shape[0], 1], dtype=float)
    # Compute the cosine of the angle between camera viewing angle and moving angle
    position_data = data_all[['pos_x', 'pos_y']].values
    feature_mat[:, -1] = compute_delta_angle(time_stamp, position_data, camera_orientation, sample_points)
    return feature_mat


def split_data(data, ratio=0.3):
    """
    Randomly split data set
    :param data: all data
    :param ratio: ratio of hold-off set
    :return: set1, set2
    """
    mask = np.random.random(data.shape[0]) < ratio
    return data[mask], data[~mask]


def test_decompose_speed(data_all):
    """
    Unit test: decompose the speed to 'forward' and 'tangent' direction, and integrate back
    :param data_all:
    :return: positions
    """
    nano_to_sec = 1e09
    num_samples = data_all.shape[0]
    time_stamp = data_all['time'].values / nano_to_sec
    position_xy = data_all[['pos_x', 'pos_y']].values
    orientation = data_all[['ori_w', 'ori_x', 'ori_y', 'ori_z']].values

    step = 1
    sample_points = np.arange(0, num_samples, step, dtype=int)
    moving_dir = (position_xy[sample_points[1:]] - position_xy[sample_points[:-1]]) \
                 / (time_stamp[sample_points[1:]] - time_stamp[sample_points[:-1]])[:, None]
    moving_dir = np.concatenate([moving_dir, [moving_dir[-1]]], axis=0)
    moving_mag = np.linalg.norm(moving_dir, axis=1)
    speed_decomposed = np.zeros([sample_points.shape[0], 2], dtype=float)
    camera_axis_local = quaternion.quaternion(1.0, 0.0, 0.0, -1.0)
    for i in range(sample_points.shape[0]):
        if moving_mag[i] < 1e-011:
            continue
        q = quaternion.quaternion(*orientation[sample_points[i]])
        camera_dir = (q * camera_axis_local * q.conj()).vec[:2]
        cos_theta = np.dot(camera_dir, moving_dir[i]) / (moving_mag[i] * np.linalg.norm(camera_dir))
        sin_theta = math.sqrt(1.0 - cos_theta**2)
        rot_mat = np.array([[cos_theta, -sin_theta], [sin_theta, cos_theta]])
        moving_dir2 = np.dot(rot_mat, camera_dir) * moving_mag[i]
        if i % 100 == 0:
            print('-----------------')
            print(moving_dir[i])
            print(moving_dir2)
        speed_decomposed[i] = np.dot(rot_mat, camera_dir) * moving_mag[i]

    # Get the position by integrating
    import scipy.integrate as integrate
    from utility import write_trajectory_to_ply
    position_inte_xy = np.cumsum((speed_decomposed[1:]+speed_decomposed[:-1]) / 2.0
                                 * (time_stamp[sample_points[1:]] - time_stamp[sample_points[:-1]])[:, None], axis=0)
    position_inte_xy = np.insert(position_inte_xy, 0, 0., axis=0)
    position_inte_xy += position_xy[sample_points[0]]
    position_output = np.concatenate([position_inte_xy, np.zeros([sample_points.shape[0], 1])], axis=1)
    write_trajectory_to_ply.write_ply_to_file('test_decompose_{}.ply'.format(step),
                                              position_output, orientation[sample_points])
    return position_output

# for tests
if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('dir')
    parser.add_argument('--window', default=300, type=int)
    parser.add_argument('--step', default=50, type=int)
    parser.add_argument('--feature', default='direct', type=str)
    parser.add_argument('--frq_threshold', default=50, type=int)
    parser.add_argument('--speed_smooth_sigma', default=0.0, type=float)

    args = parser.parse_args()

    data_dir = args.dir + '/processed'
    print('Loading dataset ' + data_dir + '/data.csv')
    data_all = pandas.read_csv(data_dir + '/data.csv')
    option = TrainingDataOption(window_size=args.window, sample_step=args.step, frq_threshold=args.frq_threshold,
                                feature=args.feature, speed_smooth_sigma=args.speed_smooth_sigma)
    # Create a small sample for testing
    N = data_all.shape[0]
    imu_columns = ['gyro_x', 'gyro_y', 'gyro_z', 'acce_x', 'acce_y', 'acce_z']
    orientation = data_all[['ori_w', 'ori_x', 'ori_y', 'ori_z']].values

    nano_to_sec = 1e09

    time_stamp = data_all['time'].values / nano_to_sec
    speed_decomposed = test_decompose_speed(data_all=data_all)
