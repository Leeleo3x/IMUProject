import argparse

import numpy as np
import pandas
from scipy.fftpack import fft
from scipy.ndimage.filters import gaussian_filter1d
import matplotlib.pyplot as plt


class TrainingDataOption:
    def __init__(self, sample_step=50, window_size=300, feature='direct', frq_threshold=100,
                 feature_smooth_sigma=None, speed_smooth_sigma=0.0):
        self.sample_step_ = sample_step
        self.window_size_ = window_size
        self.frq_threshold_ = frq_threshold
        self.feature_ = feature
        self.nanoToSec = 1000000000.0
        self.feature_smooth_sigma_ = feature_smooth_sigma
        self.speed_smooth_sigma_ = speed_smooth_sigma


def compute_fourier_feature(data, threshold):
    """
    Compute fourier coefficients as feature vector
    :param data: NxM array for N samples with M dimensions
    :return: Nxk array
    """
    farray = fft(data, axis=0)
    return np.abs(farray[1:threshold, :])


def get_training_data(data_all, imu_columns, option):
    """
    Create training data.
    :param data_all: The whole dataset. Must include 'time' column and all columns inside imu_columns
    :param imu_columns: Columns used for constructing feature vectors. Fields must exist in the dataset
    :return: [Nx(d+1)] array. Target value is appended at back
    """
    N = data_all.shape[0]
    speed_diff_r = 50
    sample_points = np.arange(option.window_size_,
                              N - 1,
                              option.sample_step_,
                              dtype=int)
    pose_data = data_all[['pos_x', 'pos_y', 'pos_z']].values
    data_used = data_all[imu_columns]
    time_stamp = data_all['time'].values / 1e09

    speed_all = np.zeros(N)
    speed_all[1:-1] = np.divide(np.linalg.norm(pose_data[2:] - pose_data[:-2], axis=1),
                                time_stamp[2:] - time_stamp[:-2])
    # filter the speed with gaussian filter
    if option.speed_smooth_sigma_ > 0:
        speed_all = gaussian_filter1d(speed_all, option.speed_smooth_sigma_, axis=0)

    if option.feature_ == 'direct':
        local_imu_list = [list(data_used[ind-option.window_size_:ind].values.flatten())
                          for ind in sample_points]
    elif option.feature_ == 'fourier':
        local_imu_list = [compute_fourier_feature(data_used[ind-option.window_size_:ind].values, option.frq_threshold_)
                          .flatten('F') for ind in sample_points]
    else:
        print('Feature type not supported: ' + option.feature_)
        raise ValueError

    return np.concatenate([np.array(local_imu_list), speed_all[sample_points, None]], axis=1)


def split_data(data, ratio=0.3):
    """
    Randomly split data set
    :param data: all data
    :param ratio: ratio of hold-off set
    :return: set1, set2
    """
    mask = np.random.random(data.shape[0]) < ratio
    return data[mask], data[~mask]


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
    data_factory = SpeedRegressionTrainingData(option)

    # Create a small sample for testing
    N = data_all.shape[0]
    imu_columns = ['gyro_w', 'gyro_x', 'gyro_y', 'gyro_z', 'acce_x', 'acce_y', 'acce_z']

    print('Test smoothing speed data')
    speed_smoothed = data_factory.get_training_data(data_all, imu_columns=imu_columns)[:, -1]

    plt.plot(speed_smoothed)
    plt.show()
