import argparse

import numpy as np
import pandas
from scipy.fftpack import fft
import matplotlib.pyplot as plt


class TrainingDataOption:
    def __init__(self, sample_step=50, window_size=300, feature='direct', frq_threshold=100):
        self.sample_step_ = sample_step
        self.window_size_ = window_size
        self.frq_threshold_ = frq_threshold
        self.feature_ = feature
        self.nanoToSec = 1000000000.0


class SpeedRegressionTrainData:
    # static variables

    def __init__(self, option):
        self.option_ = option

    @staticmethod
    def compute_speed(pose_data, time, ind):
        return np.linalg.norm(pose_data[ind+1] - pose_data[ind-1]) / (time[ind+1] - time[ind-1])

    def compute_fourier_feature(self, data):
        """
        Compute fourier coefficients as feature vector
        :param data: NxM array for N samples with M dimensions
        :return: Nxk array
        """
        theta = self.option_.frq_threshold_
        farray = fft(data, axis=0)
        return np.abs(farray[1:theta, :])

    def CreateTrainingData(self, data_all, imu_columns):
        """
        Create training data.
        :param data_all: The whole dataset. Must include 'time' column and all columns inside imu_columns
        :param imu_columns: Columns used for constructing feature vectors. Fields must exist in the dataset
        :return: Two pandas.DataFrames: one for feature, one for target
        """
        N = data_all.shape[0]
        sample_points = np.arange(self.option_.window_size_,
                                  N - 1,
                                  self.option_.sample_step_,
                                  dtype=int)
        pose_data = data_all[['pos_x', 'pos_y', 'pos_z']]
        data_used = data_all[imu_columns]

        if self.option_.feature_ == 'direct':
            local_imu_list = [list(data_used[ind-self.option_.window_size_:ind].values.flatten())
                              for ind in sample_points]
        elif self.option_.feature_ == 'fourier':
            local_imu_list = [self.compute_fourier_feature(data_used[ind-self.option_.window_size_:ind].values)
                                  .flatten('F')
                              for ind in sample_points]
        else:
            print('Feature type not supported: ' + self.option_.feature_)
            raise ValueError

        speed = [self.compute_speed(pose_data.values, data_all['time'].values, ind) * self.option_.nanoToSec
                 for ind in sample_points]

        return pandas.DataFrame(local_imu_list), pandas.DataFrame(speed)

# for tests
if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('dir')
    parser.add_argument('--window', default=300, type=int)
    parser.add_argument('--step', default=50, type=int)
    parser.add_argument('--feature', default='direct', type=str)
    parser.add_argument('--frq_threshold', default=100, type=int)

    args = parser.parse_args()

    data_dir = args.dir + '/processed'
    print('Loading dataset ' + data_dir + '/data.csv')
    data_all = pandas.read_csv(data_dir + '/data.csv')

    option = TrainingDataOption()
    data_factory = SpeedRegressionTrainData(option)

    # Create a small sample for testing
    N = data_all.shape[0]
    imu_columns = ['gyro_w', 'gyro_x', 'gyro_y', 'gyro_z', 'acce_x', 'acce_y', 'acce_z']
    data_sample = data_all[imu_columns][1000:1500]

    # testing for fourier feature
    fourier_vector = data_factory.compute_fourier_feature(data_sample)
    plt.figure()
    for i in range(fourier_vector.shape[1]):
        # will ignore the direct component for clearance
        plt.plot(fourier_vector[1:, i])
    plt.legend(imu_columns)
    plt.show()
