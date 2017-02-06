import numpy as np
from scipy.optimize import least_squares
import scipy.integrate as integrate
FLAGS = None

class SpeedFunctor:
    def __init__(self, time_stamp, linacce, variable_ind, sigma_a=0.1, sigma_s=0.01):
        """
        Construct a functor
        :param time_stamp: time_stamp of the linear acceleration
        :param linacce: the raw linear acceleration signal
        :param variable_ind: the index inside the linacce of optimizing variables
        :param sigma_a: weighting factor for data term
        :param sigma_s: weighting factor for speed coherence term
        """
        self.variable_ind_ = variable_ind
        # any records after the last speed sample is useless
        self.time_stamp_ = time_stamp[:variable_ind[-1]]
        self.linacce_ = linacce[:variable_ind[-1]]
        self.sigma_a_ = sigma_a
        self.sigma_s_ = sigma_s
        # initial bias: bias at time 0
        self.initial_bias_ = np.array([0., 0., 0.])
        # Pre-compute the interpolation coefficients
        # y[i] = alpha[i-1] * x[i-1] + (1.0 - alpha[i-1]) * x[i]
        self.alpha_ = np.empty(self.time_stamp_.shape[0], dtype=float)
        self.alpha_[:self.variable_ind_[0]] = self.time_stamp_[:self.variable_ind_[0]] / self.variable_ind_[0]
        self.inverse_ind_ = np.empty(self.time_stamp_.shape[0], dtype=int)
        for i in range(1, self.variable_ind_.shape[0]):
            self.alpha_[variable_ind[i-1]:variable_ind[i]] = \
                (self.time_stamp_[variable_ind[i-1]:variable_ind[i]] - variable_ind[i]) / (variable_ind[i] - variable_ind[i-1])

    def __call__(self, x, *args, **kwargs):
        """
        Evaluate residuals given x
        :param x: 3N array
        :param args:
        :param kwargs:
        :return:
        """
        x = x.reshape([-1, 3])
        # first add regularization term
        loss = np.linalg.norm(x, axis=1)
        # add data term
        # first compute corrected linear acceleration

    def jac(self, x, *args, **kwargs):
        """
        Evaluate jacobi
        :param x: 3N array
        :return: 3N array
        """
        pass

def optimize_linear_acceleration(time_stamp, linacce,  speed_timestamp, regressed_speed, sparse_location = None):
    """
    Optimize linear acceleration with regressed speed
    :param time_stamp: the time stamp of linear acceleration
    :param linacce: linear acceleration
    :param speed_timestamp:
    :param regressed_speed:
    :param sparse_location: position of sampled location. Default is the same with speed timestamp
    :return: refined linear acceleration
    """
    if sparse_location is None:
        sparse_location = speed_timestamp

if __name__ == '__main__':
    import argparse
    import matplotlib.pyplot as plt
    import pandas
    import sklearn.svm as svm
    import training_data as td

    from sklearn.externals import joblib

    parser = argparse.ArgumentParser()
    parser.add_argument('dir', type=str)
    parser.add_argument('calibration', type=str)
    parser.add_argument('model', type=str)
    parser.add_argument('--output', type=str)
    parser.add_argument('--step', type=int)

    FLAGS = parser.parse_args()

    data_all = pandas.read_csv(FLAGS.dir + '/processed/data.csv')
    regressor = joblib.load(FLAGS.model)

    imu_columns = ['gyro_x', 'gyro_y', 'gyro_z', 'linacce_x', 'linacce_y', 'linacce_z']
    options = td.TrainingDataOption(feature='fourier', sample_step=FLAGS.step)
    test_data = td.get_training_data(data_all, imu_columns, options)

    speed = regressor.predict(test_data)

    speed_ind = np.arange(options.window_size_, speed.shape[0] - 1,
                          options.sample_step_,
                          dtype=int)
