import warnings
import numpy as np
import quaternion
from scipy.optimize import least_squares

FLAGS = None


class SpeedMagnitudeFunctor:
    def __init__(self, time_stamp, linacce, target_speed, speed_ind, variable_ind, sigma_a=0.1, sigma_s=0.01):
        """
        Construct a functor
        :param time_stamp: time_stamp of the linear acceleration in seconds
        :param linacce: the raw linear acceleration signal
        :param speed_ind: the indice of target speed
        :param variable_ind: the index inside the linacce of optimizing variables
        :param sigma_a: weighting factor for data term
        :param sigma_s: weighting factor for speed coherence term
        """
        if time_stamp[0] > 1e08:
            warnings.warn('The value of time_stamp is large, forgot to convert to seconds?')
        assert variable_ind[-1] > speed_ind[-1], \
            print('variable_ind[-1]:{}, speed_ind[-1]:{}'.format(variable_ind[-1], speed_ind[-1]))
        self.variable_ind_ = variable_ind
        self.time_stamp_ = time_stamp
        self.interval_ = (self.time_stamp_[1:variable_ind[-1]] - self.time_stamp_[:variable_ind[-1]-1])
        # any records after the last speed sample is useless
        self.linacce_ = linacce[:variable_ind[-1]]
        # Store the rotation matrix

        # predicted speed
        self.target_speed_ = target_speed
        self.speed_ind_ = speed_ind

        self.sigma_a_ = sigma_a
        self.sigma_s_ = sigma_s
        # initial bias: bias at time 0
        self.initial_bias_ = np.array([[0., 0., 0.]])
        # Pre-compute the interpolation coefficients
        # y[i] = alpha[i-1] * x[i-1] + (1.0 - alpha[i-1]) * x[i]
        self.alpha_ = np.empty((self.variable_ind_[-1]), dtype=float)
        self.alpha_[:variable_ind[0]] = (self.time_stamp_[:variable_ind[0]] - self.time_stamp_[0]) / self.time_stamp_[variable_ind[0]]

        self.inverse_ind_ = np.zeros([variable_ind[-1]], dtype=int)
        for i in range(1, self.variable_ind_.shape[0]):
            self.inverse_ind_[variable_ind[i-1]:variable_ind[i]] = i
            start_id, end_id = variable_ind[i-1], variable_ind[i]
            self.alpha_[start_id:end_id] = (self.time_stamp_[start_id:end_id] - self.time_stamp_[start_id]) \
                                           / (self.time_stamp_[end_id] - self.time_stamp_[start_id])
        self.alpha_ = 1.0 - self.alpha_

    def __call__(self, x, *args, **kwargs):
        """
        Evaluate residuals given x
        :param x: 3N array
        :param args:
        :param kwargs:
        :return: The loss vector
        """
        x = x.reshape([-1, 3])
        loss = np.linalg.norm(x, axis=1)
        # first add regularization term
        # add data term
        # first compute corrected linear acceleration
        # append the initial bias at the end for convenience
        x = np.concatenate([x, self.initial_bias_], axis=0)
        corrected_linacce = self.linacce_ + self.alpha_[:, None] * x[self.inverse_ind_ - 1] \
                                          + (1.0 - self.alpha_[:, None]) * x[self.inverse_ind_]
        delta_speed = (corrected_linacce[1:] + corrected_linacce[:-1]) * self.interval_[:, None] / 2.0

        speed = np.linalg.norm(np.cumsum(delta_speed, axis=0), axis=1)
        # Next, compute the difference between integrated speed and target speed.
        # Notice that there is one element's off between $speed and $target_speed.
        loss = np.concatenate([loss, speed[self.speed_ind_ - 1] - self.target_speed_], axis=0) * self.sigma_s_
        return loss

    def jac(self, x, *args, **kwargs):
        """
        Evaluate jacobi
        :param x: 3N array
        :return: 3N array
        """
        pass

    def correct_acceleration(self, input_acceleration, bias):
        assert bias.shape[0] == self.variable_ind_.shape[0],\
            'bias.shape[0]: {}, variable_ind.shape[0]: {}'.format(bias.shape[0], self.variable_ind_.shape[0])
        bias = np.concatenate([bias, self.initial_bias_], axis=0)
        corrected_linacce = input_acceleration + self.alpha_[:, None] * bias[self.inverse_ind_ - 1] \
                                               + (1.0 - self.alpha_[:, None]) * bias[self.inverse_ind_]

        return corrected_linacce


def optimize_linear_acceleration(time_stamp, orientation, linacce,  speed_ind, regressed_speed, param,
                                 initial=None, sparse_location=None, verbose=0):
    """
    Optimize linear acceleration with regressed speed
    :param time_stamp: the time stamp of linear acceleration
    :param orientation: orientation at each time stamp in the form of quaternion
    :param linacce: linear acceleration
    :param speed_ind: indics of predicted speed
    :param regressed_speed:
    :param param: dictionary of parameters.
    :param initial: initial value. Default is all zero
    :param sparse_location: position of sampled location. Default is the same with speed timestamp
    :param method: Method to use.
    :param verbose: verbose level
    :return: refined linear acceleration
    """
    if sparse_location is None:
        sparse_location = speed_ind.copy()
        sparse_location[-1] += 1
    if initial is None:
        initial = np.zeros(sparse_location.shape[0] * 3, dtype=float)
    assert initial.shape[0] == sparse_location.shape[0] * 3
    assert initial.ndim == 1
    assert orientation.shape[0] == linacce.shape[0]

    # first convert the acceleration to the global coordinate frame
    directed_acce = np.empty(linacce.shape, dtype=float)
    for i in range(orientation.shape[0]):
        rot = quaternion.as_rotation_matrix(quaternion.quaternion(*orientation[i]))
        directed_acce[i] = np.dot(rot, linacce[i, :].transpose()).flatten()

    cost_functor = SpeedMagnitudeFunctor(time_stamp=time_stamp,
                                         linacce=directed_acce,
                                         target_speed=regressed_speed,
                                         speed_ind=speed_ind,
                                         variable_ind=sparse_location,
                                         sigma_a=param['sigma_a'],
                                         sigma_s=param['sigma_s'])
    optimizer = least_squares(cost_functor, initial, jac='3-point', verbose=verbose)
    corrected = linacce.copy()
    corrected[:sparse_location[-1]] = cost_functor.correct_acceleration(corrected[:sparse_location[-1]],
                                                                        optimizer.x.reshape([-1, 3]))
    return optimizer, corrected


if __name__ == '__main__':
    import time
    import argparse
    import sys
    import os
    sys.path.append(os.path.dirname(os.path.abspath(__file__) + '/..'))
    from algorithms.double_integration import IMU_double_integration
    from utility.write_trajectory_to_ply import write_ply_to_file
    import matplotlib.pyplot as plt
    import pandas
    import sklearn.svm as svm

    import training_data as td
    from sklearn.externals import joblib

    parser = argparse.ArgumentParser()
    parser.add_argument('dir', type=str)
    parser.add_argument('model', type=str)
    parser.add_argument('--calibration', type=str, default=None)
    parser.add_argument('--output', type=str)
    parser.add_argument('--step', type=int, default=50)
    parser.add_argument('--verbose', type=int, default=2)
    parser.add_argument('--sigma_a', type=float, default=0.1)
    parser.add_argument('--sigma_s', type=float, default=0.1)

    FLAGS = parser.parse_args()

    data_all = pandas.read_csv(FLAGS.dir + '/processed/data.csv')
    regressor = joblib.load(FLAGS.model)

    imu_columns = ['gyro_x', 'gyro_y', 'gyro_z', 'linacce_x', 'linacce_y', 'linacce_z']

    time_stamp = data_all['time'].values
    time_stamp -= time_stamp[0]

    orientation = data_all[['ori_w', 'ori_x', 'ori_y', 'ori_z']].values
    linacce = data_all[['linacce_x', 'linacce_y', 'linacce_z']].values

    test_N = linacce.shape[0]

    print('Predicting speed...')
    options = td.TrainingDataOption(feature='fourier', sample_step=FLAGS.step, frq_threshold=100)
    speed_ind = np.arange(options.window_size_, test_N - 1,
                          options.sample_step_,
                          dtype=int)
    test_data = td.get_training_data(data_all[:test_N], imu_columns, options, sample_points=speed_ind)
    predicted_speed = regressor.predict(test_data[:, :-1])
    predicted_speed = np.maximum(predicted_speed, 0.0)

    variable_ind = speed_ind.copy()
    variable_ind[-1] += 1.0

    plt.figure('Speed Regression')
    plt.plot(time_stamp[speed_ind], test_data[:, -1])
    plt.plot(time_stamp[speed_ind], predicted_speed)
    plt.legend(['Ground truth', 'Predicted'])

    print('Solving...')
    start_t = time.clock()
    optimizer, corrected = optimize_linear_acceleration(time_stamp=time_stamp,
                                                        orientation=orientation,
                                                        linacce=linacce,
                                                        speed_ind=speed_ind,
                                                        regressed_speed=predicted_speed,
                                                        param={'sigma_s': FLAGS.sigma_s, 'sigma_a': FLAGS.sigma_a},
                                                        verbose=FLAGS.verbose)
    bias = optimizer.x
    # plot the result
    bias = bias.reshape([-1, 3])

    # show the double integration result
    if FLAGS.output is not None:
        print('Writing corrected trajectory...')
        orientation = data_all[['ori_w', 'ori_x', 'ori_y', 'ori_z']].values
        integrated_position = IMU_double_integration(time_stamp, orientation, corrected)
        write_ply_to_file(FLAGS.output, position=integrated_position, orientation=orientation)

    plt.figure('Corrected')
    plt.plot(time_stamp, corrected)
    plt.legend(['x', 'y', 'z'])
    plt.figure('bias')
    legends = 'xyz'
    for i in range(bias.shape[1]):
        plt.subplot(311 + i)
        plt.plot(variable_ind, bias[:, i])
        plt.legend(legends[i])
    plt.show()
