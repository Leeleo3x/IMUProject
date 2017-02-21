import warnings
import math
import numpy as np
import quaternion
from scipy.ndimage.filters import gaussian_filter1d
from scipy.optimize import least_squares
from numba import jit

FLAGS = None
nano_to_sec = 1e09


class SparseAccelerationBiasCostFunction:
    """
    The cost function is a collection of cost functors, each associated with a weight
    """
    def __init__(self):
        self.functors_ = []
        self.weights_ = []

    def add_functor(self, functor, weight):
        self.functors_.append(functor)
        self.weights_.append(weight)

    def __call__(self, x, *args, **kwargs):
        """
        Evaluate the residual
        :param x: current state
        :param args:
        :param kwargs:
        :return: a loss vector
        """
        return np.concatenate([self.functors_[i] * self.weights_[i] for i in range(len(self.functors_))], axis=0)


class SparseAccelerationBiasFunctor:
    """
    Base class for imu acceleration bias estimation on sparse grid
    """
    def __init__(self, time_stamp, linacce, speed_ind, variable_ind):
        """
        :param time_stamp: time_stamp of the linear acceleration in seconds
        :param linacce: the raw linear acceleration signal
        :param speed_ind: the indice of target speed
        :param variable_ind: the index inside the linacce of optimizing variables
        """
        if time_stamp[-1] > 1e08:
            warnings.warn('The value of time_stamp is large, forgot to convert to seconds?')
        assert variable_ind[-1] > speed_ind[-1], \
            print('variable_ind[-1]:{}, speed_ind[-1]:{}'.format(variable_ind[-1], speed_ind[-1]))
        self.variable_ind_ = variable_ind
        self.time_stamp_ = time_stamp
        self.interval_ = (self.time_stamp_[1:variable_ind[-1]] - self.time_stamp_[:variable_ind[-1] - 1])
        # any records after the last speed sample is useless
        self.linacce_ = linacce[:variable_ind[-1]]

        self.speed_ind_ = speed_ind
        self.initial_bias_ = np.zeros([1, self.linacce_.shape[1]], dtype=float)
        # Pre-compute the interpolation coefficients
        # y[i] = alpha[i-1] * x[i-1] + (1.0 - alpha[i-1]) * x[i]
        self.alpha_ = np.empty((self.variable_ind_[-1]), dtype=float)
        self.alpha_[:variable_ind[0]] = (self.time_stamp_[:variable_ind[0]] - self.time_stamp_[0]) / self.time_stamp_[
            variable_ind[0]]

        self.inverse_ind_ = np.zeros([variable_ind[-1]], dtype=int)
        for i in range(1, self.variable_ind_.shape[0]):
            self.inverse_ind_[variable_ind[i - 1]:variable_ind[i]] = i
            start_id, end_id = variable_ind[i - 1], variable_ind[i]
            self.alpha_[start_id:end_id] = (self.time_stamp_[start_id:end_id] - self.time_stamp_[start_id]) \
                                           / (self.time_stamp_[end_id] - self.time_stamp_[start_id])
        self.alpha_ = 1.0 - self.alpha_

    def correct_acceleration(self, input_acceleration, bias):
        assert bias.shape[0] == self.variable_ind_.shape[0], \
            'bias.shape[0]: {}, variable_ind.shape[0]: {}'.format(bias.shape[0], self.variable_ind_.shape[0])
        bias = np.concatenate([bias, self.initial_bias_], axis=0)
        corrected_linacce = input_acceleration + self.alpha_[:, None] * bias[self.inverse_ind_ - 1] \
                                               + (1.0 - self.alpha_[:, None]) * bias[self.inverse_ind_]

        return corrected_linacce


class SpeedMagnitudeFunctor(SparseAccelerationBiasFunctor):
    def __init__(self, time_stamp, linacce, target_speed, speed_ind, variable_ind, sigma_r=0.1, sigma_s=0.01):
        """
        Construct a functor
        :param sigma_a: weighting factor for data term
        :param sigma_s: weighting factor for speed coherence term
        """
        super().__init__(time_stamp, linacce, speed_ind, variable_ind)
        # predicted speed
        self.target_speed_ = target_speed
        self.sigma_r_ = sigma_r
        self.sigma_s_ = sigma_s

    #@jit
    def __call__(self, x, *args, **kwargs):
        """
        Evaluate residuals given x
        :param x: 3N array
        :param args:
        :param kwargs:
        :return: The loss vector
        """
        loss = np.copy(x)
        x = x.reshape([-1, self.linacce_.shape[1]])
        # first add regularization term
        # add data term
        # first compute corrected linear acceleration
        # append the initial bias at the end for convenience
        x = np.concatenate([x, self.initial_bias_], axis=0)
        corrected_linacce = self.linacce_ + self.alpha_[:, None] * x[self.inverse_ind_ - 1]\
                            + (1.0 - self.alpha_[:, None]) * x[self.inverse_ind_]
        delta_speed = (corrected_linacce[1:] + corrected_linacce[:-1]) * self.interval_[:, None] / 2.0

        speed = np.linalg.norm(np.cumsum(delta_speed, axis=0), axis=1)
        # Next, compute the difference between integrated speed and target speed.
        # Notice that there is one element's off between $speed and $target_speed.
        loss = np.concatenate([loss, (speed[self.speed_ind_ - 1] - self.target_speed_) * self.sigma_s_], axis=0)
        return loss

    def jac(self, x, *args, **kwargs):
        """
        Evaluate jacobi
        :param x: 3N array
        :return: 3N array
        """
        pass
# end of SpeedMagnitudeFunctor


class ZeroSpeedFunctor(SparseAccelerationBiasFunctor):
    def __init__(self, time_stamp, linacce, speed_ind, variable_ind, sigma_z):
        """
        Constructor for ZeroSpeedFunctor
        :param time_stamp:
        :param linacce:
        :param speed_ind:
        :param variable_ind:
        :param sigma_z:
        """
        super().__init__(time_stamp, linacce, speed_ind, variable_ind)
        self.sigma_z_ = sigma_z
        print(self.inverse_ind_)
    #@jit
    def __call__(self, x, *args, **kwargs):
        loss = np.copy(x)
        x = x.reshape([-1, self.linacce_.shape[1]])
        # first add regularization term
        # add data term
        # first compute corrected linear acceleration
        # append the initial bias at the end for convenience
        x = np.concatenate([x, self.initial_bias_], axis=0)
        corrected_linacce = self.linacce_ + self.alpha_[:, None] * x[self.inverse_ind_ - 1] \
                                          + (1.0 - self.alpha_[:, None]) * x[self.inverse_ind_]
        delta_speed = (corrected_linacce[1:] + corrected_linacce[:-1]) * self.interval_[:, None] / 2.0
        speed = np.cumsum(delta_speed, axis=0)
        speed_loss = np.abs(speed[self.speed_ind_ - 1, :]).ravel() * self.sigma_z_
        loss = np.concatenate([loss, speed_loss], axis=0)
        return loss
# end of ZeroSpeedFunctor


class SpeedAndAngleFunctor(SparseAccelerationBiasFunctor):
    def __init__(self, time_stamp, linacce, orientation,
                 target_speed, cos_array, speed_ind, variable_ind,
                 sigma_s, sigma_a, sigma_r=1.0):
        assert target_speed.shape[0] == cos_array.shape[0], 'target_speed.shape[0]: {}, cos_arrays.shape[0]:{}'\
            .format(target_speed.shape[0], cos_array.shape[0])
        super().__init__(time_stamp, linacce, speed_ind, variable_ind)
        self.target_speed_ = target_speed
        self.cos_array_ = cos_array
        self.sigma_s_ = sigma_s
        self.sigma_a_ = sigma_a
        self.sigma_r_ = sigma_r
        self.camera_axis_ = np.empty(linacce.shape, dtype=float)
        camera_axis_local = quaternion.quaternion(1., 0., 0., -1.)
        for i in range(linacce.shape[0]):
            q = quaternion.quaternion(*orientation[i])
            self.camera_axis_[i] = (q * camera_axis_local * q.conj()).vec[:linacce.shape[1]]
        self.camera_axis_ /= np.linalg.norm(self.camera_axis_, axis=1)[:, None]

    def __call__(self, x, *args, **kwargs):
        """
        Evaluate residuals given x
        :param x: 2N array
        :param args:
        :param kwargs:
        :return: The loss vector
        """
        loss_regularization = np.copy(x)
        x = x.reshape([-1, self.linacce_.shape[1]])
        x = np.concatenate([x, self.initial_bias_], axis=0)
        corrected_linacce = self.linacce_ + self.alpha_[:, None] * x[self.inverse_ind_ - 1] \
                                          + (1.0 - self.alpha_[:, None]) * x[self.inverse_ind_]
        delta_speed = (corrected_linacce[1:] + corrected_linacce[:-1]) * self.interval_[:, None] / 2.0
        speed = np.cumsum(delta_speed, axis=0)
        # speed magnitude
        speed_mag = np.linalg.norm(np.cumsum(delta_speed, axis=0), axis=1)
        loss_speed = (speed_mag[self.speed_ind_ - 1] - self.target_speed_)
        loss_direction = np.zeros(self.speed_ind_.shape[0], dtype=float)
        for i in range(self.speed_ind_.shape[0]):
            if speed_mag[i] > 1e-11:
                loss_direction[i] = np.dot(speed[self.speed_ind_[i]], self.camera_axis_[self.speed_ind_[i]]) \
                                    / speed_mag[i] - self.cos_array_[i]

        # speed direction
        loss = np.concatenate([loss_regularization * self.sigma_r_, loss_speed * self.sigma_s_,
                               loss_direction * self.sigma_a_], axis=0)
        return loss


def optimize_linear_acceleration(time_stamp, directed_acce, speed_ind, param,
                                 constraint_dict=None,
                                 method='speed_magunitude', initial=None, sparse_location=None, verbose=0):
    """
    Optimize linear acceleration with regressed speed
    :param time_stamp: the time stamp of linear acceleration
    :param orientation: orientation at each time stamp in the form of quaternion
    :param directed_acce: oriented linear acceleration
    :param speed_ind: indics of predicted speed
    :param constraint_dict: dictionary of constraints
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
        initial = np.zeros(sparse_location.shape[0] * directed_acce.shape[1], dtype=float)
    assert initial.ndim == 1
    assert initial.shape[0] == directed_acce.shape[1] * sparse_location.shape[0]
    assert orientation.shape[0] == linacce.shape[0]

    if verbose > 0:
        print('Using ' + method)

    if method == 'speed_magnitude':
        cost_functor = SpeedMagnitudeFunctor(time_stamp=time_stamp,
                                             linacce=directed_acce,
                                             target_speed=constraint_dict['speed_magnitude'],
                                             speed_ind=speed_ind,
                                             variable_ind=sparse_location,
                                             sigma_r=param['sigma_r'],
                                             sigma_s=param['sigma_s'])
    elif method == 'speed_and_angle':
        cost_functor = SpeedAndAngleFunctor(time_stamp=time_stamp,
                                            linacce=directed_acce,
                                            orientation=constraint_dict['orientation'],
                                            target_speed=constraint_dict['speed_magnitude'],
                                            cos_array=constraint_dict['cos_array'],
                                            speed_ind=speed_ind,
                                            variable_ind=sparse_location,
                                            sigma_s=param['sigma_s'],
                                            sigma_a=param['sigma_a'],
                                            sigma_r=param['sigma_r'])
    else:
        raise ValueError('Method not supported:' + method)

    optimizer = least_squares(cost_functor, initial, jac='3-point', max_nfev=200, verbose=verbose)
    corrected = directed_acce
    corrected[:sparse_location[-1]] = cost_functor.correct_acceleration(corrected[:sparse_location[-1]],
                                                                        optimizer.x.reshape([-1, directed_acce.shape[1]]))

    # print('Initial speed at constraint:')
    # cum_speed = np.cumsum((linacce[1:] + linacce[:-1]) * (time_stamp[1:] - time_stamp[:-1])[:, None] / 2.0, axis=0)
    # print(cum_speed[speed_ind] - 1)
    # print('final loss vector:')
    # print(cost_functor(optimizer.x)[variable_ind.shape[0]:])
    # corrected_speed = np.cumsum((corrected[1:] + corrected[:-1]) * (time_stamp[1:]-time_stamp[:-1])[:, None] / 2.0, axis=0)
    # print('corrected speed at constraint:')
    # print(corrected_speed[speed_ind - 1])

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
    parser.add_argument('--method', type=str, default='speed_magnitude')
    parser.add_argument('--output', type=str)
    parser.add_argument('--step', type=int, default=50)
    parser.add_argument('--verbose', type=int, default=2)
    parser.add_argument('--sigma_s', type=float, default=0.5)
    parser.add_argument('--sigma_a', type=float, default=1.0)
    parser.add_argument('--sigma_z', type=float, default=1.0)
    parser.add_argument('--sigma_r', type=float, default=1.0)
    FLAGS = parser.parse_args()

    """
    Read the pre-processing the data
    """
    data_all = pandas.read_csv(FLAGS.dir + '/processed/data.csv')
    regressor = joblib.load(FLAGS.model)

    imu_columns = ['gyro_x', 'gyro_y', 'gyro_z', 'linacce_x', 'linacce_y', 'linacce_z']

    time_stamp = data_all['time'].values / nano_to_sec
    time_stamp -= time_stamp[0]

    orientation = data_all[['ori_w', 'ori_x', 'ori_y', 'ori_z']].values
    linacce = data_all[['linacce_x', 'linacce_y', 'linacce_z']].values

    directed_acce = np.empty(linacce.shape, dtype=float)
    for i in range(linacce.shape[0]):
        q = quaternion.quaternion(*orientation[i])
        directed_acce[i] = (q * quaternion.quaternion(1.0, *linacce[i]) * q.conj()).vec

    test_N = linacce.shape[0]

    param = {'sigma_s': FLAGS.sigma_s, 'sigma_z': FLAGS.sigma_z,
             'sigma_r': FLAGS.sigma_r, 'sigma_a': FLAGS.sigma_a}
    variable_ind = np.arange(1, test_N, 200, dtype=int)
    variable_ind[-1] = test_N - 1

    """
    Construct constraints
    """
    constraint_dict = {}
    print('Predicting speed...')
    options = td.TrainingDataOption(feature='fourier', sample_step=FLAGS.step, frq_threshold=100)
    predict_speed_ind = np.arange(options.window_size_, test_N - 1,
                                  options.sample_step_,
                                  dtype=int)
    # variable_ind = speed_ind.copy()
    # variable_ind[-1] += 1.0
    # test_data = td.get_training_data(data_all[:test_N], imu_columns, options, sample_points=predict_speed_ind)
    # predicted_speed = regressor.predict(test_data[:, :-1])
    # predicted_speed = np.maximum(predicted_speed, 0.0)

    warnings.warn('Currently using ground truth as constraint')
    predicted_speed = np.linalg.norm(td.compute_speed(time_stamp, data_all[['pos_x', 'pos_y']].values,
                                                      predict_speed_ind), axis=1)
    constraint_dict['speed_magnitude'] = predicted_speed

    constraint_ind = None
    if FLAGS.method == 'speed_magnitude' or FLAGS.method == 'speed_and_angle':
        constraint_ind = predict_speed_ind
    else:
        raise ValueError('Not implemented')

    predicted_cos_array = None
    if FLAGS.method == 'speed_and_angle':
        predicted_cos_array = td.compute_delta_angle(time_stamp,
                                                     position=data_all[['pos_x', 'pos_y']].values,
                                                     orientation=orientation,
                                                     sample_points=predict_speed_ind)
        constraint_dict['cos_array'] = predicted_cos_array
        constraint_dict['orientation'] = orientation

    print('Solving...')
    optimizer, corrected = optimize_linear_acceleration(time_stamp=time_stamp,
                                                        directed_acce=directed_acce[:, :2],
                                                        speed_ind=constraint_ind,
                                                        constraint_dict=constraint_dict,
                                                        sparse_location=variable_ind,
                                                        method=FLAGS.method,
                                                        param=param,
                                                        verbose=FLAGS.verbose)

    bias = optimizer.x
    # plot the result
    bias = bias.reshape([-1, 2])

    corrected = np.concatenate([corrected, np.zeros([corrected.shape[0], 1], dtype=float)], axis=1)
    """
    Visualize and save result
    """
    # computed speed
    time_interval = (time_stamp[1:] - time_stamp[:-1])[:, None]
    for i in range(linacce.shape[0]):
        rot = quaternion.as_rotation_matrix(quaternion.quaternion(*orientation[i]))
        linacce[i] = np.dot(rot, linacce[i].transpose()).flatten()

    filter_sigma = 30.0
    corrected = gaussian_filter1d(corrected, sigma=filter_sigma, axis=0)
    linacce = gaussian_filter1d(linacce, sigma=filter_sigma, axis=0)
    raw_speed = np.cumsum((linacce[1:] + linacce[:-1]) * time_interval / 2.0, axis=0)
    corrected_speed = np.cumsum((corrected[1:] + corrected[:-1]) * time_interval / 2.0, axis=0)
    # show the double integration result
    orientation = data_all[['ori_w', 'ori_x', 'ori_y', 'ori_z']].values
    position_corrected = IMU_double_integration(time_stamp, orientation, corrected, no_transform=True, only_xy=True)
    position_raw = IMU_double_integration(time_stamp, orientation, linacce, no_transform=True, only_xy=True)
    if FLAGS.output is not None:
        write_ply_to_file(FLAGS.output + '_' + FLAGS.method + '.ply',
                          position=position_corrected, orientation=orientation, length=0.5, kpoints=50)
        write_ply_to_file(FLAGS.output + '_' + FLAGS.method + '_raw.ply',
                          position=position_raw, orientation=orientation, length=0.5, kpoints=50)
        print('Result written to ' + FLAGS.output)

    plt.figure('bias')
    legends = 'xyz'
    for i in range(bias.shape[1]):
        plt.subplot(bias.shape[1] * 100 + 11 + i)
        plt.plot(time_stamp[variable_ind], bias[:, i])
        plt.legend(legends[i])

    plt.figure('Speed Magnitude')
    # plt.plot(time_stamp[predict_speed_ind], predicted_speed)
    plt.plot(time_stamp[predict_speed_ind], predicted_speed)
    plt.plot(time_stamp[1:], np.linalg.norm(raw_speed, axis=1))
    plt.plot(time_stamp[1:], np.linalg.norm(corrected_speed, axis=1))
    plt.legend(['Predicted', 'Raw', 'Corrected'])

    if FLAGS.method == 'speed_and_angle':
        raw_cos_array = td.compute_delta_angle(time_stamp=time_stamp, position=position_raw,
                                               orientation=orientation,
                                               sample_points=predict_speed_ind)
        corrected_cos_array = td.compute_delta_angle(time_stamp=time_stamp, position=position_corrected,
                                                     orientation=orientation,
                                                     sample_points=predict_speed_ind)
        plt.figure('Cosine')
        plt.plot(time_stamp[predict_speed_ind], predicted_cos_array)
        plt.plot(time_stamp[predict_speed_ind], raw_cos_array)
        plt.plot(time_stamp[predict_speed_ind], corrected_cos_array)
        plt.legend(['Ground truth', 'Raw', 'Corrected'])

    plt.show()
