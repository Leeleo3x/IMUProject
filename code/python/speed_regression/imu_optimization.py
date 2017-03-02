import warnings
import math
import time
import numpy as np
import quaternion
from scipy.ndimage.filters import gaussian_filter1d
from scipy.optimize import least_squares
from numba import jit

FLAGS = None
nano_to_sec = 1e09


def rotate_vector(input, orientation):
    output = np.empty(input.shape, dtype=float)
    for i in range(input.shape[0]):
        q = quaternion.quaternion(*orientation[i])
        output[i] = (q * quaternion.quaternion(1.0, *input[i]) * q.conj()).vec
    return output

# @jit
# def rotate_vector(input, orientation):
#     output = np.empty(input.shape, dtype=float)
#     for i in range(input.shape[0]):
#         a, b, c, d = orientation[i]
#         rm = np.array([[a*a+b*b-c*c-d*d, 2*b*c-2*a*d, 2*b*d+2*a*c],
#                        [2*b*c+2*a*d, a*a-b*b+c*c-d*d, 2*c*d-a*a*b],
#                        [2*b*d-2*a*c, 2*c*d+2*a*b, a*a-b*b-c*c+d*d]])
#         output[i] = (np.matmul(rm, input[i].transpose())).flatten()
#     return output


class SparseAccelerationBiasCostFunction:
    """
    The cost function is a collection of cost functors, each associated with a weight
    """
    def __init__(self):
        self.functors_ = []
        self.weights_ = []
        self.identifiers_ = []

    def add_functor(self, functor, identifier, weight=1.0):
        self.functors_.append(functor)
        self.weights_.append(weight)
        self.identifiers_ += identifier

    def __call__(self, x, *args, **kwargs):
        """
        Evaluate the residual
        :param x: current state
        :param args:
        :param kwargs:
        :return: a loss vector
        """
        return np.concatenate([self.functors_[i](x) * self.weights_[i] for i in range(len(self.functors_))], axis=0)


class SparseAccelerationBiasFunctor:
    """
    Base class for imu acceleration bias estimation on sparse grid
    """
    identifier_ = 'unknown'

    def __init__(self, time_stamp, orientation, linacce, variable_ind):
        """
        :param time_stamp: time_stamp of the linear acceleration in seconds
        :param linacce: the raw linear acceleration signal
        :param speed_ind: the indice of target speed
        :param variable_ind: the index inside the linacce of optimizing variables
        """
        if time_stamp[-1] > 1e08:
            warnings.warn('The value of time_stamp is large, forgot to convert to seconds?')
        self.variable_ind_ = variable_ind
        self.time_stamp_ = time_stamp
        self.interval_ = (self.time_stamp_[1:variable_ind[-1]] - self.time_stamp_[:variable_ind[-1] - 1])
        # any records after the last speed sample is useless
        self.linacce_ = linacce[:variable_ind[-1]]
        self.orientation_ = orientation[:variable_ind[-1]]

        self.initial_bias_ = np.zeros([1, self.linacce_.shape[1]], dtype=float)
        # Pre-compute the interpolation coefficients
        # y[i] = alpha[i] * x[i-1] + (1.0 - alpha[i]) * x[i]
        self.alpha_ = np.empty((self.variable_ind_[-1]), dtype=float)
        self.alpha_[:variable_ind[0]] = (self.time_stamp_[:variable_ind[0]] - self.time_stamp_[0]) / (self.time_stamp_[
            variable_ind[0]] - self.time_stamp_[0])

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

    identifier_ = 'speed_magnitude'

    def __init__(self, time_stamp, orientation, linacce, target_speed, speed_ind, variable_ind):
        """
        Construct a functor
        :param sigma_a: weighting factor for data term
        :param sigma_s: weighting factor for speed coherence term
        """
        super().__init__(time_stamp, orientation, linacce, variable_ind)
        # predicted speed
        self.target_speed_ = target_speed
        self.speed_ind_ = speed_ind

    #@jit
    def __call__(self, x, *args, **kwargs):
        """
        Evaluate residuals given x
        :param x: 3N array
        :param args:
        :param kwargs:
        :return: The loss vector
        """
        x = x.reshape([-1, self.linacce_.shape[1]])
        # first compute corrected linear acceleration
        # append the initial bias at the end for convenience
        x = np.concatenate([x, self.initial_bias_], axis=0)
        corrected_linacce = self.linacce_ + self.alpha_[:, None] * x[self.inverse_ind_ - 1]\
                                          + (1.0 - self.alpha_[:, None]) * x[self.inverse_ind_]
        # compute corrected speed
        directed_acce = rotate_vector(self.linacce_, self.orientation_)
        directed_speed = np.cumsum((directed_acce[1:] + directed_acce[:-1]) * self.interval_[:, None] / 2.0, axis=0)
        return self.evaluate_on_speed(directed_speed)

    def evaluate_on_speed(self, speed):
        speed = np.linalg.norm(speed, axis=1)
        loss = (speed[self.speed_ind_] - self.target_speed_)
        return loss

    def jac(self, x, *args, **kwargs):
        """
        Evaluate jacobi
        :param x: 3N array
        :return: 3N array
        """
        pass
# end of SpeedMagnitudeFunctor


class ZeroVerticalTranslationFunctor(SparseAccelerationBiasFunctor):

    identifier_ = 'zero_vertical_translation'

    def __init__(self, time_stamp, orientation, linacce, speed_ind, variable_ind):
        super().__init__(time_stamp, orientation, linacce, variable_ind)
        self.speed_ind_ = speed_ind

    def __call__(self, x, *args, **kwargs):
        pass

    def evaluate_on_speed(self, speed):
        return np.cumsum(speed[self.speed_ind_, 2], axis=0)
# end of ZeroVerticalTranslationFunctor


class VerticalSpeedFunctor(SparseAccelerationBiasFunctor):

    identifier_ = 'vertical_speed'

    def __init__(self, time_stamp, orientation, linacce, vertical_speed, speed_ind, variable_ind):
        super().__init__(time_stamp, orientation, linacce, variable_ind)
        self.vertical_speed_ = vertical_speed
        self.speed_ind_ = speed_ind

    def __call__(self, x, *args, **kwargs):
        pass

    def evaluate_on_speed(self, speed):
        return speed[self.speed_ind_, 2] - self.vertical_speed_.flatten()
# end of VerticalSpeedFunctor


class BiasWeightDecay:
    def __call__(self, x, *args, **kwargs):
        return x


class ZeroSpeedFunctor(SparseAccelerationBiasFunctor):
    identifier_ = 'zero_speed'

    def __init__(self, time_stamp, orientation, linacce, speed_ind, variable_ind):
        """
        Constructor for ZeroSpeedFunctor
        :param time_stamp:
        :param linacce:
        :param speed_ind:
        :param variable_ind:
        :param sigma_z:
        """
        super().__init__(time_stamp, orientation, linacce, variable_ind)
        self.speed_ind_ = speed_ind

    def __call__(self, x, *args, **kwargs):
        x = x.reshape([-1, self.linacce_.shape[1]])
        # first add regularization term
        # add data term
        # first compute corrected linear acceleration
        # append the initial bias at the end for convenience
        x = np.concatenate([x, self.initial_bias_], axis=0)
        corrected_linacce = self.linacce_ + self.alpha_[:, None] * x[self.inverse_ind_ - 1] \
                                          + (1.0 - self.alpha_[:, None]) * x[self.inverse_ind_]

        directed_acce = rotate_vector(self.linacce_, self.orientation_)
        speed = np.cumsum((directed_acce[1:] + directed_acce[:-1]) * self.interval_[:, None] / 2.0, axis=0)
        return self.evaluate_on_speed(speed)

    def evaluate_on_speed(self, speed):
        loss = speed[self.speed_ind_ - 1, :].ravel()
        return loss
# end of ZeroSpeedFunctor


class AngleFunctor(SparseAccelerationBiasFunctor):

    identifier_ = 'angle'

    def __init__(self, time_stamp, orientation, linacce, cos_array, speed_ind, variable_ind):
        super().__init__(time_stamp, orientation, linacce, variable_ind)
        self.speed_ind_ = speed_ind
        self.cos_array_ = cos_array
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
        x = x.reshape([-1, self.linacce_.shape[1]])
        x = np.concatenate([x, self.initial_bias_], axis=0)
        corrected_linacce = self.linacce_ + self.alpha_[:, None] * x[self.inverse_ind_ - 1] \
                                          + (1.0 - self.alpha_[:, None]) * x[self.inverse_ind_]
        speed = np.cumsum((corrected_linacce[1:] + corrected_linacce[:-1]) * self.interval_[:, None] / 2.0, axis=0)
        return self.evaluate_on_speed(speed)

    @jit
    def evaluate_on_speed(self, speed):
        speed_mag = np.linalg.norm(speed, axis=1)
        loss = np.zeros(self.speed_ind_.shape[0], dtype=float)
        for i in range(self.speed_ind_.shape[0]):
            if speed_mag[i] > 1e-11:
                loss[i] = np.dot(speed[self.speed_ind_[i]], self.camera_axis_[self.speed_ind_[i]]) \
                          / speed_mag[i] - self.cos_array_[i]
        return loss


class SharedSpeedFunctorSet(SparseAccelerationBiasFunctor):
    """
    A collection of functors that require a shared rotation operation
    """
    def __init__(self, time_stamp, orientation, linacce, variable_ind):
        super().__init__(time_stamp, orientation, linacce, variable_ind)
        self.functors_ = []
        self.weights_ = []
        self.identifiers_ = []

    def add_functor(self, functor, identifier, weight=1.0):
        if len(self.functors_) > 0:
            assert functor.linacce_.shape[0] == self.orientation_.shape[0]
        self.functors_.append(functor)
        self.weights_.append(weight)
        self.identifiers_ += identifier

    def __call__(self, x, *args, **kwargs):
        x = x.reshape([-1, self.initial_bias_.shape[1]])
        x = np.concatenate([x, self.initial_bias_], axis=0)
        corrected_linacce = self.linacce_ + self.alpha_[:, None] * x[self.inverse_ind_ - 1] \
                            + (1.0 - self.alpha_[:, None]) * x[self.inverse_ind_]

        directed_acce = rotate_vector(corrected_linacce, self.orientation_)
        # speed = np.cumsum((directed_acce[1:] + directed_acce[:-1]) * self.interval_[:, None] / 2.0, axis=0)
        speed = np.cumsum(directed_acce[:-1] * self.interval_[:, None], axis=0)
        speed = np.concatenate([np.zeros([1, self.linacce_.shape[1]]), speed], axis=0)
        loss = np.concatenate([self.functors_[i].evaluate_on_speed(speed)
                               * self.weights_[i] for i in range(len(self.functors_))], axis=0)
        return loss


if __name__ == '__main__':
    import argparse
    import sys
    import os
    # sys.path.append(os.path.dirname(os.path.abspath(__file__) + '/..'))
    sys.path.append('/home/yanhang/Documents/research/IMUProject/code/python')

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
    parser.add_argument('--method', type=str, default='speed_and_angle')
    parser.add_argument('--step', type=int, default=10)
    parser.add_argument('--interval', type=int, default=20)
    parser.add_argument('--verbose', type=int, default=2)
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

    # test_N = linacce.shape[0]
    test_N = 8000

    time_stamp = time_stamp[:test_N]
    linacce = linacce[:test_N]
    orientation = orientation[:test_N]

    variable_ind = np.arange(1, test_N, 20, dtype=int)
    variable_ind[-1] = test_N - 1

    """
    Construct constraints
    """
    print('Predicting speed...')
    options = td.TrainingDataOption(feature='fourier', sample_step=FLAGS.step, frq_threshold=100)
    # constraint_ind = np.arange(options.window_size_, test_N - 1,
    #                            options.sample_step_,
    #                            dtype=int)

    constraint_ind = np.arange(0, test_N - 1,
                               options.sample_step_,
                               dtype=int)

    warnings.warn('Currently using ground truth as constraint')
    position_tango = data_all[['pos_x', 'pos_y', 'pos_z']].values
    predicted_speed = td.compute_speed(time_stamp, position_tango, constraint_ind)

    # try smooth the predicted speed
    # predicted_speed = gaussian_filter1d(predicted_speed, sigma=10.0, axis=0)

    predicted_speed_margnitude = np.linalg.norm(predicted_speed, axis=1)

    # predicted_cos_array = None
    # if FLAGS.method == 'speed_and_angle':
    predicted_cos_array, valid_array = td.compute_delta_angle(time_stamp,
                                                              position=position_tango,
                                                              orientation=orientation,
                                                              sample_points=constraint_ind)

    # NOTICE: the values inside the cos_array are not all valid (the speed direction is undefined for static camera).
    #         Therefore it is necessary to construct a separate constraint index array
    constraint_ind_angle = constraint_ind[valid_array]
    predicted_cos_array = predicted_cos_array[valid_array]

    # write predicted speed to file for c++ optimizer
    # with open(FLAGS.dir + '/processed/speed_magnitude.txt', 'w') as f:
    #     f.write('{:d}\n'.format(constraint_ind.shape[0]))
    #     for i in range(constraint_ind.shape[0]):
    #         f.write('{:d} {:f}\n'.format(constraint_ind[i], predicted_speed_margnitude[i]))
    #
    # with open(FLAGS.dir + '/processed/vertical_speed.txt', 'w') as f:
    #     f.write('{:d}\n'.format(constraint_ind.shape[0]))
    #     for i in range(constraint_ind.shape[0]):
    #         f.write('{:d} {:f}\n'.format(constraint_ind[i], predicted_speed[i, 2]))
    #
    # with open(FLAGS.dir + '/processed/cos_array.txt', 'w') as f:
    #     f.write('{:d}\n'.format(constraint_ind_angle.shape[0]))
    #     for i in range(constraint_ind_angle.shape[0]):
    #         f.write('{:d} {:f}\n'.format(constraint_ind_angle[i], predicted_cos_array[i]))

    print('Constructing problem...')
    ##########################################################
    # Constructing problem
    cost_function = SparseAccelerationBiasCostFunction()
    cost_function.add_functor(BiasWeightDecay(), ['weight_decay'], 1.0)

    # Speed related functors
    magnitude_functor = SpeedMagnitudeFunctor(time_stamp, orientation, linacce,
                                              predicted_speed_margnitude, constraint_ind, variable_ind)
    zero_z_translation = ZeroVerticalTranslationFunctor(time_stamp, orientation, linacce, constraint_ind, variable_ind)
    vertical_speed = VerticalSpeedFunctor(time_stamp, orientation, linacce, predicted_speed[:, 2][:, None],
                                          constraint_ind, variable_ind)
    angle_cosine = AngleFunctor(time_stamp, orientation, linacce, predicted_cos_array, constraint_ind_angle, variable_ind)
    speed_constraints = SharedSpeedFunctorSet(time_stamp, orientation, linacce, variable_ind)

    sigma_zp = 100.0
    sigma_a = 10.0
    sigma_s = 1.0
    sigma_vs = 1.0

    speed_constraints.add_functor(magnitude_functor, [magnitude_functor.identifier_], sigma_s)
    # speed_constraints.add_functor(zero_z_translation, sigma_zp)
    # speed_constraints.add_functor(angle_cosine, [angle_cosine.identifier_], sigma_a)
    speed_constraints.add_functor(vertical_speed, [vertical_speed.identifier_], sigma_vs)
    cost_function.add_functor(speed_constraints, speed_constraints.identifiers_)

    print('Solving...')
    print('Functors: ', cost_function.identifiers_)
    output_name = 'speed_magnitude_vertical_speed_{}'.format(test_N)
    max_nfev = 50

    # Optimize
    init_bias = np.zeros(variable_ind.shape[0] * 3, dtype=float)
    start_t = time.clock()
    optimizer = least_squares(cost_function, init_bias, jac='2-point',
                              max_nfev=max_nfev, verbose=FLAGS.verbose)
    print('Time usage: {:.2f}s'.format(time.clock() - start_t))

    corrected_linacce = np.empty(linacce.shape, dtype=float)
    np.copyto(corrected_linacce, linacce)

    bias = optimizer.x.reshape([-1, 3])
    corrected_linacce[:variable_ind[-1]] = magnitude_functor.correct_acceleration(linacce[:variable_ind[-1]], bias)
    directed_corrected = np.empty(corrected_linacce.shape, dtype=float)
    directed_corrected = rotate_vector(corrected_linacce, orientation)

    """
    Visualize and save result
    """
    # computed speed
    time_interval = (time_stamp[1:] - time_stamp[:-1])[:, None]
    linacce = rotate_vector(linacce, orientation)

    filter_sigma = 5
    corrected = gaussian_filter1d(directed_corrected, sigma=filter_sigma, axis=0)
    linacce = gaussian_filter1d(linacce, sigma=filter_sigma, axis=0)
    raw_speed = np.cumsum((linacce[1:] + linacce[:-1]) * time_interval / 2.0, axis=0)
    corrected_speed = np.cumsum((corrected[1:] + corrected[:-1]) * time_interval / 2.0, axis=0)
    # show the double integration result
    position_corrected = IMU_double_integration(time_stamp, orientation, corrected, no_transform=True)
    position_raw = IMU_double_integration(time_stamp, orientation, linacce, no_transform=True)

    raw_path = FLAGS.dir + '/raw.ply'
    write_ply_to_file(raw_path,
                      position=position_raw, orientation=orientation, length=0.5, kpoints=50)

    output_path = FLAGS.dir + '/optimized_' + output_name + '.ply'
    write_ply_to_file(output_path,
                      position=position_corrected, orientation=orientation, length=0.5, kpoints=50)
    print('Result written to ' + output_path)

    # Plot results
    plt.figure('bias')
    legends = 'xyz'
    for i in range(bias.shape[1]):
        plt.subplot(bias.shape[1] * 100 + 11 + i)
        plt.plot(time_stamp[variable_ind], bias[:, i])
        plt.legend(legends[i])

    if SpeedMagnitudeFunctor.identifier_ in cost_function.identifiers_:
        plt.figure('Speed Magnitude')
        plt.plot(time_stamp[constraint_ind], predicted_speed_margnitude)
        plt.plot(time_stamp[1:], np.linalg.norm(raw_speed, axis=1))
        plt.plot(time_stamp[1:], np.linalg.norm(corrected_speed, axis=1))
        plt.legend(['Predicted', 'Raw', 'Corrected'])

    if AngleFunctor.identifier_ in cost_function.identifiers_:
        raw_cos_array, _ = td.compute_delta_angle(time_stamp=time_stamp, position=position_raw,
                                                  orientation=orientation,
                                                  sample_points=constraint_ind_angle)
        corrected_cos_array, _ = td.compute_delta_angle(time_stamp=time_stamp, position=position_corrected,
                                                        orientation=orientation,
                                                        sample_points=constraint_ind_angle)
        plt.figure('Cosine')
        plt.plot(time_stamp[constraint_ind_angle], predicted_cos_array)
        plt.plot(time_stamp[constraint_ind_angle], raw_cos_array)
        plt.plot(time_stamp[constraint_ind_angle], corrected_cos_array)
        plt.legend(['Ground truth', 'Raw', 'Corrected'])

    if VerticalSpeedFunctor.identifier_ in cost_function.identifiers_:
        plt.figure('Vertical speed')
        plt.plot(time_stamp[constraint_ind], predicted_speed[:, 2])
        plt.plot(time_stamp[1:], raw_speed[:, 2])
        plt.plot(time_stamp[1:], corrected_speed[:, 2])
        plt.legend(['Ground truth', 'Raw', 'Corrected'])
    plt.show()
