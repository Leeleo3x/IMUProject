import time
import os
import warnings

import numpy as np
from sklearn import gaussian_process as gaussian_process
from sklearn.externals import joblib
import training_data as td
import pandas

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('list', type=str)
    parser.add_argument('--output', type=str)
    parser.add_argument('--step', type=int, default=10)
    parser.add_argument('--window', type=int, default=200)
    parser.add_argument('--feature', type=str, default='fourier')
    parser.add_argument('--alpha', default=1e-10, type=float)
    parser.add_argument('--discard_direct', default=False, type=bool)

    args = parser.parse_args()

    data_root = os.path.dirname(args.list)

    with open(args.list) as f:
        datasets = f.readlines()

    options = td.TrainingDataOption(feature=args.feature, sample_step=args.step, window_size=args.window)
    imu_columns = ['gyro_w', 'gyro_x', 'gyro_y', 'gyro_z', 'linacce_x', 'linacce_y', 'linacce_z']
    training_set_all = []

    for data in datasets:
        if len(data) == 0:
            continue
        if data[0] == '#':
            continue
        info = data.strip('\n').split(',')
        data_path = data_root + '/' + info[0] + '/processed/data.csv'
        if not os.path.exists(data_path):
            warnings.warn('File ' + data_path + ' does not exist, omit the folder')
            continue

        print('Loading ' + info[0])
        data_all = pandas.read_csv(data_root + '/' + info[0].strip('\n') + '/processed/data.csv')
        training_set_all.append(td.get_training_data(data_all, imu_columns=imu_columns, option=options))

    training_set_all = np.concatenate(training_set_all, axis=0)
    print('Brief: training sample: {}, feature dimension: {}'
          .format(training_set_all.shape[0], training_set_all.shape[1]-1))

    # Train the gaussian process
    regressor = gaussian_process.GaussianProcessRegressor(alpha=args.alpha)
    print('Fitting Gaussian Process...')
    start_t = time.clock()
    regressor.fit(training_set_all[:, :-1], training_set_all[:, -1])
    print('Done in {:.2f} seconds'.format(time.clock() - start_t))

    if len(args.output) > 0:
        joblib.dump(regressor, args.output)
        print('Model saved at ' + args.output)
