import argparse
import warnings
import os

from sklearn import svm
from sklearn.metrics import log_loss
from sklearn.externals import joblib
import numpy as np
import matplotlib.pyplot as plt
import pandas

import training_data as td

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('list')
    parser.add_argument('--window', default=300, type=int)
    parser.add_argument('--step', default=50, type=int)
    parser.add_argument('--feature', default='direct', type=str)
    parser.add_argument('--frq_threshold', default=100, type=int)
    parser.add_argument('--only_on', default='', type=str)

    args = parser.parse_args()

    with open(args.list) as f:
        dataset_list = [s.strip('\n') for s in f.readlines()]

    root_dir = os.path.dirname(args.list)
    features_all = []
    targets_all = []

    features_dict = {}
    targets_dict = {}

    for dataset in dataset_list:
        info = dataset.split(',')
        data_path = root_dir + '/' + info[0] + '/processed/data.csv'
        if not os.path.exists(data_path):
            warnings.warn('File ' + data_path + ' does not exist, omit the folder')
            continue
        motion_type = 'unknown'
        if len(info) == 2:
            motion_type = info[1]

        if len(args.only_on) > 0 and args.only_on != motion_type:
            print('Only use ' + args.only_on, ', skip current dataset')
            continue

        print('Loading dataset ' + data_path)
        data_all = pandas.read_csv(data_path)

        print('Creating training set')
        options = td.TrainingDataOption(sample_step=args.step, window_size=args.window,
                                        method=args.feature, frq_threshold=args.frq_threshold)
        data_factory = td.SpeedRegressionTrainData(option=options)

        imu_columns = ['gyro_w', 'gyro_x', 'gyro_y', 'gyro_z', 'acce_x', 'acce_y', 'acce_z']
        features, targets = data_factory.CreateTrainingData(data_all=data_all, imu_columns=imu_columns)
        features_all.append(features)
        targets_all.append(targets)

        # append the dataset to different motion for more detailed performance report
        if motion_type not in features_dict:
            features_dict[motion_type] = [features]
            targets_dict[motion_type] = [targets]
        else:
            features_dict[motion_type].append(features)
            targets_dict[motion_type].append(targets)

    assert len(features_all) > 0, 'No data was loaded'
    features_all = np.concatenate(features_all, axis=0)
    targets_all = np.concatenate(targets_all, axis=0).flatten()
    print('------------------\nProperties')
    print('Dimension of feature matrix: ', features_all.shape)
    print('Dimension of target vector: ', targets_all.shape)
    print('Number of training samples in each category:')
    for k in features_dict.keys():
        features_dict[k] = np.concatenate(features_dict[k], axis=0)
        targets_dict[k] = np.concatenate(targets_dict[k], axis=0).flatten()
    for k, v in features_dict.items():
        print(k + ': {}'.format(v.shape[0]))
    print('Training SVM')
    regressor = svm.SVR(C=5.0, epsilon=0.2)
    regressor.fit(features_all, targets_all)

    print('------------------\nPerformance')
    print('Training score: ', regressor.score(features_all, targets_all))
    for k in features_dict.keys():
        print('Training score on {}: {}'.format(k, regressor.score(features_dict[k], targets_dict[k])))
