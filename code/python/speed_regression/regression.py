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
    parser.add_argument('--split_ratio', default=0.25, type=float)
    parser.add_argument('--output', default='', type=str)
    parser.add_argument('--C', default=None, type=float)
    parser.add_argument('--e', default=None, type=float)

    args = parser.parse_args()

    with open(args.list) as f:
        dataset_list = [s.strip('\n') for s in f.readlines()]

    root_dir = os.path.dirname(args.list)
    features_all = []
    targets_all = []

    features_dict = {}
    targets_dict = {}

    for dataset in dataset_list:
        if len(dataset) > 0 and dataset[0] == '#':
            continue

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
                                        feature=args.feature, frq_threshold=args.frq_threshold)
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
    bestC = 0
    bestE = 0

    total_samples = targets_all.shape[0]
    validation_mask = np.random.rand(total_samples) < args.split_ratio
    features_validation = features_all[validation_mask]
    targets_validation = targets_all[validation_mask]
    features_train = features_all[~validation_mask]
    targets_train = targets_all[~validation_mask]
    assert targets_validation.shape[0] + targets_train.shape[0] == total_samples
    print('Size of training set: {}, validation set: {}'.format(targets_train.shape[0], targets_validation.shape[0]))

    if args.C is None or args.e is None:
        # run grid search
        bestScore = -np.inf

        for c in [0.01, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 30.0, 50.0]:
            for e in [0.01, 0.05, 0.1, 0.5, 1.0, 5.0, 10.0]:
                print('**********************\nTry with c={}, e={}'.format(c, e))
                regressor = svm.SVR(C=c, epsilon=e)
                regressor.fit(features_train, targets_train)

                valid_score = regressor.score(features_validation, targets_validation)
                train_score = regressor.score(features_train, targets_train)

                if valid_score > bestScore:
                    bestC = c
                    bestE = e
                    bestScore = valid_score
                print('Training score: {}, validation score: {}'.format(train_score, valid_score))
                # for k in features_dict.keys():
                #     print('Training score on {}: {}'.format(k, regressor.score(features_dict[k], targets_dict[k])))
        print('All done. Optimal parameter: C={}, e={}, score={}'.format(bestC, bestE, bestScore))
    else:
        bestC = args.C
        bestE = args.e

    print('Train with parameter C={}, e={}'.format(bestC, bestE))
    regressor = svm.SVR(C=bestC, epsilon=bestE)
    regressor.fit(features_train, targets_train)
    print('Training score: {}, validation score: {}'
          .format(regressor.score(features_train, targets_train),
                  regressor.score(features_validation, targets_validation)))
    if len(args.output) > 0:
        joblib.dump(regressor, args.output)
        print('Model written to ' + args.output)
