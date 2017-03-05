import time
import argparse
import warnings
import os

from sklearn import svm
from sklearn.externals import joblib
from sklearn.model_selection import GridSearchCV
import numpy as np
import pandas

# from speed_regression import training_data as td
# from speed_regression import grid_search
import training_data as td
from sklearn.linear_model import SGDRegressor
from sklearn.datasets import dump_svmlight_file

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('list')
    parser.add_argument('--window', default=200, type=int)
    parser.add_argument('--step', default=10, type=int)
    parser.add_argument('--feature', default='direct', type=str)
    parser.add_argument('--target', default='local_speed', type=str)
    parser.add_argument('--frq_threshold', default=50, type=int)
    parser.add_argument('--discard_direct', default=True, type=bool)
    parser.add_argument('--split_ratio', default=0.3, type=float)
    parser.add_argument('--output', default='', type=str)
    parser.add_argument('--svmlight_path', default=None, type=str)
    parser.add_argument('--C', default=None, type=float)
    parser.add_argument('--e', default=None, type=float)
    parser.add_argument('--g', default=None, type=float)

    args = parser.parse_args()

    with open(args.list) as f:
        dataset_list = [s.strip('\n') for s in f.readlines()]

    root_dir = os.path.dirname(args.list)
    training_feature_all = []
    training_target_all = []
    training_dict = {}

    options = td.TrainingDataOption(sample_step=args.step, window_size=args.window,
                                    feature=args.feature, target=args.target)

    for dataset in dataset_list:
        if len(dataset) > 0 and dataset[0] == '#':
            continue

        info = dataset.split(',')
        data_path = root_dir + '/' + info[0] + '/processed/data.csv'
        if not os.path.exists(data_path):
            warnings.warn('File ' + data_path + ' does not exist, omit the folder')
            continue

        print('Loading dataset ' + data_path)
        data_all = pandas.read_csv(data_path)

        print('Creating training set')
        imu_columns = ['gyro_x', 'gyro_y', 'gyro_z', 'linacce_x', 'linacce_y', 'linacce_z']

        extra_args = {'frq_threshold': args.frq_threshold,
                      'discard_direct': args.discard_direct,
                      'target_smooth_sigma': 10}
        training_feature, training_target = td.get_training_data(data_all=data_all, imu_columns=imu_columns,
                                                                 option=options, extra_args=extra_args)
        training_feature_all.append(training_feature)
        training_target_all.append(training_target)

    assert len(training_feature_all) > 0, 'No data was loaded'
    training_feature_all = np.concatenate(training_feature_all, axis=0)
    training_target_all = np.concatenate(training_target_all, axis=0)

    print("{} samples in total".format(training_target_all.shape[0]))

    # optionally write dataset in svmlight format
    # if args.svmlight_path is not None:
    #     for chn in range(training_target_all.shape[1]):
    #         path = args.svmlight_path + '_{}'.format(chn)
    #         with open(path, 'wb') as f:
    #             dump_svmlight_file(training_feature_all, training_target_all[:, chn], f)
    #             print('Dataset written to ' + path)

    # Training separate regressor for each target channel

    if training_target_all.ndim == 1:
        training_target_all = training_target_all[:, None]

    for chn in range(training_target_all.shape[1]):
        print('Training SVM for target ', chn)
        bestC = 0
        bestE = 0

        regressor = None
        if args.C is None or args.e is None or args.g is None:
            print('Running grid search')

            # search_dict = {'epsilon': [0.001, 0.01, 0.1, 1.0, 10.0],
            #                'loss': ['squared_loss', 'huber', 'epsilon_insensitive']}
            # grid_searcher = GridSearchCV(SGDRegressor(loss='epsilon_insensitive'), search_dict, n_jobs=6, verbose=3)

            search_dict = {'C': [0.1, 1.0, 10.0],
                           'epsilon': [0.001, 0.01, 0.1, 1.0],
                           'kernel': ['rbf']}
            grid_searcher = GridSearchCV(svm.SVR(), search_dict, n_jobs=6, verbose=3)

            start_t = time.clock()
            grid_searcher.fit(training_feature_all, training_target_all[:, chn])

            time_passage = time.clock() - start_t
            print('All done, time usage: {:.3f}s ({:.3f}h)'.format(time_passage, time_passage / 3600.0))
            print('Optimal parameter: ', grid_searcher.best_params_)
            print('Best score: ', grid_searcher.best_score_)
            regressor = grid_searcher.best_estimator_
        else:
            print('Train with parameter C={}, e={}, gamma={}'.format(args.C, args.e, args.g))
            regressor = svm.SVR(C=args.C, epsilon=args.e, gamma=args.g)
            regressor.fit(training_feature_all, training_target_all[:, chn])
            # score = mean_squared_error(regressor.predict(training_set_all[:, :-1]), training_set_all[:, -1])
            score = regressor.score(training_feature_all, training_target_all[:, chn])
            print('Training score:', score)

        # write model to file
        if len(args.output) > 0:
            out_path = '{}_{}'.format(args.output, chn)
            joblib.dump(regressor, out_path)
            print('Model written to ' + out_path)
