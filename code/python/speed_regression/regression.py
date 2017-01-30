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
import grid_search

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('list')
    parser.add_argument('--window', default=300, type=int)
    parser.add_argument('--step', default=10, type=int)
    parser.add_argument('--feature', default='direct', type=str)
    parser.add_argument('--frq_threshold', default=50, type=int)
    parser.add_argument('--only_on', default='', type=str)
    parser.add_argument('--split_ratio', default=0.3, type=float)
    parser.add_argument('--output', default='', type=str)
    parser.add_argument('--C', default=None, type=float)
    parser.add_argument('--e', default=None, type=float)
    parser.add_argument('--g', default=None, type=float)

    args = parser.parse_args()

    with open(args.list) as f:
        dataset_list = [s.strip('\n') for s in f.readlines()]

    root_dir = os.path.dirname(args.list)
    training_set_all = []
    training_dict = {}

    options = td.TrainingDataOption(sample_step=args.step, window_size=args.window,
                                    feature=args.feature, frq_threshold=args.frq_threshold)

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
        imu_columns = ['gyro_w', 'gyro_x', 'gyro_y', 'gyro_z', 'linacce_x', 'linacce_y', 'linacce_z']
        training_set = td.get_training_data(data_all=data_all, imu_columns=imu_columns, option=options)
        training_set_all.append(training_set)

        # append the dataset to different motion for more detailed performance report
        if motion_type not in training_dict:
            training_dict[motion_type] = [training_set]
        else:
            training_dict[motion_type].append(training_set)

    assert len(training_set_all) > 0, 'No data was loaded'
    training_set_all = np.concatenate(training_set_all, axis=0)
    print('------------------\nProperties')
    print('Number of training samples in each category:')
    for k in training_dict.keys():
        training_dict[k] = np.concatenate(training_dict[k], axis=0)
    for k, v in training_dict.items():
        print(k + ': {}'.format(v.shape[0]))

    print('Training SVM')
    bestC = 0
    bestE = 0

    regressor = None
    if args.C is None or args.e is None or args.g is None:
        # run grid search
        # search_dict = {'c': [0.1, 0.5, 1.0, 5.0, 1.0, 20.0, 30.0, 50.0],
        #                'e': [0.01, 0.05, 0.1, 0.5, 1.0]}
        # grid_searcher = grid_search.SVRGridSearch(search_dict)
        # best_param, best_score = grid_searcher.run(training_set_all)
        print('Running grid search')
        search_dict = {'C': [0.01, 0.1, 1.0, 10.0, 100.0, 1000.0],
                       'epsilon': [0.001, 0.01, 0.1, 1.0],
                       'gamma': [0.0001, 0.001, 0.01, 0.1],
                       'kernel': ['rbf']}
        grid_searcher = GridSearchCV(svm.SVR(), search_dict, n_jobs=6, verbose=2)
        grid_searcher.fit(training_set_all[:, :-1], training_set_all[:, -1])
        # bestC = best_param['c']
        # bestE = best_param['e']
        bestC = grid_searcher.best_params_['C']
        bestE = grid_searcher.best_params_['epsilon']
        bestG = grid_searcher.best_params_['gamma']
        print('All done. Optimal parameter: C={}, e={}, g={}, score={}'
              .format(bestC, bestE, bestG, grid_searcher.best_score_))
        regressor = grid_searcher.best_estimator_
    else:
        print('Train with parameter C={}, e={}, gamma={}'.format(args.C, args.e, args.g))
        regressor = svm.SVR(C=args.C, epsilon=args.e, gamma=args.g)
        regressor.fit(training_set_all[:, :-1], training_set_all[:, -1])
        # score = mean_squared_error(regressor.predict(training_set_all[:, :-1]), training_set_all[:, -1])
        score = regressor.score(training_set_all[:, :-1], training_set_all[:, -1]);
        print('Training score:', score)

    # write model to file
    if len(args.output) > 0:
        joblib.dump(regressor, args.output)
        print('Model written to ' + args.output)
