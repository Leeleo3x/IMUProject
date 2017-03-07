import time
import argparse
import warnings
import os

from sklearn import svm
from sklearn.metrics import r2_score
from sklearn.externals import joblib
from sklearn.model_selection import GridSearchCV
import numpy as np
import pandas
import cv2

# from speed_regression import training_data as td
# from speed_regression import grid_search

import training_data as td

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
    parser.add_argument('--train_cv', action='store_true')
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
                      'target_smooth_sigma': 30.0,
                      'feature_smooth_alpha': 0.2}

        training_feature, training_target = td.get_training_data(data_all=data_all, imu_columns=imu_columns,
                                                                 option=options, extra_args=extra_args)
        training_feature_all.append(training_feature)
        training_target_all.append(training_target)

    assert len(training_feature_all) > 0, 'No data was loaded'
    training_feature_all = np.concatenate(training_feature_all, axis=0)
    training_target_all = np.concatenate(training_target_all, axis=0)

    print("{} samples in total".format(training_target_all.shape[0]))

    # Training separate regressor for each target channel
    if training_target_all.ndim == 1:
        training_target_all = training_target_all[:, None]

    for chn in range(training_target_all.shape[1]):
        print('Training SVM for target ', chn)
        bestC = 0
        bestE = 0

        regressor = None
        regressor_cv = None

        if args.C is None or args.e is None:
            print('Running grid search')

            search_dict = {'C': [0.1, 1.0, 10.0, 100.0],
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
            if args.train_cv:
                print('Training with OpenCV, C: {}, epsilon: {}'.format(args.C, args.e))
                regressor_cv = cv2.ml.SVM_create()
                regressor_cv.setType(cv2.ml.SVM_EPS_SVR)
                regressor_cv.setKernel(cv2.ml.SVM_RBF)
                regressor_cv.setC(args.C)
                regressor_cv.setP(args.e)
                regressor_cv.setGamma(1.0 / training_feature_all.shape[1])
                regressor_cv.setTermCriteria((cv2.TERM_CRITERIA_MAX_ITER + cv2.TERM_CRITERIA_EPS, 10000, 1e-09))
                training_feature_cv = training_feature_all.astype(np.float32)
                training_target_cv = training_target_all[:, chn].astype(np.float32)

                regressor_cv.train(training_feature_cv, cv2.ml.ROW_SAMPLE, training_target_cv)

                predicted_training = regressor_cv.predict(training_feature_cv)[1]
                score_cv = r2_score(training_target_all[:, chn], predicted_training)
                print('Training score by OpenCV: {}, support vectors: {}'.format(score_cv, regressor_cv.getSupportVectors().shape[0]))
            else:
                print('Train with parameter C={}, e={}'.format(args.C, args.e))
                regressor = svm.SVR(C=args.C, epsilon=args.e)
                regressor.fit(training_feature_all, training_target_all[:, chn])
                # score = mean_squared_error(regressor.predict(training_set_all[:, :-1]), training_set_all[:, -1])
                score = regressor.score(training_feature_all, training_target_all[:, chn])
                print('Training score:', score)

        # write model to file
        if len(args.output) > 0:
            # out_path = '{}_{}'.format(args.output, chn)
            # joblib.dump(regressor, out_path)
            # print('Model written to ' + out_path)

            if args.train_cv:
                cv_model_path = '{}_w{}_s{}_{}.yml'.format(args.output, args.window, args.step, chn)
                print('CV model written into ', cv_model_path)
                regressor_cv.save(cv_model_path)
