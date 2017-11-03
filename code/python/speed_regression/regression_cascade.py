import os
import sys
import warnings
import cv2
import numpy as np
import pandas
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from sklearn.metrics import accuracy_score
from sklearn import svm
from sklearn.model_selection import GridSearchCV

sys.path.append('/home/yanhang/Documents/research/IMUProject/code/python')
sys.path.append('/Users/yanhang/Documents/research/IMUProject/code/python')
import speed_regression.training_data as td


class SVMOption:
    def __init__(self, svm_type=cv2.ml.SVM_C_SVC, kernel_type=cv2.ml.SVM_RBF, degree=1,
                 gamma=1, C=5, e=0.01, max_iter=10000):
        self.svm_type = svm_type
        self.kernel_type = kernel_type
        self.degree = degree
        self.gamma = gamma
        self.C = C
        self.e = e
        self.max_iter = max_iter
        self.kParams__ = 7

    def to_string(self):
        type_str = 'SVM' if self.svm_type==cv2.ml.SVM_C_SVC else 'SVR'
        kernel_type_str = 'RBF'
        if self.kernel_type == cv2.ml.SVM_LINEAR:
            kernel_type_str = 'Linear'
        elif self.kernel_type == cv2.ml.SVM_POLY:
            kernel_type_str = 'Poly'
        return '%s %s %d %f %f %f %d' % (type_str, kernel_type_str, self.degree,
                                         self.gamma, self.C, self.e, self.max_iter)

    def from_string(self, input_string):
        buffer = input_string.strip().split()
        assert len(buffer) == self.kParams__
        if buffer[0] == 'SVM':
            self.svm_type = cv2.ml.SVM_C_SVC
        elif buffer[0] == 'SVR':
            self.svm_type = cv2.ml.SVM_EPS_SVR
        else:
            raise ValueError('Invalid SVM type: ', buffer[0])
        if buffer[1] == 'RBF':
            self.kernel_type = cv2.ml.SVM_RBF
        elif buffer[1] == 'Linear':
            self.kernel_type = cv2.ml.SVM_LINEAR
        elif buffer[1] == 'Poly':
            self.kernel_type = cv2.ml.SVM_POLY
        else:
            raise ValueError('Invalid kernel type: ', buffer[1])
        self.degree = int(buffer[2])
        self.gamma = float(buffer[3])
        self.C = float(buffer[4])
        self.e = float(buffer[5])
        self.max_iter = int(buffer[6])


def create_svm(svm_options):
    svm = cv2.ml.SVM_create()
    svm.setType(svm_options.svm_type)
    svm.setKernel(svm_options.kernel_type)
    svm.setDegree(svm_options.degree)
    svm.setGamma(svm_options.gamma)
    svm.setC(svm_options.C)
    svm.setP(svm_options.e)
    svm.setTermCriteria((cv2.TERM_CRITERIA_COUNT + cv2.TERM_CRITERIA_EPS, svm_options.max_iter, 1e-09))
    return svm


class SVRCascadeOption:
    """
    This structure represents the options for SVR cascading model.
    """
    def __init__(self, num_classes=1, num_channels=1, svm_option=None, svr_options=None):
        self.num_classes = num_classes
        self.num_channels = num_channels
        if svm_option is None:
            svm_option = SVMOption()
        self.svm_option = svm_option
        if svr_options is None:
            svr_options = [SVMOption() for _ in range(self.num_channels * self.num_classes)]
        self.svr_options = svr_options
        self.version_tag = 'v1.0'

    def write_to_file(self, path):
        with open(path, 'w') as f:
            f.write(self.version_tag + '\n')
            f.write('%d %d\n' % (self.num_classes, self.num_channels))
            f.write(self.svm_option.to_string() + '\n')
            for svr_opt in self.svr_options:
                f.write(svr_opt.to_string() + '\n')

    def load_from_file(self, path):
        with open(path, 'r') as f:
            version = f.readline().strip()
            if version != self.version_tag:
                raise ValueError('The version of the file does match the current version: {} vs {}'
                                 .format(self.version_tag, version))

            header = f.readline().strip().split()
            self.num_classes = int(header[0])
            self.num_channels = int(header[1])
            svm_line = f.readline()
            self.svm_option.from_string(svm_line)
            self.svr_options = [SVMOption() for _ in range(self.num_classes * self.num_channels)]
            for cls in range(self.num_classes):
                for chn in range(self.num_channels):
                    svr_line = f.readline()
                    self.svr_options[cls * self.num_channels + chn].from_string(svr_line)


class SVRCascade:
    def __init__(self, options, class_map):
        self.num_classes = options.num_classes
        self.num_channels = options.num_channels
        assert len(options.svr_options) == self.num_classes * self.num_channels
        self.classifier = create_svm(options.svm_option)
        self.regressors = [create_svm(opt) for opt in options.svr_options]
        self.options = options
        self.class_map = class_map

    def train(self, train_feature, train_label, train_response):
        assert train_response.shape[1] == self.num_channels
        print('Training classifier')
        train_feature_cv = train_feature.astype(np.float32)
        self.classifier.train(train_feature_cv, cv2.ml.ROW_SAMPLE, train_label)
        predicted_train = self.classifier.predict(train_feature_cv)[1].ravel()
        error_svm = accuracy_score(train_label, predicted_train)
        print('Classifier trained. Training accuracy: %f' % error_svm)
        for cls_name, cls in self.class_map.items():
            feature_in_class = train_feature_cv[train_label == cls, :]
            target_in_class = train_response[train_label == cls, :]
            if cls_name == 'transition':
                continue
            for chn in range(self.num_channels):
                rid = cls * self.num_channels + chn
                print('Training regressor for class %d, channel %d' % (cls, chn))
                self.regressors[rid].train(feature_in_class, cv2.ml.ROW_SAMPLE,
                                           target_in_class[:, chn].astype(np.float32))
                predicted = self.regressors[rid].predict(feature_in_class)[1]
                print('Regressor for class %d  channel %d trained. Training error: %f(r2), %f(MSE)' %
                      (cls, chn, r2_score(predicted, target_in_class[:, chn]),
                       mean_squared_error(predicted, target_in_class[:, chn])))
        print('All done')

    def test(self, test_feature, true_label=None, true_responses=None):
        feature_cv = test_feature.astype(np.float32)
        labels = self.classifier.predict(feature_cv)[1].ravel()
        if true_label is not None:
            print('Classification accuracy: ', accuracy_score(true_label, labels))

        index_array = np.array([i for i in range(test_feature.shape[0])])
        reverse_index = []
        predicted_class = []
        for cls_name, cls in self.class_map.items():
            feature_in_class = feature_cv[labels == cls, :]
            predicted_in_class = np.zeros([feature_in_class.shape[0], self.num_channels])
            if feature_in_class.shape[0] == 0 or cls_name == 'transition':
                predicted_class.append(predicted_in_class)
                reverse_index.append(index_array[labels == cls])
                continue
            for chn in range(self.num_channels):
                rid = cls * self.num_channels + chn
                predicted_in_class[:, chn] = self.regressors[rid].predict(feature_in_class)[1].ravel()
            predicted_class.append(predicted_in_class)
            reverse_index.append(index_array[labels == cls])
        if true_responses is not None:
            for cls_name, cls in self.class_map:
                # We store the error in both R2 score and MSE score
                true_in_class = true_responses[labels == cls, :]
                if true_in_class.shape[0] == 0 or cls_name == 'transition':
                    continue
                for chn in range(self.num_channels):
                    r2 = r2_score(true_in_class[:, chn], predicted_class[cls][:, chn])
                    mse = mean_squared_error(true_in_class[:, chn], predicted_class[cls][:, chn])
                    print('Error for class %d, channel %d: %f(R2), %f(MSE)' % (cls, chn, r2, mse))
        predicted_all = np.empty([test_feature.shape[0], self.num_channels])
        for cls in range(self.num_classes):
            predicted_all[reverse_index[cls], :] = predicted_class[cls]

        if true_responses is not None:
            for chn in range(self.num_channels):
                r2 = r2_score(true_responses[:, chn], predicted_all[:, chn])
                mse = mean_squared_error(true_responses[:, chn], predicted_all[:, chn])
                print('Overall regression error for channel %d: %f(R2), %f(MSE)' % (chn, r2, mse))
        return labels, predicted_all


def write_model_to_file(path, model, suffix=''):
    if not os.path.exists(path):
        print('Folder {} not exist. Creating'.format(path))
        os.makedirs(path)
    model.options.write_to_file(path + '/option.txt')
    with open(path + '/class_map.txt', 'w') as f:
        f.write('%d\n' % len(model.class_map))
        for k, v in class_map.items():
            f.write('{:s} {:d}\n'.format(k, v))
    model.classifier.save(path + '/classifier{}.yaml'.format(suffix))
    for cls in range(model.num_classes):
        for chn in range(model.num_channels):
            model.regressors[cls * model.num_channels + chn].save(path +
                                                                  '/regressor{}_{}_{}.yaml'.format(suffix, cls, chn))


def load_model_from_file(path, suffix=''):
    options = SVRCascadeOption()
    options.load_from_file(path + '/option.txt')
    class_map = {}
    with open(path + '/class_map.txt') as f:
        num_classes = int(f.readline().strip())
        assert num_classes == options.num_classes
        for i in range(num_classes):
            line = f.readline().strip().split()
            class_map[line[0]] = int(line[1])
    model = SVRCascade(options, class_map)
    model.classifier = cv2.ml.SVM_load(path + '/classifier{}.yaml'.format(suffix))
    for cls in range(options.num_classes):
        for chn in range(options.num_channels):
            rid = cls * options.num_channels + chn
            model.regressors[rid] = cv2.ml.SVM_load(path + '/regressor{}_{}_{}.yaml'.format(suffix, cls, chn))
    return model


def get_best_option(train_feature, train_label, class_map, train_response, svm_search_dict=None, svr_search_dict=None,
                    n_split=3, n_jobs=6, verbose=3):
    if svm_search_dict is None:
        svm_search_dict = {'C': [1.0, 10.0]}

    # First find best parameters for the classifier
    svm_grid_searcher = GridSearchCV(svm.SVC(), svm_search_dict, cv=n_split, n_jobs=n_jobs, verbose=verbose)
    svm_grid_searcher.fit(train_feature, train_label)
    svm_best_param = svm_grid_searcher.best_params_
    print('SVM fitted. Optimal parameters: ', svm_best_param)
    svm_option = SVMOption()
    svm_option.svm_type = cv2.ml.SVM_C_SVC
    svm_option.kernel_type = cv2.ml.SVM_RBF
    svm_option.C = svm_best_param['C']
    svm_option.gamma = 1. / train_feature.shape[1]
    if svr_search_dict is None:
        svr_search_dict = {'C': [1.0, 10.0],
                           'epsilon': [0.001, 0.01]}
    svr_options = []
    num_classes = max(train_label) + 1
    assert num_classes == len(class_map)
    num_channels = train_response.shape[1]
    for cls_name, cls in class_map.items():
        for chn in range(num_channels):
            svr_option = SVMOption()
            svr_option.svm_type = cv2.ml.SVM_EPS_SVR
            svr_option.kernel_type = cv2.ml.SVM_RBF
            svr_option.gamma = 1. / train_feature.shape[1]
            if cls_name is not 'transition':
                svr_grid_searcher = GridSearchCV(svm.SVR(), svr_search_dict, cv=n_split,
                                                 scoring='neg_mean_squared_error', n_jobs=n_jobs, verbose=verbose)
                svr_grid_searcher.fit(train_feature[train_label == cls, :], train_response[train_label == cls, chn])
                best_svr_param = svr_grid_searcher.best_params_
                svr_option.C = best_svr_param['C']
                svr_option.e = best_svr_param['epsilon']
            svr_options.append(svr_option)
    print('All done')
    return SVRCascadeOption(num_classes, num_channels, svm_option, svr_options)


def get_best_option_analytical(train_feature, train_label, train_response):
    num_classes = max(train_label) + 1
    num_channels = train_response.shape[1]
    svr_options = []
    for cls in range(num_classes):
        train_in_class = train_feature[train_label == cls, :]
        for chn in range(num_channels):
            response_in_class = train_response[train_label == cls, chn]
            mean_response = np.mean(response_in_class)
            dev_response = np.std(response_in_class)
            svr_option = SVMOption()
            svr_option.svm_type = cv2.ml.SVM_EPS_SVR
            svr_option.kernel_type = cv2.ml.SVM_RBF
            svr_option.C = max(abs(mean_response + 3 * dev_response), abs(mean_response - 3 * dev_response))
            pass


def load_datalist(path, option, class_map=None):
    root_dir = os.path.dirname(path)
    with open(path) as f:
        dataset_list = [s.strip('\n') for s in f.readlines()]
    feature_all = []
    label_all = []
    responses_all = []
    build_classmap = False
    if class_map is None:
        class_map = {}
        build_classmap = True
    imu_columns = ['gyro_x', 'gyro_y', 'gyro_z', 'linacce_x', 'linacce_y', 'linacce_z']
    for dataset in dataset_list:
        if len(dataset) > 0 and dataset[0] == '#':
            continue
        info = dataset.split(',')
        if len(info) != 2:
            warnings.warn('Line ' + dataset + ' has the wrong format. Skipped.')
            continue
        data_path = root_dir + '/' + info[0] + '/processed/data.csv'
        if not os.path.exists(data_path):
            warnings.warn('File ' + data_path + ' does not exist. Skipped.')
            continue
        print('Loading dataset ' + data_path + ', type: ' + info[1])
        if info[1] not in class_map:
            if build_classmap:
                class_map[info[1]] = len(class_map)
            else:
                warnings.warn('Class %s not found in the class map. Skipped' % info[1])
                continue

        data_all = pandas.read_csv(data_path)
        extra_args = {'target_smooth_sigma': 30.0,
                      'feature_smooth_sigma': 2.0}
        feature, target = td.get_training_data(data_all=data_all, imu_columns=imu_columns,
                                               option=option, extra_args=extra_args)
        feature_all.append(feature)
        responses_all.append(target)
        label = class_map[info[1]]
        label_all.append(np.array([label for _ in range(feature.shape[0])]))
    feature_all = np.concatenate(feature_all, axis=0)
    label_all = np.concatenate(label_all, axis=0)
    responses_all = np.concatenate(responses_all, axis=0)
    return feature_all, label_all, responses_all, class_map


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('list')
    parser.add_argument('--output_path', default=None, type=str)
    parser.add_argument('--subsample', default=1, type=int)
    parser.add_argument('--step_size', default=10, type=int)
    parser.add_argument('--cv', default=3, type=int)
    parser.add_argument('--option', default=None, type=str)
    args = parser.parse_args()

    option = td.TrainingDataOption()
    option.sample_step_ = args.step_size
    feature_all, label_all, responses_all, class_map = load_datalist(path=args.list, option=option)
    responses_all = responses_all[:, [0, 2]]

    feature_all = feature_all[0:-1:args.subsample]
    label_all = label_all[0:-1:args.subsample]
    responses_all = responses_all[0:-1:args.subsample]

    print('Data loaded. Total number of samples: ', feature_all.shape[0])

    for key, value in class_map.items():
        print('%d samples in %s(label %d)' % (len(label_all[label_all==value]), key, value))

    best_option = SVRCascadeOption()
    if args.option:
        best_option.load_from_file(args.option)
        print('Options loaded from file: ', args.option)
    else:
        print('No option file is provided, running grid search')
        best_option = get_best_option(feature_all, label_all, class_map, responses_all, n_split=args.cv)
        best_option.write_to_file(args.output_path + '/option.txt')
    model = SVRCascade(best_option, class_map)
    model.train(feature_all, label_all, responses_all)

    if args.output_path:
        write_model_to_file(args.output_path, model)
