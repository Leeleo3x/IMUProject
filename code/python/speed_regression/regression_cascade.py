import os
import warnings
import cv2
import numpy as np
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from sklearn.metrics import accuracy_score

import training_data as td


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


class SVRCascade:
    def __init__(self, num_classes, svm_option, svr_options, num_channels):
        self.num_classes = num_classes
        self.num_channels = num_channels
        assert len(svr_options) == self.num_classes * self.num_channels
        self.classifier = create_svm(svm_option)
        self.regressors = [create_svm(option) for option in svr_options]

    def train(self, train_feature, train_label, train_response):
        assert train_response.shape[1] == self.num_channels
        print('Training classifier')
        train_feature_cv = train_feature.astype(np.float32)
        self.classifier.train(train_feature_cv, cv2.ml.ROW_SAMPLE, train_label)
        predicted_train = self.classifier.predict(train_feature_cv)[1]
        error_svm = accuracy_score(train_label, predicted_train)
        print('Classifier trained. Training accuracy: %d' % error_svm)
        for cls in range(self.num_classes):
            feature_in_class = train_feature_cv[train_label == cls, :]
            for chn in range(self.num_channels):
                rid = cls * self.num_channels + chn
                print('Training regressor for class %d, channel %d' % (cls, chn))
                target_in_class = train_response[train_label == cls, chn].astype(np.float32)
                self.regressors[rid].train(feature_in_class, cv2.ml.ROW_SAMPLE, target_in_class)
                predicted = self.regressors[rid].predict(feature_in_class)[1]
                print('Regressor for class %d  channel %d trained. Training error: %f(r2), %f(MSE)' %
                      (cls, chn, r2_score(predicted, target_in_class),
                       mean_squared_error(predicted, target_in_class)))
        print('All done')

    def test(self, test_feature, true_label=None, true_responses=None):
        feature_cv = test_feature.astype(np.float32)
        labels = self.classifier.predict(feature_cv)[1]
        if true_label is not None:
            print('Classification accuracy: %d', accuracy_score(true_label, labels))

        index_array = np.array([i for i in range(test_feature[0])])
        reverse_index = []
        predicted_class = []
        for cls in range(self.num_classes):
            feature_in_class = feature_cv[labels == cls, 1:]
            predicted_in_class = np.empty([feature_in_class.shape[0], self.num_channels])
            for chn in range(self.num_channels):
                rid = cls * self.num_channels + chn
                predicted_in_class[:, chn] = self.regressors[rid].predict(feature_in_class)[1]
            predicted_class.append(predicted_in_class)
            reverse_index.append(index_array[labels == cls])

        if true_responses is not None:
            for cls in range(self.num_classes):
                # We store the error in both R2 score and MSE score
                true_in_class = true_responses[labels == cls, :]
                for chn in range(self.num_channels):
                    r2 = r2_score(true_in_class[:, chn], predicted_class[cls, chn, :])
                    mse = mean_squared_error(true_in_class[:, chn], predicted_class[cls, chn, :])
                    print('Error for class %d, channel %d: %f(R2), %f(MSE)' % (cls, chn, r2, mse))
        predicted_all = np.empty([test_feature.shape[0], self.num_channels])
        for cls in range(self.num_classes):
            predicted_all[reverse_index, :] = predicted_class[cls, :]

        if true_responses is not None:
            for chn in range(self.num_channels):
                r2 = r2_score(true_responses[:, chn], predicted_all[:, chn])
                mse = mean_squared_error(true_responses[:, chn], predicted_all[:, chn])
                print('Overall regression error for channel %d: %f(R2), %f(MSE)',
                      chn, r2, mse)


def load_datalist(path, option):
    root_dir = os.path.dirname(path)
    with open(path) as f:
        dataset_list = [s.strip('\n') for s in f.readlines()]
    feature_all = []
    label_all = []
    responses_all = []
    class_map = {}
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
            class_map[info[1]] = len(class_map)
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
    import pandas

    parser = argparse.ArgumentParser()
    parser.add_argument('list')
    args = parser.parse_args()

    option = td.TrainingDataOption()
    feature_all, label_all, responses_all, class_map = load_datalist(path=args.list, option=option)
    responses_all = responses_all[:, [0, 2]]
    print('Data loaded. Total number of samples: ', feature_all.shape[0])

    for key, value in class_map.items():
        print('%d samples in %s' % (len(label_all[label_all==value]), key))

    gamma = 1.0 / feature_all.shape[1]
    num_classes = len(class_map)
    num_channels = responses_all.shape[1]
    svm_option = SVMOption()
    svm_option.gamma = gamma
    svm_option.svm_type = cv2.ml.SVM_C_SVC
    svr_option = [SVMOption() for _ in range(len(class_map) * responses_all.shape[1])]
    for opt in svr_option:
        opt.gamma = gamma
        opt.svm_type = cv2.ml.SVM_EPS_SVR
    model = SVRCascade(num_classes, svm_option, svr_option, num_channels)
    model.train(feature_all, label_all, responses_all)
