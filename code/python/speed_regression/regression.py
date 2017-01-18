import argparse
import os

from sklearn import svm
import numpy as np
import matplotlib.pyplot as plt
import pandas

import training_data as td

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('dir')
    parser.add_argument('--window', default=300, type=int)
    parser.add_argument('--step', default=50, type=int)

    args = parser.parse_args()

    data_dir = args.dir + '/processed'
    print('Loading dataset ' + data_dir + '/data.csv')
    data_all = pandas.read_csv(data_dir + '/data.csv')

    print('Creating training set')
    options = td.TrainingDataOption(sample_step=args.step, window_size=args.window)
    data_factory = td.SpeedRegressionTrainData(option=options)

    imu_columns = ['gyro_w', 'gyro_x', 'gyro_y', 'gyro_z', 'acce_x', 'acce_y', 'acce_z']
    features, targets = data_factory.CreateTrainingData(data_all=data_all, imu_columns=imu_columns)

    print(features.shape)
    plt.figure()
    plt.plot(targets)
    plt.show()
