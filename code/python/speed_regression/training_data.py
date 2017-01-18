import numpy as np
import pandas


class TrainingDataOption:
    def __init__(self, sample_step=50, window_size=300, frq_threshold=5):
        self.sample_step_ = sample_step
        self.window_size_ = window_size
        self.frq_threshold_ = frq_threshold
        self.nanoToSec = 1000000000.0


class SpeedRegressionTrainData:
    # static variables

    def __init__(self, option):
        self.option_ = option

    @staticmethod
    def compute_speed(pose_data, time, ind):
        return np.linalg.norm(pose_data[ind+1] - pose_data[ind-1]) / (time[ind+1] - time[ind-1])

    def CreateTrainingData(self, data_all, imu_columns):
        N = data_all.shape[0]
        sample_points = np.arange(self.option_.window_size_,
                                  N - 1,
                                  self.option_.sample_step_,
                                  dtype=int)
        pose_data = data_all[['pos_x', 'pos_y', 'pos_z']]
        data_used = data_all[imu_columns]

        local_imu_list = [list(data_used[ind-self.option_.window_size_:ind].values.flatten())
                          for ind in sample_points]
        speed = [self.compute_speed(pose_data.values, data_all['time'].values, ind) * self.option_.nanoToSec
                 for ind in sample_points]

        return pandas.DataFrame(local_imu_list), pandas.DataFrame(speed)

