//
// Created by yanhang on 2/6/17.
//

#include "imu_dataset.h"

namespace IMUProject{
    IMUDataset::IMUDataset(const std::string &directory, unsigned char load_control): file_io_(directory){
        cv::Ptr<cv::ml::TrainData> data_all = cv::ml::TrainData::loadFromCSV(file_io_.GetProcessedData(), 1);
        CHECK(data_all.get()) << "Can not open " << file_io_.GetProcessedData();
        cv::Mat data_mat = data_all->getSamples();

        timestamp_ = data_mat.colRange(layout_.time_stamp, layout_.time_stamp + 1).clone();
        orientation_ = data_mat.colRange(layout_.orientation, layout_.orientation + 4).clone();
        position_ = data_mat.colRange(layout_.position, layout_.position + 3).clone();

        if(load_control & IMU_GYRO_BIT != 0){
            gyrocope_ = data_mat.colRange(layout_.gyro, layout_.gyro + 3).clone();
        }

        if(load_control & IMU_ACCELEROMETER_BIT != 0){
            accelerometer_ = data_mat.colRange(layout_.accelerometer, layout_.accelerometer + 3).clone();
        }

        if(load_control & IMU_LINEAR_ACCELERATION != 0){
            linear_acceleration_ = data_mat.colRange(layout_.linear_acceleration, layout_.linear_acceleration + 3).clone();
        }

        if(load_control & IMU_GRAVITY != 0){
            linear_acceleration_ = data_mat.colRange(layout_.gravity, layout_.gravity + 3).clone();
        }
    }
}//namespace IMUProject