//
// Created by yanhang on 2/6/17.
//

#ifndef PROJECT_IMU_DATASET_H
#define PROJECT_IMU_DATASET_H

#include <vector>

#include <glog/logging.h>
#include <opencv2/opencv.hpp>

namespace IMUProject{

    class FileIO{
    public:
        FileIO(const std::string& directory): directory_(directory){}

        inline const std::string GetDirectory() const{
            return directory_;
        }

        inline const std::string GetProcessedData() const{
            return directory_ + "/processed/data.csv";
        }

        inline const std::string GetRegressedSpeed() const{
            return directory_ + "/processed/speed.txt";
        }
    private:
        const std::string directory_;
    };

    // Structure to specify the column layout
    struct DataLayout{
        const int time_stamp = 0;
        const int gyro = 1;
        const int accelerometer = 4;
        const int linear_acceleration = 7;
        const int gravity = 10;
        const int position = 13;
        const int orientation = 14;
    };

    constexpr unsigned char IMU_GYRO_BIT = 8;
    constexpr unsigned char IMU_ACCELEROMETER_BIT = 4;
    constexpr unsigned char IMU_LINEAR_ACCELERATION = 2;
    constexpr unsigned char IMU_GRAVITY = 1;

    class IMUDataset {
        /// Constructor for IMUDataset
        /// \param directory Root directory of the dataset
        /// \param load_control
        IMUDataset(const std::string& directory, unsigned char load_control = 255);

        // getters
        inline const cv::Mat GetGyro() const{
            return gyrocope_;
        }
        inline cv::Mat GetGyro(){
            return gyrocope_;
        }

        inline const cv::Mat GetLinearAcceleration() const{
            return linear_acceleration_;
        }
        inline cv::Mat GetLinearAcceleration(){
            return linear_acceleration_;
        }

        inline const cv::Mat GetAccelerometer() const{
            return accelerometer_;
        }

        inline cv::Mat GetAccelerometer(){
            return accelerometer_;
        }

        inline const cv::Mat GetGravity() const{
            return gravity_;
        }

        inline cv::Mat GetGravity(){
            return gravity_;
        }

        inline const cv::Mat GetOrientation() const{
            return orientation_;
        }

        inline cv::Mat GetOrientation(){
            return orientation_;
        }

        inline const cv::Mat GetPosition() const{
            return position_;
        }

        inline cv::Mat GetPosition(){
            return position_;
        }

        inline const cv::Mat GetTimeStamp() const{
            return timestamp_;
        }
    private:
        const FileIO file_io_;
        const DataLayout layout_;
        cv::Mat timestamp_;
        cv::Mat gyrocope_;
        cv::Mat accelerometer_;
        cv::Mat linear_acceleration_;
        cv::Mat gravity_;
        cv::Mat orientation_;
        cv::Mat position_;
    };

}
#endif //PROJECT_IMU_DATASET_H
