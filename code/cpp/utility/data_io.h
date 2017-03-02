//
// Created by yanhang on 2/6/17.
//

#ifndef PROJECT_IMU_DATASET_H
#define PROJECT_IMU_DATASET_H

#include <vector>
#include <string>

#include <Eigen/Eigen>
#include <glog/logging.h>
#include <opencv2/opencv.hpp>

namespace IMUProject{

    class FileIO{
    public:
        FileIO(const std::string& directory): directory_(directory){}

        const std::string& GetDirectory() const{
            return directory_;
        }

        inline const std::string GetProcessedData() const{
            char buffer[128] = {};
            sprintf(buffer, "%s/processed/data.csv", directory_.c_str());
            return std::string(buffer);
        }
    private:
        const std::string directory_;
    };

    // Structure to specify the column layout
    struct DataLayout{
	    DataLayout(){}
        const int time_stamp = 0;
        const int gyro = 1;
        const int accelerometer = 4;
        const int linear_acceleration = 7;
        const int gravity = 10;
        const int position = 13;
        const int orientation = 16;
        const int rotation_vector = 20;
    };



    class IMUDataset {
    public:
        /// Constructor for IMUDataset
        /// \param directory Root directory of the dataset
        /// \param load_control
        IMUDataset(const std::string& directory, unsigned char load_control = 255);

        // getters
        inline const cv::Mat& GetGyro() const{
            return gyrocope_;
        }
        inline cv::Mat& GetGyro(){
            return gyrocope_;
        }

        inline const cv::Mat& GetLinearAcceleration() const{
            return linear_acceleration_;
        }
        inline cv::Mat& GetLinearAcceleration(){
            return linear_acceleration_;
        }

        inline const cv::Mat& GetAccelerometer() const{
            return accelerometer_;
        }

        inline cv::Mat& GetAccelerometer(){
            return accelerometer_;
        }

        inline const cv::Mat& GetGravity() const{
            return gravity_;
        }

        inline cv::Mat& GetGravity(){
            return gravity_;
        }

        inline cv::Mat& GetRotationVector(){
            return rotation_vector_;
        }

        inline const cv::Mat& GetRotationVector() const{
            return rotation_vector_;
        }

        inline cv::Mat& GetMagnetometer(){
            return magnetometer_;
        }

        inline const cv::Mat& GetMagnetometer() const{
            return magnetometer_;
        }

        inline const cv::Mat& GetOrientation() const{
            return orientation_;
        }

        inline cv::Mat& GetOrientation(){
            return orientation_;
        }

        inline const cv::Mat& GetPosition() const{
            return position_;
        }

        inline cv::Mat& GetPosition(){
            return position_;
        }

        inline const cv::Mat& GetTimeStamp() const{
            return timestamp_;
        }

        static constexpr unsigned char IMU_GYRO = 1;
        static constexpr unsigned char IMU_ACCELEROMETER = 2;
        static constexpr unsigned char IMU_LINEAR_ACCELERATION = 4;
        static constexpr unsigned char IMU_GRAVITY = 8;
        static constexpr unsigned char IMU_ROTATION_VECTOR = 16;
        static constexpr unsigned char IMU_POSITION = 32;
        static constexpr unsigned char IMU_ORIENTATION = 64;

        static constexpr double kNanoToSec = 1e09;

    private:
        const FileIO file_io_;
        const DataLayout layout_;

        //data from IMU
        cv::Mat timestamp_;
        cv::Mat gyrocope_;
        cv::Mat accelerometer_;
        cv::Mat linear_acceleration_;
        cv::Mat gravity_;
        cv::Mat magnetometer_;
        cv::Mat rotation_vector_;

        // data from tango
        cv::Mat orientation_;
        cv::Mat position_;
    };

	/// Write a trajecotry to PLY file
	/// \param path output path
	/// \param position Nx3 cv::Mat contains the position
	/// \param orientation Nx4 cv::Mat contains the orientation as quaternion
	/// \param axis_length the length of the axis, set to negative value to omit axis
	/// \param kpoints
	/// \param interval the interval of axis visualization, set to negative value to omit axis
    void WriteToPly(const std::string& path, const cv::Mat& position, const cv::Mat& orientation,
                    const double axis_length = 0.5, const int kpoints = 100, const int interval=100);

} //namespace IMUProject
#endif //PROJECT_IMU_DATASET_H
