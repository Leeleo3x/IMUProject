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

        inline const std::string GetPlainTextData() const{
            char buffer[128] = {};
            sprintf(buffer, "%s/processed/data_plain.txt", directory_.c_str());
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
        inline const std::vector<Eigen::Vector3d>& GetGyro() const{
            return gyrocope_;
        }
        inline std::vector<Eigen::Vector3d>& GetGyro(){
            return gyrocope_;
        }

        inline const std::vector<Eigen::Vector3d>& GetLinearAcceleration() const{
            return linear_acceleration_;
        }
        inline std::vector<Eigen::Vector3d>& GetLinearAcceleration(){
            return linear_acceleration_;
        }

        inline const std::vector<Eigen::Vector3d>& GetAccelerometer() const{
            return accelerometer_;
        }

        inline std::vector<Eigen::Vector3d>& GetAccelerometer(){
            return accelerometer_;
        }

        inline const std::vector<Eigen::Vector3d>& GetGravity() const{
            return gravity_;
        }

        inline std::vector<Eigen::Vector3d>& GetGravity(){
            return gravity_;
        }

        inline std::vector<Eigen::Quaterniond>& GetRotationVector(){
            return rotation_vector_;
        }

        inline const std::vector<Eigen::Quaterniond>& GetRotationVector() const{
            return rotation_vector_;
        }

        inline const std::vector<Eigen::Quaterniond>& GetOrientation() const{
            return orientation_;
        }

        inline std::vector<Eigen::Quaterniond>& GetOrientation(){
            return orientation_;
        }

        inline const std::vector<Eigen::Vector3d>& GetPosition() const{
            return position_;
        }

        inline std::vector<Eigen::Vector3d>& GetPosition(){
            return position_;
        }

        inline const std::vector<double>& GetTimeStamp() const{
            return timestamp_;
        }

        static constexpr unsigned char IMU_GYRO = 1;
        static constexpr unsigned char IMU_ACCELEROMETER = 2;
        static constexpr unsigned char IMU_LINEAR_ACCELERATION = 4;
        static constexpr unsigned char IMU_GRAVITY = 8;
        static constexpr unsigned char IMU_ROTATION_VECTOR = 16;
        static constexpr unsigned char IMU_POSITION = 32;
        static constexpr unsigned char IMU_ORIENTATION = 64;

        static constexpr double kNanoToSec = 1000000000.0;

    private:
        const FileIO file_io_;
        const DataLayout layout_;

        //data from IMU
        std::vector<double> timestamp_;
        std::vector<Eigen::Vector3d> gyrocope_;
        std::vector<Eigen::Vector3d> accelerometer_;
        std::vector<Eigen::Vector3d> linear_acceleration_;
        std::vector<Eigen::Vector3d> gravity_;
        std::vector<Eigen::Quaterniond> rotation_vector_;

        // data from tango
        std::vector<Eigen::Quaterniond> orientation_;
        std::vector<Eigen::Vector3d> position_;
    };

	/// Write a trajecotry to PLY file
	/// \param path output path
	/// \param position Nx3 cv::Mat contains the position
	/// \param orientation Nx4 cv::Mat contains the orientation as quaternion
	/// \param axis_length the length of the axis, set to negative value to omit axis
	/// \param kpoints
	/// \param interval the interval of axis visualization, set to negative value to omit axis
    void WriteToPly(const std::string& path, const double* ts, const Eigen::Vector3d* position,
                    const Eigen::Quaterniond* orientation, const int N, const bool only_xy = false,
                    const Eigen::Vector3d traj_color=Eigen::Vector3d(0, 255, 255),
                    const double axis_length = 0.5, const int kpoints = 100, const int interval=200);

} //namespace IMUProject
#endif //PROJECT_IMU_DATASET_H
