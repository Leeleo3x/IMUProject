//
// Created by yanhang on 2/6/17.
//

#include "data_io.h"
#include <OpenMesh/Core/IO/MeshIO.hh>
#include <OpenMesh/Core/Mesh/TriMesh_ArrayKernelT.hh>

namespace IMUProject{
    IMUDataset::IMUDataset(const std::string &directory, unsigned char load_control): file_io_(directory) {
        cv::Ptr<cv::ml::TrainData> data_all = cv::ml::TrainData::loadFromCSV(file_io_.GetProcessedData(), 1, 0);
        CHECK(data_all.get()) << "Can not open " << file_io_.GetProcessedData();
        cv::Mat data_mat = data_all->getSamples();
        timestamp_ = data_mat.colRange(layout_.time_stamp, layout_.time_stamp + 1).clone();

        if (load_control & IMU_ORIENTATION) {
            orientation_ = data_mat.colRange(layout_.orientation, layout_.orientation + 4).clone();
            LOG(INFO) << "Orientation loaded";
        }

        if (load_control & IMU_POSITION) {
            position_ = data_mat.colRange(layout_.position, layout_.position + 3).clone();
            LOG(INFO) << "Position loaded";
        }

        if(load_control & IMU_GYRO){
            gyrocope_ = data_mat.colRange(layout_.gyro, layout_.gyro + 3).clone();
            LOG(INFO) << "Gyroscope loaded";
        }

        if(load_control & IMU_ACCELEROMETER){
            accelerometer_ = data_mat.colRange(layout_.accelerometer, layout_.accelerometer + 3).clone();
            LOG(INFO) << "Accelerometer loaded";
        }

        if(load_control & IMU_LINEAR_ACCELERATION){
            linear_acceleration_ = data_mat.colRange(layout_.linear_acceleration, layout_.linear_acceleration + 3).clone();
            LOG(INFO) << "Linear acceleration loaded";
        }

        if(load_control & IMU_GRAVITY){
            gravity_ = data_mat.colRange(layout_.gravity, layout_.gravity + 3).clone();
            LOG(INFO) << "Gravity loaded";
        }

        if (load_control & IMU_MAGNETOMETER){
            magnetometer_ = data_mat.colRange(layout_.magetometer, layout_.magetometer + 3).clone();
            LOG(INFO) << "Magnetometer loaded";
        }

        if(load_control & IMU_ROTATION_VECTOR){
            rotation_vector_ = data_mat.colRange(layout_.rotation_vector, layout_.rotation_vector + 4).clone();
            LOG(INFO) << "Rotation vector loaded";
        }
    }

    void WriteToPly(const std::string& path, const cv::Mat position, const cv::Mat orientation,
                    const double axis_length, const int kpoints){
        using TriMesh = OpenMesh::TriMesh_ArrayKernelT<>;
        CHECK_EQ(position.rows, orientation.rows);

        std::vector<TriMesh::VertexHandle> vhandle((size_t) position.rows);

    }
}//namespace IMUProject