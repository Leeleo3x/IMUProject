//
// Created by yanhang on 2/6/17.
//

#include "data_io.h"
#include <OpenMesh/Core/IO/MeshIO.hh>
#include <OpenMesh/Core/Mesh/TriMesh_ArrayKernelT.hh>

#include <fstream>

namespace IMUProject{

    IMUDataset::IMUDataset(const std::string &directory, unsigned char load_control): file_io_(directory) {
        cv::Ptr<cv::ml::TrainData> data_all = cv::ml::TrainData::loadFromCSV(file_io_.GetProcessedData(), 1, 0);
        CHECK(data_all.get()) << "Can not open " << file_io_.GetProcessedData();
        cv::Mat data_mat = data_all->getSamples();
	    const int kSamples = data_mat.rows;
	    timestamp_.resize(kSamples);
	    for(int i=0; i<kSamples; ++i){
		    timestamp_[i] = (double)data_mat.at<float>(i, 0);
	    }

        for(int i=0; i<kSamples; ++i){
            timestamp_[i] /= kNanoToSec;
        }
        if (load_control & IMU_ORIENTATION) {
	        cv::Mat mat = data_mat.colRange(layout_.orientation, layout_.orientation + 4);
	        orientation_.resize(kSamples);
	        for(int i=0; i<kSamples; ++i){
		        const float* v = (float *) mat.ptr(i);
		        //printf("%f, %f, %f, %f\n", v[0], v[1], v[2], v[3]);
		        orientation_[i].w() = v[0];
		        orientation_[i].x() = v[1];
		        orientation_[i].y() = v[2];
		        orientation_[i].z() = v[3];
	        }
            LOG(INFO) << "Orientation loaded";
        }

        if (load_control & IMU_POSITION) {
	        cv::Mat mat = data_mat.colRange(layout_.position, layout_.position + 3);
	        position_.resize(kSamples);
	        for(int i=0; i<kSamples; ++i){
		        for(int j=0; j<3; ++j) {
			        position_[i][j] = static_cast<double>(mat.at<float>(i, j));
		        }
	        }
            LOG(INFO) << "Position loaded";
        }

        if(load_control & IMU_GYRO){
	        cv::Mat mat = data_mat.colRange(layout_.gyro, layout_.gyro + 3);
	        gyrocope_.resize(kSamples);
	        for(int i=0; i<kSamples; ++i){
		        for(int j=0; j<3; ++j) {
			        gyrocope_[i][j] = static_cast<double>(mat.at<float>(i, j));
		        }
	        }
            LOG(INFO) << "Gyroscope loaded";
        }

        if(load_control & IMU_ACCELEROMETER){
	        cv::Mat mat = data_mat.colRange(layout_.accelerometer, layout_.accelerometer + 3);
	        accelerometer_.resize(kSamples);
	        for(int i=0; i<kSamples; ++i){
		        for(int j=0; j<3; ++j) {
			        accelerometer_[i][j] = static_cast<double>(mat.at<float>(i, j));
		        }
	        }
            LOG(INFO) << "Accelerometer loaded";
        }

        if(load_control & IMU_LINEAR_ACCELERATION){
	        cv::Mat mat = data_mat.colRange(layout_.linear_acceleration, layout_.linear_acceleration + 3);
	        linear_acceleration_.resize(kSamples);
	        for(int i=0; i<kSamples; ++i){
		        for(int j=0; j<3; ++j) {
			        linear_acceleration_[i][j] = static_cast<double>(mat.at<float>(i, j));
		        }
	        }
            LOG(INFO) << "Linear acceleration loaded";
        }

        if(load_control & IMU_GRAVITY){
	        cv::Mat mat = data_mat.colRange(layout_.gravity, layout_.gravity + 3);
	        gravity_.resize(kSamples);
	        for(int i=0; i<kSamples; ++i){
		        for(int j=0; j<3; ++j) {
			        gravity_[i][j] = static_cast<double>(mat.at<float>(i, j));
		        }
	        }
            LOG(INFO) << "Gravity loaded";
        }

        if(load_control & IMU_ROTATION_VECTOR){
	        cv::Mat mat = data_mat.colRange(layout_.rotation_vector, layout_.rotation_vector + 4);
	        rotation_vector_.resize(kSamples);
	        for(int i=0; i<kSamples; ++i){
		        const double* v = (double *) mat.ptr(i);
		        rotation_vector_[i].w() = v[0];
		        rotation_vector_[i].x() = v[1];
		        rotation_vector_[i].y() = v[2];
		        rotation_vector_[i].z() = v[3];
	        }
            LOG(INFO) << "Rotation vector loaded";
        }
    }

	void WriteToPly(const std::string& path, const std::vector<Eigen::Vector3d>& position,
	                const std::vector<Eigen::Quaterniond>& orientation,
	                const double axis_length, const int kpoints, const int interval){
	    using TriMesh = OpenMesh::TriMesh_ArrayKernelT<>;
	    TriMesh mesh;
        mesh.request_vertex_colors();

	    constexpr int traj_color[3] = {0, 255, 255};
	    constexpr int axis_color[3][3] = {{255, 0, 0}, {0, 255, 0}, {0, 0, 255}};

	    // First add trajectory points
	    for (int i = 0; i < position.size(); ++i) {
		    TriMesh::VertexHandle vertex = mesh.add_vertex(TriMesh::Point(position[i][0], position[i][1], position[i][2]));
		    mesh.set_color(vertex, TriMesh::Color(traj_color[0], traj_color[1], traj_color[2]));
	    }

        // Then add axis points
	    if (kpoints > 0 && interval > 0 && axis_length > 0) {
		    Eigen::Matrix3d local_axis = Eigen::Matrix3d::Identity();
		    for (int i = 0; i < position.size(); i += interval) {
			    Eigen::Matrix3d axis_dir = orientation[i].toRotationMatrix() * local_axis;
			    for (int j = 0; j < kpoints; ++j) {
				    for(int k=0; k<3; ++k){
					    Eigen::Vector3d pt = position[i] + axis_length / kpoints * j * axis_dir.block<3,1>(0, k);
					    TriMesh::VertexHandle vertex = mesh.add_vertex(TriMesh::Point(pt[0], pt[1], pt[2]));
					    mesh.set_color(vertex, TriMesh::Color(axis_color[k][0], axis_color[k][1], axis_color[k][2]));
				    }
			    }
		    }
	    }
	    // Write file
	    OpenMesh::IO::Options wopt;
	    wopt += OpenMesh::IO::Options::VertexColor;

	    try{
		    OpenMesh::IO::write_mesh(mesh, path, wopt);
	    }catch(const std::runtime_error& e){
            CHECK(true) << e.what();
	    }
    }

}//namespace IMUProject