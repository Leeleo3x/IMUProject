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
                    const double axis_length, const int kpoints, const int interval) {
	    using TriMesh = OpenMesh::TriMesh_ArrayKernelT<>;
	    CHECK_EQ(position.rows, orientation.rows);
	    CHECK_EQ(position.type(), CV_64FC1);
	    CHECK_EQ(orientation.type(), CV_64FC1);

	    TriMesh mesh;

	    constexpr int traj_color[3] = {0, 255, 255};
	    constexpr int axis_color[3][3] = {{255, 0, 0}, {0, 255, 0}, {0, 0, 255}};

	    // First add trajectory points
	    for (int i = 0; i < position.rows; ++i) {
		    const double *pt = (double *) position.ptr(i);
		    TriMesh::VertexHandle vertex = mesh.add_vertex(TriMesh::Point(pt[0], pt[1], pt[2]));
		    mesh.set_color(vertex, TriMesh::Color(traj_color[0], traj_color[1], traj_color[2]));
	    }

	    // Then add axis points
	    if (kpoints > 0 && interval > 0 && axis_length > 0) {
		    Eigen::Matrix3d local_axis = Eigen::Matrix3d::Identity();
		    for (int i = 0; i < position.rows; i += interval) {
			    const double *ori_ptr = (double *) orientation.ptr(i);
			    const double *pos_ptr = (double *) position.ptr(i);
			    Eigen::Quaterniond q(ori_ptr[0], ori_ptr[1], ori_ptr[2], ori_ptr[3]);
			    Eigen::Matrix3d axis_dir = q.toRotationMatrix() * local_axis;
			    Eigen::Vector3d pos(pos_ptr[0], pos_ptr[1], pos_ptr[2]);
			    for (int j = 0; j < kpoints; ++j) {
				    for(int k=0; k<3; ++k){
					    Eigen::Vector3d pt = pos + axis_length / kpoints * j * axis_dir.block<3,1>(0, k);
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
		    std::cout << e.what();
	    }
    }

}//namespace IMUProject