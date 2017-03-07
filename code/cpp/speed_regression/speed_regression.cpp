//
// Created by Yan Hang on 3/4/17.
//

#include "speed_regression.h"

using namespace std;
using namespace cv;

namespace IMUProject{

	cv::Mat ComputeLocalSpeedTarget(const std::vector<double>& time_stamp,
	                                const std::vector<Eigen::Vector3d>& position,
	                                const std::vector<Eigen::Quaterniond>& orientation,
	                                const std::vector<int>& sample_points,
	                                const int smooth_size){
		const int N = (int)time_stamp.size();
		CHECK_EQ(position.size(), N);
		CHECK_EQ(orientation.size(), N);

		std::vector<Eigen::Vector3d> global_speed(position.size(), Eigen::Vector3d(0, 0, 0));
		for(auto i=0; i<N - 1; ++i){
			global_speed[i] = (position[i + 1] - position[i]) / (time_stamp[i + 1] - time_stamp[i]);
		}
		global_speed[global_speed.size() - 2] = global_speed[global_speed.size() - 1];

		Mat local_speed_all(N, 3, CV_32FC1, cv::Scalar::all(0));
		float* ls_ptr = (float *) local_speed_all.data;
		for(auto i=0; i<N; ++i){
			Eigen::Vector3d local_speed = orientation[i].conjugate() * global_speed[i];
			ls_ptr[i * 3] = (float)local_speed[0];
			ls_ptr[i * 3 + 1] = (float)local_speed[1];
			ls_ptr[i * 3 + 2] = (float)local_speed[2];
		}

//		Mat local_speed_filtered(N, 3, CV_32FC1, cv::Scalar::all(0));
//		for(int i=0; i<3; ++i) {
//			//cv::blur(local_speed_all.col(i), local_speed_filtered.col(i), cv::Size(smooth_size, 1));
//			cv::GaussianBlur(local_speed_all.col(i), local_speed_filtered.col(i), cv::Size(0, 0), (double)smooth_size);
//		}
		Mat local_speed_filtered = local_speed_all;

		Mat local_speed((int)sample_points.size(), 3, CV_32FC1, cv::Scalar::all(0));
		for(auto i=0; i<sample_points.size(); ++i){
			const int ind = sample_points[i];
			local_speed.at<float>(i, 0) = local_speed_filtered.at<float>(ind, 0);
			local_speed.at<float>(i, 1) = local_speed_filtered.at<float>(ind, 1);
			local_speed.at<float>(i, 2) = local_speed_filtered.at<float>(ind, 2);
		}

		return local_speed;
	}

	cv::Mat ComputeDirectFeature(const Eigen::Vector3d* gyro,
	                             const Eigen::Vector3d* linacce,
								 const int N){
		Mat feature(1, 6 * N, CV_32FC1, cv::Scalar::all(0));
		for(int i=0; i<N; ++i){
			feature.at<float>(0, i * 6 + 0) = (float) gyro[i][0];
			feature.at<float>(0, i * 6 + 1) = (float) gyro[i][1];
			feature.at<float>(0, i * 6 + 2) = (float) gyro[i][2];
			feature.at<float>(0, i * 6 + 3) = (float) linacce[i][0];
			feature.at<float>(0, i * 6 + 4) = (float) linacce[i][1];
			feature.at<float>(0, i * 6 + 5) = (float) linacce[i][2];
		}

		return feature;
	}


}