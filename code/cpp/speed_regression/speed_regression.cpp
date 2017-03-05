//
// Created by Yan Hang on 3/4/17.
//

#include "speed_regression.h"

using namespace std;
using namespace cv;

namespace IMUProject{

	cv::Mat ComputeLocalSpeed(const std::vector<double>& time_stamp,
	                          const std::vector<Eigen::Vector3d>& position,
	                          const std::vector<Eigen::Quaterniond>& orientation,
	                          const std::vector<int>& sample_points,
	                          const int smooth_size){
		const int N = (int)time_stamp.size();
		CHECK_EQ(position.size(), N);
		CHECK_EQ(orientation.size(), N);

		std::vector<Eigen::Vector3d> global_speed(position.size());
		for(auto i=1; i<N; ++i){
			global_speed[i] = (position[i] - position[i-1]) / (time_stamp[i] - time_stamp[i-1]);
		}

		Mat local_speed_all(N, 3, CV_32FC1, cv::Scalar::all(0));
		float* ls_ptr = (float *) local_speed_all.data;
		for(auto i=1; i<N; ++i){
			Eigen::Vector3d local_speed = orientation[i].inverse() * global_speed[i];
			ls_ptr[i * 3] = (float)local_speed[0];
			ls_ptr[i * 3 + 1] = (float)local_speed[1];
			ls_ptr[i * 3 + 2] = (float)local_speed[2];
		}

		Mat local_speed_filtered(N, 3, CV_32FC1, cv::Scalar::all(0));
		cv::blur(local_speed_all, local_speed_filtered, cv::Size(smooth_size, 1));

		Mat local_speed((int)sample_points.size(), 3, CV_32FC1, cv::Scalar::all(0));
		for(auto i=0; i<sample_points.size(); ++i){
			const int ind = sample_points[i];
			local_speed.at<float>(i, 0) = local_speed_filtered.at<float>(ind, 0);
			local_speed.at<float>(i, 1) = local_speed_filtered.at<float>(ind, 1);
			local_speed.at<float>(i, 2) = local_speed_filtered.at<float>(ind, 2);
		}

		return local_speed;
	}

	cv::Mat ComputeDirectFeature(const std::vector<Eigen::Vector3d>& gyro,
	                             const std::vector<Eigen::Vector3d>& linacce,
	                             const int smooth_size){
		const int N = (int) gyro.size();
		CHECK_EQ(N, (int)linacce.size());
		Mat feature(1, 6 * N, CV_32FC1, cv::Scalar::all(0));
		for(int i=0; i<N; ++i){
			feature.at<float>(i, 0) = (float) gyro[i][0];
			feature.at<float>(i, 1) = (float) gyro[i][1];
			feature.at<float>(i, 2) = (float) gyro[i][2];
			feature.at<float>(i, 3) = (float) linacce[i][0];
			feature.at<float>(i, 4) = (float) linacce[i][1];
			feature.at<float>(i, 5) = (float) linacce[i][2];
		}

		return feature;
	}


}