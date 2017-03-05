//
// Created by Yan Hang on 3/4/17.
//

#ifndef PROJECT_SPEED_REGRESSION_H
#define PROJECT_SPEED_REGRESSION_H

#include <vector>
#include <string>

#include <opencv2/opencv.hpp>
#include <Eigen/Eigen>
#include <Eigen/Geometry>
#include <glog/logging.h>

namespace IMUProject {

	enum TargetType{
		LOCAL_SPEED,
		SPEED_MAGNITUDE,
	};

	enum FeatureType{
		DIRECT,
		FOURIER
	};

	struct TrainingDataOption{
		TrainingDataOption(const int step=10, const int window=200,
		                   const FeatureType feature=DIRECT, const TargetType target=LOCAL_SPEED):
				step_(step), window_(window), feature_(feature), target_(target){}
		int step_;
		int window_;

		FeatureType feature_;
		TargetType target_;
	};

	cv::Mat ComputeLocalSpeedTarget(const std::vector<double>& time_stamp,
	                                const std::vector<Eigen::Vector3d>& position,
	                                const std::vector<Eigen::Quaterniond>& orientation,
	                                const std::vector<int>& sample_points,
	                                const int smooth_size);

	cv::Mat ComputeDirectFeature(const Eigen::Vector3d* gyro,
	                             const Eigen::Vector3d* linacce, const int N);



} // namespace IMUProject

#endif //PROJECT_SPEED_REGRESSION_H
