//
// Created by Yan Hang on 3/1/17.
//

#ifndef PROJECT_UTILITY_H
#define PROJECT_UTILITY_H

#include <opencv2/opencv.hpp>
#include <Eigen/Eigen>
#include <glog/logging.h>

namespace IMUProject{

	std::vector<Eigen::Vector3d> Rotate3DVector(const std::vector<Eigen::Vector3d>& input,
	                                            const std::vector<Eigen::Quaterniond>& orientation);
	std::vector<Eigen::Vector3d> Integration(const std::vector<double>& ts,
	                                         const std::vector<Eigen::Vector3d>& input,
	                                         const Eigen::Vector3d& initial=Eigen::Vector3d(0, 0, 0));

	void LowPassFilter(Eigen::Vector3d* data, const int N, const double alpha = 1.0);

	void GaussianFilter(Eigen::Vector3d* data, const int N, const double sigma);

}//namespace IMUProject

#endif //PROJECT_UTILITY_H
