//
// Created by Yan Hang on 3/1/17.
//

#include "utility.h"

namespace IMUProject{

	std::vector<Eigen::Vector3d> Rotate3DVector(const std::vector<Eigen::Vector3d>& input,
	                                            const std::vector<Eigen::Quaterniond>& orientation){
		std::vector<Eigen::Vector3d> output(input.size());
		for(int i=0; i<input.size(); ++i){
			output[i] = orientation[i].toRotationMatrix() * input[i];
		}

		return output;
	}

	std::vector<Eigen::Vector3d> Rotate3DVector(const std::vector<Eigen::Vector3d>& input,
												const std::vector<Eigen::Matrix3d>& orientation){
		std::vector<Eigen::Vector3d> output(input.size());
		for(int i=0; i<input.size(); ++i){
			output[i] = orientation[i] * input[i];
		}

		return output;
	}

	std::vector<Eigen::Vector3d> Integration(const std::vector<double>& ts,
	                                         const std::vector<Eigen::Vector3d>& input,
	                                         const Eigen::Vector3d& initial){
		CHECK_EQ(ts.size(), input.size());
		std::vector<Eigen::Vector3d> output(ts.size());
		output[0] = initial;
		for(int i=1; i<ts.size(); ++i){
			output[i] = output[i-1] + (input[i-1] + input[i]) / 2.0 * (ts[i] - ts[i-1]);
		}

		return output;
	}

	void LowPassFilter(Eigen::Vector3d* data, const int N, const double alpha){

	}

	void GaussianFilter(Eigen::Vector3d* data, const int N, const double sigma){
		for(int i=0; i<3; ++i){
			cv::Mat col_mat(N, 1, CV_64FC1, cv::Scalar::all(0));
			double* ptr = (double*) col_mat.data;
			for(int j=0; j<N; ++j){
				ptr[j] = data[j][i];
			}
			cv::GaussianBlur(col_mat, col_mat, cv::Size(0, 0), sigma);
			for(int j=0; j<N; ++j){
				data[j][i] = ptr[j];
			}
		}
	}

}//namespace IMUProject