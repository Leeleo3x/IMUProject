//
// Created by yanhang on 2/6/17.
//

#include "imu_optimization.h"

using namespace std;

namespace IMUProject {

	SizedSharedSpeedFunctor::SizedSharedSpeedFunctor(const std::vector<double> &time_stamp,
	                                                 const std::vector<Eigen::Vector3d> &linacce,
	                                                 const std::vector<Eigen::Quaterniond> &orientation,
	                                                 const std::vector<int> &constraint_ind,
	                                                 const std::vector<double> &target_speed_mag,
	                                                 const std::vector<double> &target_vspeed,
	                                                 const Eigen::Vector3d &init_speed,
	                                                 const double weight_sm,
	                                                 const double weight_vs) :
			linacce_(linacce), constraint_ind_(constraint_ind),
			target_speed_mag_(target_speed_mag), target_vspeed_(target_vspeed),
			init_speed_(init_speed), weight_sm_(std::sqrt(weight_sm)), weight_vs_(std::sqrt(weight_vs)) {
		// Sanity check
		CHECK_EQ(constraint_ind_.size(), Config::kConstriantPoints);
		CHECK_LE(constraint_ind_.size(), target_speed_mag_.size());
		CHECK_LE(constraint_ind_.size(), target_vspeed_.size());

		// stores orientations as rotation matrices
		rotations_.resize(orientation.size());
		for (int i = 0; i < Config::kTotalCount; ++i) {
			rotations_[i] = orientation[i].toRotationMatrix();
		}

		variable_ind_.resize(Config::kSparsePoints, 0);
		for (int i = 0; i < Config::kSparsePoints; ++i) {
			variable_ind_[i] = (i + 1) * Config::kSparseInterval - 1;
		}

//		if (variable_ind_.back() >= Config::kTotalCount - 1) {
//			variable_ind_.back() = Config::kTotalCount - 2;
//		}

		alpha_.resize(Config::kTotalCount, 0.0);
		inverse_ind_.resize(Config::kTotalCount, 0);
		dt_.resize(Config::kTotalCount, 0);
		for (int i = 0; i < Config::kTotalCount - 1; ++i) {
			dt_[i] = time_stamp[i + 1] - time_stamp[i];
		}
		dt_[dt_.size() - 1] = dt_[dt_.size() - 2];

		// Compute the interpolation weights and inverse indexing
		// y[i] = (1.0 - alpha[i]) * x[i-1] + alpha[i] * x[i]
		for (int j = 0; j <= variable_ind_[0]; ++j) {
			alpha_[j] = (time_stamp[j] - time_stamp[0]) / (time_stamp[variable_ind_[0]] - time_stamp[0]);
		}

		for (int i = 1; i < variable_ind_.size(); ++i) {
			for (int j = variable_ind_[i - 1] + 1; j <= variable_ind_[i]; ++j) {
				inverse_ind_[j] = i;
				alpha_[j] = (time_stamp[j] - time_stamp[variable_ind_[i - 1]]) /
				            (time_stamp[variable_ind_[i]] - time_stamp[variable_ind_[i - 1]]);
			}
		}

	}

	std::vector<Eigen::Vector3d> SizedSharedSpeedFunctor::correcte_acceleration(const std::vector<Eigen::Vector3d> &input,
	                                               const std::vector<double> &bx, const std::vector<double> &by, const std::vector<double> &bz) {
		CHECK_EQ(input.size(), alpha_.size());
		CHECK_EQ(bx.size(), Config::kSparsePoints);
		CHECK_EQ(by.size(), bx.size());
		CHECK_EQ(bz.size(), bx.size());
		std::vector<Eigen::Vector3d> output(input.size());
		for(int i=0; i<output.size(); ++i) {
			const int inv_ind = inverse_ind_[i];
			output[i] = input[i] + alpha_[i] * Eigen::Vector3d(bx[inv_ind], by[inv_ind], bz[inv_ind]);
			if (inv_ind > 0) {
				output[i] += (1.0 - alpha_[i]) * Eigen::Vector3d(bx[inv_ind - 1],
				                                                 by[inv_ind - 1],
				                                                 bz[inv_ind - 1]);
			}
		}
		return output;
	}
}//IMUProject