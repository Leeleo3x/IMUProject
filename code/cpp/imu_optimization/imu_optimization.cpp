//
// Created by yanhang on 2/6/17.
//

#include "imu_optimization.h"

using namespace std;

namespace IMUProject {

	SparseGrid::SparseGrid(const std::vector<double> &time_stamp,
	                       const int variable_count,
	                       const std::vector<int>* variable_ind):
			kTotalCount((int)time_stamp.size()) ,kVariableCount(variable_count){
		alpha_.resize((size_t)kTotalCount);
		inverse_ind_.resize((size_t)kTotalCount);
		variable_ind_.resize((size_t)kVariableCount);

		if(variable_ind != nullptr){
			CHECK_EQ(variable_ind->size(), kVariableCount);
			for(auto i=0; i<kVariableCount; ++i){
				variable_ind_[i] = (*variable_ind)[i];
			}
		}else{
			const int interval = kTotalCount / kVariableCount;
			for(int i=0; i<kVariableCount; ++i){
				variable_ind_[i] = (i + 1) * interval - 1;
			}
		}

		// Compute the interpolation weights and inverse indexing
		// y[i] = (1.0 - alpha[i]) * x[i-1] + alpha[i] * x[i]
		for (int j = 0; j <= variable_ind_[0]; ++j) {
			CHECK_GT(time_stamp[variable_ind_[0]] - time_stamp[0], std::numeric_limits<double>::epsilon());
			alpha_[j] = (time_stamp[j] - time_stamp[0]) / (time_stamp[variable_ind_[0]] - time_stamp[0]);
		}

		for (int i = 1; i < variable_ind_.size(); ++i) {
			CHECK_GT(time_stamp[variable_ind_[i]] - time_stamp[variable_ind_[i-1]], std::numeric_limits<double>::epsilon())
				<< variable_ind_[i] << ' ' << variable_ind_[i-1] << ' ' << time_stamp[variable_ind_[i]] << ' ' <<time_stamp[variable_ind_[i-1]];
			for (int j = variable_ind_[i - 1] + 1; j <= variable_ind_[i]; ++j) {
				inverse_ind_[j] = i;
				alpha_[j] = (time_stamp[j] - time_stamp[variable_ind_[i - 1]]) /
				            (time_stamp[variable_ind_[i]] - time_stamp[variable_ind_[i - 1]]);
			}
		}
	}

	SizedSharedSpeedFunctor::SizedSharedSpeedFunctor(const std::vector<double> &time_stamp,
	                                                 const std::vector<Eigen::Vector3d> &linacce,
	                                                 const std::vector<Eigen::Quaterniond> &orientation,
	                                                 const std::vector<int> &constraint_ind,
	                                                 const std::vector<double> &target_speed_mag,
	                                                 const std::vector<double> &target_vspeed,
	                                                 const Eigen::Vector3d &init_speed,
	                                                 const double weight_sm,
	                                                 const double weight_vs) :
			linacce_(linacce), orientation_(orientation), constraint_ind_(constraint_ind),
			target_speed_mag_(target_speed_mag), target_vspeed_(target_vspeed),
			init_speed_(init_speed), weight_sm_(std::sqrt(weight_sm)), weight_vs_(std::sqrt(weight_vs)) {
		// Sanity check
		CHECK_EQ(constraint_ind_.size(), Config::kConstriantPoints);
		CHECK_LE(constraint_ind_.size(), target_speed_mag_.size());
		CHECK_LE(constraint_ind_.size(), target_vspeed_.size());

		grid_.reset(new SparseGrid(time_stamp, Config::kSparsePoints));

		dt_.resize(Config::kTotalCount, 0);
		for (int i = 0; i < Config::kTotalCount - 1; ++i) {
			dt_[i] = time_stamp[i + 1] - time_stamp[i];
		}
		dt_[dt_.size() - 1] = dt_[dt_.size() - 2];
	}

}//IMUProject