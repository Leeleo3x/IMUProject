//
// Created by yanhang on 2/6/17.
//

#include "imu_optimization.h"

namespace IMUProject{

    SizedSharedSpeedFunctor::SizedSharedSpeedFunctor(const cv::Mat& time_stamp, const cv::Mat& linacce,
                                                     const cv::Mat& orientation,
                                                     const std::vector<int>& constraint_ind,
                                                     const std::vector<float>& target_speed_mag,
                                                     const std::vector<float>& target_vspeed,
                                                     const Eigen::Vector3d& init_speed):
            linacce_(linacce), constraint_ind_(constraint_ind),
            target_speed_mag_(target_speed_mag), target_vspeed_(target_vspeed),
            init_speed_(init_speed){
        // Sanity check
        CHECK_EQ(constraint_ind_.size(), kConstriantPoints);
        CHECK_EQ(constraint_ind_.size(), target_speed_mag_.size());
        CHECK_EQ(constraint_ind_.size(), target_vspeed_.size());
        CHECK_GE(time_stamp.rows, kTotalCount);
        CHECK_EQ(time_stamp.cols, 1);
        CHECK_EQ(linacce.rows, time_stamp.rows);
        CHECK_EQ(linacce.cols, 3);
        CHECK_EQ(orientation.rows, time_stamp.rows);
        CHECK_EQ(orientation.cols, 4);

        // stores orientations as rotation matrices
        rotations_.resize((size_t)orientation.rows);
        for(int i=0; i<kTotalCount; ++i){
            const float* q = (float *)orientation.ptr(i);
            rotations_[i] = Eigen::Quaterniond((double)q[0], (double)q[1], (double)q[2], (double)q[3]).toRotationMatrix();
        }

        variable_ind_.resize(kSparsePoints, 0);
        for(int i=0; i<kSparsePoints; ++i){
            variable_ind_[i] = i * kSparseInterval;
        }

        const float* ts = (float* ) time_stamp.data;
        alpha_.resize(kTotalCount, 0.0);
        inverse_ind_.resize(kTotalCount, 0);
        dt_.resize(kTotalCount, 0);
        for(int i=0; i<kTotalCount - 1; ++i){
            dt_[i] = ts[i + 1] - ts[i];
        }

        // Compute the interpolation weights and inverse indexing
        // y[i] = alpha[i-1] * x[i-1] + (1.0 - alpha[i-1]) * x[i]
        for(int j=0; j<kSparseInterval; ++j){
            alpha_[j] = 1.0f - (ts[j] - ts[0]) / (ts[variable_ind_[0]] - ts[0]);
        }

        for(int i=1; i<variable_ind_.size(); ++i){
            for(int j=0; j<kSparseInterval; ++j) {
                const int ind = i * kSparseInterval + j;
                inverse_ind_[ind] = i;
                alpha_[i * kSparseInterval + j] = 1.0f - (ts[ind] - ts[variable_ind_[i - 1]]) /
                                                         (ts[variable_ind_[i]] - ts[variable_ind_[i - 1]]);
            }
        }
    }
}//IMUProject