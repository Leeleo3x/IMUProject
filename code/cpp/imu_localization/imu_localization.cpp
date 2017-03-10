//
// Created by yanhang on 3/5/17.
//

#include "imu_localization.h"

using namespace std;
namespace IMUProject{

    IMUTrajectory::IMUTrajectory(const Eigen::Vector3d& init_speed,
                                 const Eigen::Vector3d& init_position,
                                 const double sigma):init_speed_(init_speed),
                                                     init_position_(init_position), sigma_(sigma) {
        ts_.reserve(kInitCapacity_);
        linacce_.reserve(kInitCapacity_);
        orientation_.reserve(kInitCapacity_);
        speed_.reserve(kInitCapacity_);
    }

    void IMUTrajectory::CommitOptimizationResult(const SparseGrid* grid, const int start_id,
                                                 const double* bx, const double* by, const double *bz){
        std::lock_guard<std::mutex> guard(mt_);

        // correct acceleration and re-do double integration
        grid->correct_linacce_bias<double>(&linacce_[start_id], bx, by, bz);
        for(int i= start_id + 1; i<num_frames_; ++i){
            const double dt = ts_[i] - ts_[i-1];
            speed_[i] = speed_[i-1] + orientation_[i-1] * linacce_[i-1] * dt;
            position_[i] = position_[i-1] + speed_[i-1] * dt;
        }
    }


}//namespace IMUProject