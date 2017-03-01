//
// Created by yanhang on 2/6/17.
//

#include "imu_optimization.h"

using namespace std;

namespace IMUProject {

    SizedSharedSpeedFunctor::SizedSharedSpeedFunctor(const cv::Mat &time_stamp, const cv::Mat &linacce,
                                                     const cv::Mat &orientation,
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
        CHECK_EQ(constraint_ind_.size(), target_speed_mag_.size());
        CHECK_EQ(constraint_ind_.size(), target_vspeed_.size());
        CHECK_EQ(time_stamp.type(), CV_64FC1);
        CHECK_EQ(linacce.type(), CV_64FC1);
        CHECK_EQ(orientation.type(), CV_64FC1);
        CHECK_GE(time_stamp.rows, Config::kTotalCount);
        CHECK_EQ(time_stamp.cols, 1);
        CHECK_EQ(linacce.rows, time_stamp.rows);
        CHECK_EQ(linacce.cols, 3);
        CHECK_EQ(orientation.rows, time_stamp.rows);
        CHECK_EQ(orientation.cols, 4);

        // stores orientations as rotation matrices
        rotations_.resize((size_t) orientation.rows);
        for (int i = 0; i < Config::kTotalCount; ++i) {
            const double *q = (double *) orientation.ptr(i);
            rotations_[i] = Eigen::Quaterniond(q[0], q[1], q[2], q[3]).toRotationMatrix();
        }

        variable_ind_.resize(Config::kSparsePoints, 0);
        for (int i = 0; i < Config::kSparsePoints; ++i) {
            variable_ind_[i] = (i + 1) * Config::kSparseInterval - 1;
        }

        if(variable_ind_.back() >= Config::kTotalCount - 1){
            variable_ind_.back() = Config::kTotalCount - 2;
        }

        const double *ts = (double *) time_stamp.data;
        alpha_.resize(Config::kTotalCount, 0.0);
        inverse_ind_.resize(Config::kTotalCount, 0);
        dt_.resize(Config::kTotalCount, 0);
        for (int i = 0; i < Config::kTotalCount - 1; ++i) {
            dt_[i] = ts[i + 1] - ts[i];
        }

        // Compute the interpolation weights and inverse indexing
        // y[i] = alpha[i-1] * x[i-1] + (1.0 - alpha[i-1]) * x[i]
        for (int j = 0; j < variable_ind_[0]; ++j) {
            alpha_[j] = 1.0 - (ts[j] - ts[0]) / (ts[variable_ind_[0]] - ts[0]);
        }

        for (int i = 1; i < variable_ind_.size(); ++i) {
            for(int j=variable_ind_[i-1]; j<variable_ind_[i]; ++j){
                inverse_ind_[j] = i;
                alpha_[j] = 1.0 - (ts[j] - ts[variable_ind_[i - 1]]) /
                                                   (ts[variable_ind_[i]] - ts[variable_ind_[i - 1]]);
            }
        }
    }

    cv::Mat  SizedSharedSpeedFunctor::correcte_acceleration(const cv::Mat& input, const std::vector<double>& dx,
                                                            const std::vector<double>& dy, const std::vector<double>& dz){
        cv::Mat corrected;

        return corrected;
    }
}//IMUProject