//
// Created by yanhang on 2/6/17.
//

#ifndef PROJECT_IMU_OPTIMIZATION_H
#define PROJECT_IMU_OPTIMIZATION_H

#include <opencv2/opencv.hpp>
#include <Eigen/Eigen>
#include <glog/logging.h>
#include <ceres/ceres.h>

namespace IMUProject{

    struct SizedSharedSpeedFunctor{
    public:

        SizedSharedSpeedFunctor(const cv::Mat& time_stamp, const cv::Mat& linacce,
                                const cv::Mat& orientation,
                                const std::vector<int>& constraint_ind,
                                const std::vector<float>& target_speed_mag,
                                const std::vector<float>& target_vspeed,
                                const Eigen::Vector3d& init_speed);

        template <typename T>
        bool operator () (const T* const bx, const T* const by, const T* const bz, T* residual){
            Eigen::Matrix<T, 3, 1> speed = init_speed_.cast<T>();
            int cid = 0;
            for(int i=0; i<kTotalCount; ++i){
                const float* acce = (float *)linacce_.ptr(i);
                if(i == constraint_ind_[cid]){
                    // Assign residual block
                    residual[cid] = speed.norm() - (T)target_speed_mag_[cid];
                    residual[cid + kVSpeedConstraintOffset] = speed[2] - (T)target_vspeed_[cid];
                    cid++;
                }

                Eigen::Matrix<T, 3, 1> m_acce((T)acce[0], (T)acce[1], (T)acce[2]);
                if(inverse_ind_[i] == 0){
                    // Be careful about the boundary case
                    m_acce(0, 0) += bx[inverse_ind_[i]] * (1.0f - alpha_[i]);
                    m_acce(1, 0) += by[inverse_ind_[i]] * (1.0f - alpha_[i]);
                    m_acce(2, 0) += bz[inverse_ind_[i]] * (1.0f - alpha_[i]);
                }else {
                    m_acce(0, 0) += bx[inverse_ind_[i] - 1] * alpha_[i] + bx[inverse_ind_[i]] * (1.0f - alpha_[i]);
                    m_acce(1, 0) += by[inverse_ind_[i] - 1] * alpha_[i] + by[inverse_ind_[i]] * (1.0f - alpha_[i]);
                    m_acce(2, 0) += bz[inverse_ind_[i] - 1] * alpha_[i] + bz[inverse_ind_[i]] * (1.0f - alpha_[i]);
                }

                // Rotate accelerate and add to speed
                speed += rotations_[i] * m_acce;
            }
            return true;
        }

        static constexpr int kConstriantPoints = 500;
        static constexpr int kSparsePoints = 1000;
        static constexpr int kSparseInterval = 5;
        static constexpr int kTotalCount = 5000;
        static constexpr int kVSpeedConstraintOffset = 500;
    private:
        const cv::Mat& linacce_;

        //store all quaternions as rotation matrix
        std::vector<Eigen::Matrix3d> rotations_;
        std::vector<double> alpha_;
        std::vector<double> dt_;
        std::vector<int> inverse_ind_;
        std::vector<int> variable_ind_;
        const std::vector<int> constraint_ind_;
        const std::vector<double> target_speed_mag_;
        const std::vector<double> target_vspeed_;

        const Eigen::Vector3d init_speed_;
    };


}//IMUProject
#endif //PROJECT_IMU_OPTIMIZATION_H
