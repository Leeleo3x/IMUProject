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

    struct Config{
        static constexpr int kTotalCount = 5000;
        static constexpr int kConstriantPoints = 500;
        static constexpr int kSparsePoints = 250;
        static constexpr int kSparseInterval = 20;
        static constexpr int kVSpeedConstraintOffset = 500;
    };


    struct SizedSharedSpeedFunctor{
    public:
        SizedSharedSpeedFunctor(const cv::Mat& time_stamp, const cv::Mat& linacce,
                                const cv::Mat& orientation,
                                const std::vector<int>& constraint_ind,
                                const std::vector<double>& target_speed_mag,
                                const std::vector<double>& target_vspeed,
                                const Eigen::Vector3d& init_speed,
                                const double weight_sm=1.0,
                                const double weight_vs=1.0);

//        template <typename T>
//        bool operator () (const T* const bx, const T* const by, const T* const bz, T* residual) const {
//            Eigen::Matrix<T, 3, 1> speed = init_speed_.cast<T>();
//            for (int i = 0; i < Config::kConstriantPoints * 2; ++i) {
//                residual[i] = (T) 0.0;
//            }
//            int cid = 0;
//            for (int i = 0; i < Config::kTotalCount; ++i) {
//                const double *acce = (double *) linacce_.ptr(i);
//                if (i == constraint_ind_[cid]) {
//                    // Assign residual block
//                    residual[cid] = weight_sm_ * (speed.norm() - (T) target_speed_mag_[cid]);
//                    residual[cid + Config::kVSpeedConstraintOffset] = weight_vs_ * (speed[2] - (T) target_vspeed_[cid]);
//                    cid++;
//                }
//
//                Eigen::Matrix<T, 3, 1> m_acce((T) acce[0], (T) acce[1], (T) acce[2]);
//                if (inverse_ind_[i] == 0) {
//                    // Be careful about the boundary case
//                    m_acce(0, 0) += bx[inverse_ind_[i]] * (1.0f - alpha_[i]);
//                    m_acce(1, 0) += by[inverse_ind_[i]] * (1.0f - alpha_[i]);
//                    m_acce(2, 0) += bz[inverse_ind_[i]] * (1.0f - alpha_[i]);
//                } else {
//                    m_acce(0, 0) += bx[inverse_ind_[i] - 1] * alpha_[i] + bx[inverse_ind_[i]] * (1.0f - alpha_[i]);
//                    m_acce(1, 0) += by[inverse_ind_[i] - 1] * alpha_[i] + by[inverse_ind_[i]] * (1.0f - alpha_[i]);
//                    m_acce(2, 0) += bz[inverse_ind_[i] - 1] * alpha_[i] + bz[inverse_ind_[i]] * (1.0f - alpha_[i]);
//                }
//
//                // Rotate accelerate and add to speed
//                speed += rotations_[i].cast<T>() * m_acce;
//            }
//
//            return true;
//        }


            bool operator () (const double* const bx, const double* const by, const double* const bz, double* residual) const{
                for(int i=0; i<Config::kConstriantPoints * 2; ++i){
                    residual[i] = 0.0;
                }
                int cid = 0;

                std::vector<Eigen::Vector3d> corrected_acce((size_t)Config::kTotalCount);
#pragma omp parallel for
                for(int i=0; i<Config::kTotalCount; ++i){
                    corrected_acce[i] = 
                }

                Eigen::Vector3d speed = init_speed_;
                for(int i=0; i<Config::kTotalCount; ++i){
                    const double* acce = (double *)linacce_.ptr(i);
                    if(i == constraint_ind_[cid]){
                        // Assign residual block
                        residual[cid] = weight_sm_ * (speed.norm() - target_speed_mag_[cid]);
                        residual[cid + Config::kVSpeedConstraintOffset] = weight_vs_ * (speed[2] - target_vspeed_[cid]);
                        cid++;
                    }

                    Eigen::Vector3d m_acce(acce[0], acce[1], acce[2]);
                    if(inverse_ind_[i] == 0){
                        // Be careful about the boundary case
                        m_acce[0] += bx[inverse_ind_[i]] * (1.0f - alpha_[i]);
                        m_acce[1] += by[inverse_ind_[i]] * (1.0f - alpha_[i]);
                        m_acce[2] += bz[inverse_ind_[i]] * (1.0f - alpha_[i]);
                    }else {
                        m_acce[0] += bx[inverse_ind_[i] - 1] * alpha_[i] + bx[inverse_ind_[i]] * (1.0f - alpha_[i]);
                        m_acce[1] += by[inverse_ind_[i] - 1] * alpha_[i] + by[inverse_ind_[i]] * (1.0f - alpha_[i]);
                        m_acce[2] += bz[inverse_ind_[i] - 1] * alpha_[i] + bz[inverse_ind_[i]] * (1.0f - alpha_[i]);
                    }
                    speed += dt_[i] * rotations_[i] * m_acce;
                }
            return true;
        }

        cv::Mat correcte_acceleration(const cv::Mat& input, const std::vector<double>& dx,
                                      const std::vector<double>& dy, const std::vector<double>& dz);

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

        const double weight_sm_;
        const double weight_vs_;
    };


    struct WeightDecay{
    public:
        WeightDecay(const double weight): weight_(std::sqrt(weight)){}
//        template <typename T>
//        bool operator () (const T* const x, T* residual) const{
//            for(int i=0; i<Config::kSparsePoints; ++i){
//                residual[i] = weight_ * x[i];
//            }
//            return true;
//        }

        bool operator () (const double* const x, double* residual) const{
            for(int i=0; i<Config::kSparsePoints; ++i){
                residual[i] = weight_ * x[i];
            }
            return true;
        }

    private:
        const double weight_;
    };


}//IMUProject
#endif //PROJECT_IMU_OPTIMIZATION_H
