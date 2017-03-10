//
// Created by yanhang on 3/5/17.
//

#ifndef PROJECT_IMU_LOCALIZATION_H
#define PROJECT_IMU_LOCALIZATION_H

#include <thread>
#include <mutex>

#include <Eigen/Eigen>
#include <vector>

#include <imu_optimization/imu_optimization.h>

namespace IMUProject{

    struct IMULocalizationOption{
        explicit IMULocalizationOption(const int opt_interval=400): opt_interval_(opt_interval){}

        int opt_interval_;
        constexpr int opt_window_ = 2000;
        constexpr int kVar_ = 50;
        constexpr int kConstraint_ = 20;
        constexpr int reg_interval_ = 100;
    };

    class IMUTrajectory{
    public:
        IMUTrajectory(const Eigen::Vector3d& init_speed,
                      const Eigen::Vector3d& init_position,
                      const double sigma = 0.2);

        inline void AddRecord(const double t, const Eigen::Vector3d& linacce, const Eigen::Quaterniond& orientation){
            num_frames_ += 1;
            ts_.push_back(t);
            linacce_.push_back(linacce);
            orientation_.push_back(orientation);

            //TODO: Try to remove this lock...
            std::lock_guard<std::mutex> guard(mt_);
            if(num_frames_ > 1){
                const double dt = ts_[num_frames_] - ts_[num_frames_-1];
                speed_.push_back(orientation * linacce * dt);
                position_.push_back(speed_[num_frames_ - 1] * dt);
            }else{
                speed_.push_back(init_position_);
                position_.push_back(init_position_);
            }
        }

        inline const Eigen::Vector3d& GetCurrentSpeed() const{
            std::lock_guard<std::mutex> guard(mt_);
            return position_.back();
        }

        inline const Eigen::Quaterniond& GetCurrentOrientation() const{
            return orientation_.back();
        }

        void CommitOptimizationResult(const SparseGrid* grid, const int start_id,
                                      const double* bx, const double* by, const double *bz);


        static constexpr int kInitCapacity_ = 10000;

    private:
        std::vector<double> ts_;
        std::vector<Eigen::Vector3d> linacce_;
        std::vector<Eigen::Quaterniond> orientation_;
        std::vector<Eigen::Vector3d> speed_;
        std::vector<Eigen::Vector3d> position_;

        Eigen::Vector3d init_speed_;
        Eigen::Vector3d init_position_;

        int num_frames_;

        const double sigma_;

        mutable std::mutex mt_;
    };

}//namespace IMUProject

#endif //PROJECT_IMU_LOCALIZATION_H
