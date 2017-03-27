//
// Created by yanhang on 3/5/17.
//

#ifndef PROJECT_IMU_LOCALIZATION_H
#define PROJECT_IMU_LOCALIZATION_H

#include <thread>
#include <mutex>
#include <deque>
#include <condition_variable>
#include <atomic>

#include <Eigen/Eigen>
#include <vector>

#include <opencv2/opencv.hpp>

#include <imu_optimization/imu_optimization.h>

namespace IMUProject{

	enum RegressionOption{
		FULL,
		MAG,
		ORI,
        Z_ONLY,
		CONST
	};

    struct IMULocalizationOption{
        int local_opt_interval_ = 400;
        int local_opt_window_ = 1000;

        int global_opt_interval_ = 2000;
        double weight_ls_ = 1.0;
        double weight_vs_ = 1.0;

	    RegressionOption reg_option_ = FULL;

	    double const_speed_ = 1.0;

        static constexpr int reg_interval_ = 50;
        static constexpr int reg_window_ = 200;
    };

    struct FunctorSize{
        static constexpr int kVar_600_ = 12;
        static constexpr int kCon_600_ = 8;

        static constexpr int kVar_800_ = 16;
        static constexpr int kCon_800_ = 12;

        static constexpr int kVar_1000_ = 20;
        static constexpr int kCon_1000_ = 16;

        static constexpr int kVar_5000_ = 100;
        static constexpr int kCon_5000_ = 96;

        static constexpr int kVar_large_ = 100;
        static constexpr int kCon_large_ = 200;
    };

    class IMUTrajectory{
    public:

        IMUTrajectory(const Eigen::Vector3d& init_speed,
                      const Eigen::Vector3d& init_position,
                      const std::vector<cv::Ptr<cv::ml::SVM> >& regressors,
                      const double sigma = 0.2,
                      const IMULocalizationOption option=IMULocalizationOption());

        ~IMUTrajectory(){
            terminate_flag_.store(true);
            if(opt_thread_.joinable()) {
                opt_thread_.join();
            }
        }

        void AddRecord(const double t, const Eigen::Vector3d& gyro, const Eigen::Vector3d& linacce,
                       const Eigen::Vector3d& gravity, const Eigen::Quaterniond& orientation);

        /// Run optimization
        /// \param start_id The start index of optimization window. Pass -1 to run global optimization
        /// \param N
        void RunOptimization(const int start_id, const int N);

        void StartOptmizationThread();

        inline void ScheduleOptimization(const int start_id, const int N){
            std::lock_guard<std::mutex> guard(queue_lock_);
            task_queue_.push_back(std::make_pair(start_id, N));
            if(task_queue_.size() >= max_queue_task_){
                can_add_.store(false);
            }
            cv_.notify_all();
        }

        int RegressSpeed(const int end_ind);

        template<class FunctorType, int kVar, int kCon>
        const SparseGrid* ConstructProblem(const int start_id, const int N, ceres::Problem& problem,
                                           const int* constraint_ind, const Eigen::Vector3d* local_speed,
                                           const Eigen::Vector3d init_speed,
                                           std::vector<double>& bx, std::vector<double>& by, std::vector<double>& bz){
            CHECK_GE(constraint_ind_.size(), kCon);

            FunctorType * functor = new FunctorType(&ts_[start_id], N, &linacce_[start_id],
                                                    &orientation_[start_id], &R_GW_[start_id],
                                                    constraint_ind, local_speed, init_speed,
                                                    option_.weight_ls_,option_.weight_vs_);
            bx.resize((size_t)kVar,0.0);
            by.resize((size_t)kVar,0.0);
            bz.resize((size_t)kVar,0.0);

            problem.AddResidualBlock(new ceres::AutoDiffCostFunction<FunctorType, 3 * kCon, kVar, kVar, kVar>(functor),
                                     nullptr, bx.data(), by.data(), bz.data());

            problem.AddResidualBlock(new ceres::AutoDiffCostFunction<WeightDecay<kVar>, kVar, kVar>(new WeightDecay<kVar>(1.0)), nullptr, bx.data());
            problem.AddResidualBlock(new ceres::AutoDiffCostFunction<WeightDecay<kVar>, kVar, kVar>(new WeightDecay<kVar>(1.0)), nullptr, by.data());
            problem.AddResidualBlock(new ceres::AutoDiffCostFunction<WeightDecay<kVar>, kVar, kVar>(new WeightDecay<kVar>(1.0)), nullptr, bz.data());

            return functor->GetLinacceGrid();
        }

        inline const Eigen::Vector3d& GetCurrentSpeed() const{
            std::lock_guard<std::mutex> guard(mt_);
            return position_.back();
        }

        inline const Eigen::Quaterniond& GetCurrentOrientation() const{
            return orientation_.back();
        }

        inline const Eigen::Vector3d& GetCurrentPosition() const{
            std::lock_guard<std::mutex> guard(mt_);
            return position_.back();
        }

        inline const std::vector<Eigen::Vector3d>& GetPositions() const{
            std::lock_guard<std::mutex> guard(mt_);
            return position_;
        }

        inline const std::vector<Eigen::Vector3d>& GetSpeed() const{
            std::lock_guard<std::mutex> guard(mt_);
            return speed_;
        }

        inline const std::vector<Eigen::Quaterniond>& GetOrientations() const{
            std::lock_guard<std::mutex> guard(mt_);
            return orientation_;
        }

        inline const std::vector<Eigen::Vector3d>& GetLinearAcceleration() const{
            std::lock_guard<std::mutex> guard(mt_);
            return linacce_;
        }

        inline const std::vector<int>& GetConstraintInd() const{
            std::lock_guard<std::mutex> guard(mt_);
            return constraint_ind_;
        }

        inline const std::vector<Eigen::Vector3d>& GetLocalSpeed() const{
            std::lock_guard<std::mutex> guard(mt_);
            return local_speed_;
        }

        inline const int GetNumFrames() const{
            std::lock_guard<std::mutex> guard(mt_);
            return num_frames_;
        }

        inline bool CanAdd() const{
            return can_add_.load();
        }

        inline void EndTrajectory(){
            terminate_flag_.store(true);
            cv_.notify_all();
            if(opt_thread_.joinable()) {
                opt_thread_.join();
            }
        }

        void CommitOptimizationResult(const SparseGrid* grid, const int start_id,
                                      const double* bx, const double* by, const double *bz);


        static constexpr int kInitCapacity_ = 10000;

        static const Eigen::Vector3d local_gravity_dir_;

    private:
        std::vector<double> ts_;
        std::vector<Eigen::Vector3d> linacce_;
        std::vector<Eigen::Vector3d> gyro_;
        std::vector<Eigen::Vector3d> gravity_;
        std::vector<Eigen::Quaterniond> orientation_;
        std::vector<Eigen::Quaterniond> R_GW_;

        std::vector<Eigen::Vector3d> speed_;
        std::vector<Eigen::Vector3d> position_;

        std::vector<int> constraint_ind_;
        std::vector<Eigen::Vector3d> local_speed_;
        int last_constraint_ind_;

        const std::vector<cv::Ptr<cv::ml::SVM> >& regressors_;

        Eigen::Vector3d init_speed_;
        Eigen::Vector3d init_position_;

        std::deque<std::pair<int, int> > task_queue_;

        int num_frames_;

        const double sigma_;

        const IMULocalizationOption option_;

        mutable std::mutex mt_;
        mutable std::mutex queue_lock_;
        static constexpr int max_queue_task_ = 3;

        std::condition_variable cv_;
        std::atomic<bool> terminate_flag_;
        std::atomic<bool> can_add_;
        std::thread opt_thread_;
    };

}//namespace IMUProject

#endif //PROJECT_IMU_LOCALIZATION_H
