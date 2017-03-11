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
                      const std::vector<cv::Ptr<cv::ml::SVM> >& regressors,
                      const double sigma = 0.2);

       void AddRecord(const double t, const Eigen::Vector3d& gyro,
                      const Eigen::Vector3d& linacce, const Eigen::Quaterniond& orientation);

	    void RunOptimization(const int start_id, const int N);

	    void StartOptmizationThread();

	    inline void SubmitOptimizationTask(const int start_id, const int N){
		    std::lock_guard<std::mutex> guard(queue_lock_);
		    task_queue_.push_back(std::make_pair(start_id, N));
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
	    std::vector<Eigen::Vector3d> gyro_;
        std::vector<Eigen::Quaterniond> orientation_;
        std::vector<Eigen::Vector3d> speed_;
        std::vector<Eigen::Vector3d> position_;

	    std::vector<int> constraint_ind_;
	    std::vector<Eigen::Vector3d> local_speed_;

	    const std::vector<cv::Ptr<cv::ml::SVM> >& regressors_;

        Eigen::Vector3d init_speed_;
        Eigen::Vector3d init_position_;

	    std::deque<std::pair<int, int> > task_queue_;

        int num_frames_;

        const double sigma_;

        mutable std::mutex mt_;

	    mutable std::mutex queue_lock_;

	    std::condition_variable cv_;
	    std::atomic<bool> terminate_flag_;
	    std::thread opt_thread_;
    };

}//namespace IMUProject

#endif //PROJECT_IMU_LOCALIZATION_H
