//
// Created by yanhang on 3/5/17.
//

#include "imu_localization.h"


using namespace std;
namespace IMUProject{

	using Functor600 = LocalSpeedFunctor<12, 5>;
	using Functor800 = LocalSpeedFunctor<16, 7>;
	using Functor1000 = LocalSpeedFunctor<20, 9>;
	using FunctorLarge = LocalSpeedFunctor<100, 500>;

	IMUTrajectory::IMUTrajectory(const Eigen::Vector3d& init_speed,
	                             const Eigen::Vector3d& init_position,
	                             const std::vector<cv::Ptr<cv::ml::SVM> >& regressors,
	                             const double sigma):init_speed_(init_speed), init_position_(init_position),
	                                                 regressors_(regressors), sigma_(sigma) {
		ts_.reserve(kInitCapacity_);
	    gyro_.reserve(kInitCapacity_);
        linacce_.reserve(kInitCapacity_);
        orientation_.reserve(kInitCapacity_);
        speed_.reserve(kInitCapacity_);

	    terminate_flag_.store(false);
    }

	void IMUTrajectory::AddRecord(const double t, const Eigen::Vector3d& gyro,
	                              const Eigen::Vector3d& linacce, const Eigen::Quaterniond& orientation){
		num_frames_ += 1;
		ts_.push_back(t);
		linacce_.push_back(linacce);
		orientation_.push_back(orientation);
		gyro_.push_back(gyro);
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

	void IMUTrajectory::RunOptimization(const int start_id, const int N) {
		ceres::Problem problem;
		if(N == 600){

		}else if(N == 800){

		}else if(N == 1000){

		}else{

		}
	}


	void IMUTrajectory::StartOptmizationThread(){
		while(!terminate_flag_.load()){
			std::unique_lock<std::mutex> guard(queue_lock_);
			cv_.wait(guard, [this]{return !task_queue_.empty();});
			std::pair<int, int> cur_task = task_queue_.front();
			task_queue_.pop_front();
			guard.release();

			RunOptimization(cur_task.first, cur_task.second);
		}
	}
}//namespace IMUProject