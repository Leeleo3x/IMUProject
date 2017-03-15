//
// Created by yanhang on 3/5/17.
//

#include <chrono>

#include <utility/utility.h>
#include <speed_regression/speed_regression.h>

#include "imu_localization.h"
using namespace std;
namespace IMUProject{

	using Functor600 = LocalSpeedFunctor<FunctorSize::kVar_600_, FunctorSize::kCon_600_>;
	using Functor800 = LocalSpeedFunctor<FunctorSize::kVar_800_, FunctorSize::kCon_800_>;
	using Functor1000 = LocalSpeedFunctor<FunctorSize::kVar_1000_, FunctorSize::kCon_1000_>;
	using Functor5000 = LocalSpeedFunctor<FunctorSize::kVar_5000_, FunctorSize::kCon_5000_>;
	using FunctorLarge = LocalSpeedFunctor<FunctorSize::kVar_large_, FunctorSize::kCon_large_>;

	const Eigen::Vector3d IMUTrajectory::local_gravity_dir_ = Eigen::Vector3d(0, 1, 0);

	IMUTrajectory::IMUTrajectory(const Eigen::Vector3d& init_speed,
	                             const Eigen::Vector3d& init_position,
	                             const std::vector<cv::Ptr<cv::ml::SVM> >& regressors,
	                             const double sigma,
                                 const IMULocalizationOption option):
            init_speed_(init_speed), init_position_(init_position), num_frames_(0),
            regressors_(regressors), last_constraint_ind_(option.reg_window_ - option.reg_interval_),
            option_(option), sigma_(sigma) {
		ts_.reserve(kInitCapacity_);
	    gyro_.reserve(kInitCapacity_);
        linacce_.reserve(kInitCapacity_);
        orientation_.reserve(kInitCapacity_);
        speed_.reserve(kInitCapacity_);
	    terminate_flag_.store(false);
		can_add_.store(true);

		opt_thread_ = std::move(std::thread(&IMUTrajectory::StartOptmizationThread, this));
    }

	void IMUTrajectory::AddRecord(const double t, const Eigen::Vector3d& gyro, const Eigen::Vector3d& linacce,
								  const Eigen::Vector3d& gravity, const Eigen::Quaterniond& orientation){
		num_frames_ += 1;
		ts_.push_back(t);
		linacce_.push_back(linacce);
		orientation_.push_back(orientation);
		gyro_.push_back(gyro);
		gravity_.push_back(gravity);

		Eigen::Quaterniond rotor_g = Eigen::Quaterniond::FromTwoVectors(gravity, local_gravity_dir_);
		R_GW_.push_back(rotor_g * orientation.conjugate());
		//R_GW_.push_back(orientation.conjugate());

		//TODO: Try to remove this lock...
		std::lock_guard<std::mutex> guard(mt_);
		if(num_frames_ > 1){
			const double dt = ts_[num_frames_ - 1] - ts_[num_frames_-2];
			speed_.push_back(orientation * linacce * dt);
			position_.push_back(speed_[num_frames_ - 1] * dt);
		}else{
			speed_.push_back(init_speed_);
			position_.push_back(init_position_);
		}
	}

    void IMUTrajectory::CommitOptimizationResult(const SparseGrid* grid, const int start_id,
                                                 const double* bx, const double* by, const double *bz){
        // correct acceleration and re-do double integration
        CHECK_NOTNULL(grid)->correct_linacce_bias<double>(&linacce_[start_id], bx, by, bz);
		std::lock_guard<std::mutex> guard(mt_);
        for(int i= start_id + 1; i<num_frames_; ++i){
            const double dt = ts_[i] - ts_[i-1];
            speed_[i] = speed_[i-1] + orientation_[i-1] * linacce_[i-1] * dt;
            position_[i] = position_[i-1] + speed_[i-1] * dt;
        }
    }

    int IMUTrajectory::RegressSpeed(const int end_ind) {
        for(int i = last_constraint_ind_ + option_.reg_interval_; i < end_ind; i += option_.reg_interval_){
            std::vector<Eigen::Vector3d> gyro_slice(gyro_.begin() + i - option_.reg_window_, gyro_.begin() + i);
            std::vector<Eigen::Vector3d> linacce_slice(linacce_.begin() + i - option_.reg_window_,
                                                       linacce_.begin() + i);
			std::vector<Eigen::Vector3d> gravity_slice(gravity_.begin()+i-option_.reg_window_,
													   gravity_.begin()+i);
			GaussianFilter(gyro_slice.data(), (int)gyro_slice.size(), sigma_);
            GaussianFilter(linacce_slice.data(), (int)linacce_slice.size(), sigma_);

            cv::Mat feature = ComputeDirectFeatureGravity(gyro_slice.data(), linacce_slice.data(), gravity_slice.data(),
														  (int)gyro_slice.size());

			// cv::Mat feature = ComputeDirectFeature(gyro_slice.data(), linacce_slice.data(), (int)gyro_slice.size());

            // TODO: remove the redundant Y axis
            const double ls_x = static_cast<double>(regressors_[0]->predict(feature));
            const double ls_z = static_cast<double>(regressors_[2]->predict(feature));

            constraint_ind_.push_back(i);
            local_speed_.emplace_back(ls_x, 0, ls_z);
        }
        last_constraint_ind_ = constraint_ind_.back();

        return 0;
    }

	void IMUTrajectory::RunOptimization(const int start_id, const int N) {
		CHECK_GE(start_id, 0);
        CHECK_GE(N, 600);
        // Complete the speed regression up to this point

        RegressSpeed(start_id + N);

		ceres::Problem problem;
        const SparseGrid* grid = nullptr;
        std::vector<double> bx, by, bz;

		std::vector<int> cur_constraint_id;
		std::vector<Eigen::Vector3d> cur_local_speed;

		Eigen::Vector3d cur_init_speed;

		auto ConstructConstraint = [&](const int kCon){
			CHECK_GE(constraint_ind_.size(), kCon);
			if(start_id > 0) {
				for(auto i=constraint_ind_.size() - kCon; i<constraint_ind_.size(); ++i){
					cur_constraint_id.push_back(constraint_ind_[i] - start_id);
					cur_local_speed.push_back(local_speed_[i]);
				}
				cur_init_speed = speed_[start_id];
			}else{
				// Be sure to include the first and last constraint
				cur_init_speed = init_speed_;
				const float inc = (float)constraint_ind_.size() / ((float)kCon - 1);
				for(auto i=0; i<kCon - 1; ++i){
					cur_constraint_id.push_back(constraint_ind_[(int)(i * inc)]);
					cur_local_speed.push_back(local_speed_[(int)(i * inc)]);
				}
				cur_constraint_id.push_back(constraint_ind_.back());
				cur_local_speed.push_back(local_speed_.back());
			}
		};

        if(N >= 600 && N < 800){
			ConstructConstraint(FunctorSize::kCon_600_);
			CHECK_EQ(cur_constraint_id.size(), FunctorSize::kCon_600_);
			grid = ConstructProblem<Functor600, FunctorSize::kVar_600_, FunctorSize::kCon_600_>
					(start_id, N, problem, cur_constraint_id.data(), cur_local_speed.data(), cur_init_speed, bx, by, bz);
		}else if(N >= 800 && N < 1000){
			ConstructConstraint(FunctorSize::kCon_800_);
			CHECK_EQ(cur_constraint_id.size(), FunctorSize::kCon_800_);
            grid = ConstructProblem<Functor800, FunctorSize::kVar_800_, FunctorSize::kCon_800_>
					(start_id, N, problem, cur_constraint_id.data(), cur_local_speed.data(), cur_init_speed, bx, by, bz);
		}else if(N >= 1000 && N < 5000){
			ConstructConstraint(FunctorSize::kCon_1000_);
			CHECK_EQ(cur_constraint_id.size(), FunctorSize::kCon_1000_);
            grid = ConstructProblem<Functor1000, FunctorSize::kVar_1000_, FunctorSize::kCon_1000_>
					(start_id, N, problem, cur_constraint_id.data(), cur_local_speed.data(), cur_init_speed, bx, by, bz);
		}else if(N >= 5000 && N < 10100){
			ConstructConstraint(FunctorSize::kCon_5000_);
			CHECK_EQ(cur_constraint_id.size(), FunctorSize::kCon_5000_);
			grid = ConstructProblem<Functor5000, FunctorSize::kVar_5000_, FunctorSize::kCon_5000_>
					(start_id, N, problem, cur_constraint_id.data(), cur_local_speed.data(), cur_init_speed, bx, by, bz);
		}else{
			ConstructConstraint(FunctorSize::kCon_large_);
			CHECK_EQ(cur_constraint_id.size(), FunctorSize::kCon_large_);
            grid = ConstructProblem<FunctorLarge , FunctorSize::kVar_large_, FunctorSize::kCon_large_>
					(start_id, N, problem, cur_constraint_id.data(), cur_local_speed.data(), cur_init_speed, bx, by, bz);
		}

		ceres::Solver::Options solver_options;
		solver_options.max_num_iterations = 3;
		solver_options.linear_solver_type = ceres::DENSE_QR;

		ceres::Solver::Summary summary;
		ceres::Solve(solver_options, &problem, &summary);
		LOG(INFO) << summary.BriefReport();

        CommitOptimizationResult(grid, start_id, bx.data(), by.data(), bz.data());
	}

	void IMUTrajectory::StartOptmizationThread(){
		auto check_interval = std::chrono::microseconds(10);

		LOG(INFO) << "Background thread started";
		while(true){
			std::unique_lock<std::mutex> guard(queue_lock_);
			cv_.wait_for(guard, check_interval, [this]{return !task_queue_.empty();});
			if(terminate_flag_.load()){
				// Make sure to finish the remaining optimization task
				std::vector<std::pair<int, int > > remaining_tasks;
				for(const auto& t: task_queue_){
					remaining_tasks.push_back(t);
				}
				guard.unlock();

				for(const auto& t: remaining_tasks){
					RunOptimization(t.first, t.second);
				}
				break;
			}
			if(task_queue_.empty()){
				guard.unlock();
				continue;
			}

			std::pair<int, int> cur_task = task_queue_.front();
			task_queue_.pop_front();
			if(task_queue_.size() < max_queue_task_) {
				can_add_.store(true);
			}
			guard.unlock();

			RunOptimization(cur_task.first, cur_task.second);

		}
		LOG(INFO) << "Background thread terminated";
	}
}//namespace IMUProject
