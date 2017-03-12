//
// Created by yanhang on 2/6/17.
//

#ifndef PROJECT_IMU_OPTIMIZATION_H
#define PROJECT_IMU_OPTIMIZATION_H

#include <memory>

#include <opencv2/opencv.hpp>
#include <Eigen/Eigen>
#include <glog/logging.h>
#include <ceres/ceres.h>
#include <ceres/rotation.h>

namespace IMUProject {

	struct Config {
		static constexpr int kConstriantPoints = 100;
		static constexpr int kSparsePoints = 200;
		static constexpr int kOriSparsePoint = 100;
	};


	class SparseGrid{
	public:
		SparseGrid(const double* time_stamp, const int N, const int variable_count,
		           const std::vector<int>* variable_ind = nullptr);

        inline const std::vector<double>& GetAlpha() const {return alpha_;}
        inline double GetAlphaAt(const int ind) const{
            CHECK_LT(ind, alpha_.size());
            return alpha_[ind];
        }

        inline const std::vector<int>& GetVariableInd() const {return variable_ind_; }
        inline const int GetVariableIndAt(const int ind) const{
            CHECK_LT(ind, variable_ind_.size());
            return variable_ind_[ind];
        }

        inline const std::vector<int>& GetInverseInd() const {return inverse_ind_; }
        inline const int GetInverseIndAt(const int ind) const{
            CHECK_LT(ind, inverse_ind_.size());
            return inverse_ind_[ind];
        }

		inline const int GetTotalCount() const{
			return kTotalCount;
		}
        template <typename T>
        void correct_linacce_bias(Eigen::Matrix<T, 3, 1>* data, const T* bx, const T* by, const T* bz,
		const Eigen::Matrix<T, 3, 1> bias_global = Eigen::Matrix<T, 3, 1>::Zero()) const {
            for (int i = 0; i < kTotalCount; ++i) {
                const int vid = inverse_ind_[i];
                data[i] += alpha_[i] * Eigen::Matrix<T, 3, 1>(bx[vid], by[vid], bz[vid]);
                if (vid > 0) {
                    data[i] += (1.0 - alpha_[i]) * Eigen::Matrix<T, 3, 1>(bx[vid - 1], by[vid - 1], bz[vid - 1]);
                }
            }
        }

    private:
        const int kTotalCount;
        const int kVariableCount;

        std::vector<double> alpha_;
        std::vector<int> inverse_ind_;
        std::vector<int> variable_ind_;
    };

	template <int KVARIABLE, int KCONSTRAINT>
	struct LocalSpeedFunctor{
	public:
		LocalSpeedFunctor(const double* time_stamp, const int N,
		                  const Eigen::Vector3d* linacce,
		                  const Eigen::Quaterniond* orientation,
		                  const int* constraint_ind,
		                  const Eigen::Vector3d* local_speed,
		                  const Eigen::Vector3d init_speed,
						  const double weight_ls = 1.0, const double weight_vs = 1.0):
				time_stamp_(time_stamp), linacce_(linacce), orientation_(orientation),
				constraint_ind_(constraint_ind), local_speed_(local_speed), init_speed_(init_speed),
				weight_ls_(std::sqrt(weight_ls)), weight_vs_(std::sqrt(weight_vs)){
			grid_.reset(new SparseGrid(time_stamp, N, KVARIABLE));
		}

		inline const SparseGrid* GetLinacceGrid() const{
			return grid_.get();
		}
#if false
		bool operator() (const double* const bx, const double* const by, const double* const bz, double* residual) const{
				for (int i = 0; i < KCONSTRAINT * 3; ++i) {
					residual[i] = 0.0;
				}

				std::vector<Eigen::Matrix <double, 3, 1> > directed_acce(linacce_.size());
				std::vector<Eigen::Matrix <double, 3, 1> > speed((size_t) Config::kTotalCount);
				speed[0] = init_speed_ + Eigen::Matrix <double, 3, 1>(std::numeric_limits<double>::epsilon(),
				                                                      std::numeric_limits<double>::epsilon(),
				                                                      std::numeric_limits<double>::epsilon());

				directed_acce[0] = (orientation_[0] * linacce_[0]);
				for (int i = 0; i < Config::kTotalCount; ++i) {
					const int inv_ind = grid_->GetInverseIndAt(i);
					Eigen::Matrix<double, 3, 1> corrected_acce =
							linacce_[i] + grid_->GetAlphaAt(i) * Eigen::Matrix<double, 3, 1>(bx[inv_ind], by[inv_ind], bz[inv_ind]);
					if (inv_ind > 0) {
						corrected_acce = corrected_acce + (1.0 - grid_->GetAlphaAt(i)) *
						                 Eigen::Matrix<double, 3, 1>(bx[inv_ind - 1], by[inv_ind - 1], bz[inv_ind - 1]);
					}
					if (i > 0) {
						directed_acce[i] = orientation_[i] * corrected_acce;
						speed[i] = speed[i - 1] + dt_[i-1] * directed_acce[i - 1];
					}
				}

				for (int cid = 0; cid < constraint_ind_.size(); ++cid) {
					const int ind = constraint_ind_[cid];
					Eigen::Vector3d ls = orientation_[ind].conjugate() * speed[ind];
					residual[cid] = weight_ls_ * (ls[0] - local_speed_[cid][0]);
					residual[cid + KCONSTRAINT] = weight_ls_ * (ls[1] - local_speed_[cid][1]);
					residual[cid + 2 * KCONSTRAINT] = weight_ls_ * (ls[2] - local_speed_[cid][2]);
				}
				return true;
			}
#else
		template <typename T>
		bool operator() (const T* const bx, const T* const by, const T* const bz,
						 T* residual) const{

			std::vector<Eigen::Matrix <T, 3, 1> > directed_acce((size_t) grid_->GetTotalCount());
			std::vector<Eigen::Matrix <T, 3, 1> > speed((size_t) grid_->GetTotalCount());

			speed[0] = init_speed_ + Eigen::Matrix <T, 3, 1>((T)std::numeric_limits<double>::epsilon(),
			                                                 (T)std::numeric_limits<double>::epsilon(),
			                                                 (T)std::numeric_limits<double>::epsilon());
			directed_acce[0] = (orientation_[0] * linacce_[0]).template cast<T>();
			for (int i = 0; i < grid_->GetTotalCount(); ++i) {
				const int inv_ind = grid_->GetInverseIndAt(i);
				Eigen::Matrix<T, 3, 1> corrected_acce =
						linacce_[i] + grid_->GetAlphaAt(i) * Eigen::Matrix<T, 3, 1>(bx[inv_ind], by[inv_ind], bz[inv_ind]);

				if (inv_ind > 0) {
					corrected_acce = corrected_acce + (1.0 - grid_->GetAlphaAt(i)) *
					                                  Eigen::Matrix<T, 3, 1>(bx[inv_ind - 1], by[inv_ind - 1], bz[inv_ind - 1]);
				}
				if (i > 0) {
					directed_acce[i] = orientation_[i].template cast<T>() * corrected_acce;
					speed[i] = speed[i - 1] + directed_acce[i - 1] * (time_stamp_[i] - time_stamp_[i - 1]);
				}
			}

			for (int cid = 0; cid < KCONSTRAINT; ++cid) {
				const int ind = constraint_ind_[cid];
				Eigen::Matrix<T, 3, 1> ls = orientation_[ind].conjugate().template cast<T>() * speed[ind];
				residual[cid] = weight_ls_ * (ls[0] - (T)local_speed_[cid][0]);
				residual[cid + KCONSTRAINT] = weight_vs_ * speed[ind][2];
				residual[cid + 2 * KCONSTRAINT] = weight_ls_ * (ls[2] - (T)local_speed_[cid][2]);

			}
			return true;
		}
#endif

	private:
		std::shared_ptr<SparseGrid> grid_;
		const double* time_stamp_;
		const Eigen::Vector3d* linacce_;
		const Eigen::Quaterniond* orientation_;
		const int* constraint_ind_;
		const Eigen::Vector3d* local_speed_;

		const Eigen::Vector3d init_speed_;
		const double weight_ls_;
		const double weight_vs_;
	};

	template<int KVARIABLE>
	struct WeightDecay {
	public:
		WeightDecay(const double weight) : weight_(std::sqrt(weight)) {}

		template<typename T>
		bool operator()(const T *const x, T *residual) const {
			for (int i = 0; i < KVARIABLE; ++i) {
				residual[i] = weight_ * x[i];
			}
			return true;
		}
	private:
		const double weight_;
	};

//    template <int KVAR, int KCONSTRAINT>
//    struct OrientationFunctor{
//    public:
//        OrientationFunctor(const std::vector<double> &time_stamp,
//                           const std::vector<Eigen::Vector3d> &linacce,
//                           const std::vector<Eigen::Matrix3d> &orientation,
//                           const std::vector<int>& constraint_ind,
//                           const std::vector<Eigen::Vector3d> &local_speed,
//                           const Eigen::Vector3d init_speed,
//                           const double weight_ls = 1.0, const double weight_vs = 1.0):
//                time_stamp_(time_stamp), linacce_(linacce), orientation_(orientation),
//                constraint_ind_(constraint_ind), local_speed_(local_speed), init_speed_(init_speed), drift_axis_(0, 0, 1),
//                weight_ls_(std::sqrt(weight_ls)), weight_vs_(std::sqrt(weight_vs)){
//            CHECK_EQ(constraint_ind_.size(), KCONSTRAINT);
//            CHECK_EQ(local_speed_.size(), KCONSTRAINT);
//            grid_orientation_.reset(new SparseGrid(time_stamp.data(), (int) time_stamp.size(), KVAR));
//
//        }
//
//        const SparseGrid* GetOriGrid() const {return grid_orientation_.get(); }
//#if false
//        bool operator()(const double* const rz, double* residual) const{
//            std::vector<Eigen::Matrix<double, 3, 3> > corrected_orientation((size_t) grid_orientation_->GetTotalCount());
//            std::vector<Eigen::Matrix<double, 3, 1> > speed((size_t) grid_orientation_->GetTotalCount());
//            speed[0] = init_speed_;
//
//            const std::vector<double>& alpha = grid_orientation_->GetAlpha();
//            const std::vector<int>& inv_ind = grid_orientation_->GetInverseInd();
//            for(auto i=0; i<grid_orientation_->GetTotalCount(); ++i){
//                double rad_z = alpha[i] * rz[inv_ind[i]];
//                if(inv_ind[i] > 0){
//                    rad_z += (1.0 - alpha[i]) * rz[inv_ind[i] - 1];
//                }
//                Eigen::Matrix<double, 3, 3> rotor;
//                const double sv = ceres::sin(rad_z);
//                const double cv = ceres::cos(rad_z);
//                rotor << cv, -1 * sv, 0, sv, cv, 0, 0, 0, 1;
//                corrected_orientation[i] = rotor * orientation_[i];
//                if(i > 0){
//                    Eigen::Matrix<double, 3, 1> acce = corrected_orientation[i] * linacce_[i];
//                    speed[i] = speed[i-1] + acce * (time_stamp_[i] - time_stamp_[i-1]);
//                }
//            }
//
//            for (int cid = 0; cid < constraint_ind_.size(); ++cid) {
//                const int ind = constraint_ind_[cid];
//                Eigen::Matrix<double, 3, 1> ls = corrected_orientation[ind].transpose() * speed[ind];
//                residual[cid] = weight_ls_ * (ls[0] - local_speed_[cid][0]);
//                residual[cid + KCONSTRAINT] = weight_vs_ * speed[ind][2];
//                residual[cid + 2 * KCONSTRAINT] = weight_ls_ * (ls[2] - local_speed_[cid][2]);
//            }
//            return true;
//        }
//#else
//        template <typename T>
//        bool operator()(const T* const rz, T* residual) const{
//            std::vector<Eigen::Matrix<T, 3, 3> > corrected_orientation((size_t) grid_orientation_->GetTotalCount());
//            std::vector<Eigen::Matrix<T, 3, 1> > speed((size_t) grid_orientation_->GetTotalCount());
//            speed[0] = init_speed_.template cast<T>();
//
//            const std::vector<double>& alpha = grid_orientation_->GetAlpha();
//            const std::vector<int>& inv_ind = grid_orientation_->GetInverseInd();
//            for(auto i=0; i<grid_orientation_->GetTotalCount(); ++i){
//                T rad_z = alpha[i] * rz[inv_ind[i]];
//                if(inv_ind[i] > 0){
//                    rad_z += (1.0 - alpha[i]) * rz[inv_ind[i] - 1];
//                }
//                Eigen::Matrix<T, 3, 3> rotor;
//                const T sv = ceres::sin(rad_z);
//                const T cv = ceres::cos(rad_z);
//                rotor << cv, -1.0 * sv, 0, sv, cv, 0, 0, 0, 1.0;
//                corrected_orientation[i] = rotor * orientation_[i];
//                if(i > 0){
//                    Eigen::Matrix<T, 3, 1> acce = corrected_orientation[i] * linacce_[i];
//                    speed[i] = speed[i-1] + acce * (time_stamp_[i] - time_stamp_[i-1]);
//                }
//            }
//
//            for (int cid = 0; cid < constraint_ind_.size(); ++cid) {
//                const int ind = constraint_ind_[cid];
//                Eigen::Matrix<T, 3, 1> ls = corrected_orientation[ind].transpose() * speed[ind];
//                residual[cid] = weight_ls_ * (ls[0] - local_speed_[cid][0]);
//                residual[cid + KCONSTRAINT] = weight_vs_ * speed[ind][2];
//                residual[cid + 2 * KCONSTRAINT] = weight_ls_ * (ls[2] - local_speed_[cid][2]);
//            }
//            return true;
//        }
//#endif
//    private:
//        std::shared_ptr<SparseGrid> grid_orientation_;
//        const std::vector<double>& time_stamp_;
//        const std::vector<Eigen::Vector3d>& linacce_;
//        std::vector<Eigen::Matrix3d> orientation_;
//        const std::vector<int>& constraint_ind_;
//        const std::vector<Eigen::Vector3d>& local_speed_;
//
//        const Eigen::Vector3d drift_axis_;
//
//        const Eigen::Vector3d init_speed_;
//
//        const double weight_ls_;
//        const double weight_vs_;
//    };
//
//    template <int KVAR>
//    struct FirstOrderSmoothFunctor{
//        explicit FirstOrderSmoothFunctor(const double weight): weight_(std::sqrt(weight)){}
//
//        template <typename T>
//        bool operator()(const T* const x, T* residual) const{
////            residual[0] = weight_ * x[0];
////            for(int i=1; i<KVAR; ++i){
////                residual[i] = weight_ * (x[i] - x[i-1]);
////            }
//
//			for(int i=0; i<KVAR; ++i){
//				residual[i] = weight_ * x[i];
//			}
//
//            return true;
//        }
//    private:
//        const double weight_;
//    };
}//IMUProject
#endif //PROJECT_IMU_OPTIMIZATION_H
