//
// Created by yanhang on 2/6/17.
//

#ifndef PROJECT_IMU_OPTIMIZATION_H
#define PROJECT_IMU_OPTIMIZATION_H

#include <opencv2/opencv.hpp>
#include <Eigen/Eigen>
#include <glog/logging.h>
#include <ceres/ceres.h>

namespace IMUProject {

	struct Config {
		static constexpr int kTotalCount = 15200;
		static constexpr int kConstriantPoints = 1520;
		static constexpr int kSparsePoints = 760;
		static constexpr int kSparseInterval = 20;
	};


    class SparseGridInterpolator{
    public:
        SparseGridInterpolator(const std::vector<double>& time_stamp, const int variable_count,
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

        template <typename T>
        void correct_bias(Eigen::Matrix<T, 3, 1>* data, const T* bx, const T* by, const T* bz) const {
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

	struct SizedSharedSpeedFunctor {
	public:
		SizedSharedSpeedFunctor(const std::vector<double> &time_stamp, const std::vector<Eigen::Vector3d> &linacce,
								const std::vector<Eigen::Quaterniond> &orientation,
								const std::vector<int> &constraint_ind,
								const std::vector<double> &target_speed_mag,
								const std::vector<double> &target_vspeed,
								const Eigen::Vector3d &init_speed,
								const double weight_sm = 1.0,
								const double weight_vs = 1.0);

#if false
		bool operator()(const double *const bx, const double *const by, const double *const bz, double *residual) const {
			for (int i = 0; i < Config::kConstriantPoints * 2; ++i) {
				residual[i] = 0.0;
			}

			std::vector<Eigen::Matrix <double, 3, 1> > directed_acce((size_t) Config::kTotalCount);
			std::vector<Eigen::Matrix <double, 3, 1> > speed((size_t) Config::kTotalCount);
			speed[0] = init_speed_ + Eigen::Matrix <double, 3, 1>(std::numeric_limits<double>::epsilon(),
			                                                      std::numeric_limits<double>::epsilon(),
			                                                      std::numeric_limits<double>::epsilon());

			directed_acce[0] = (rotations_[0] * linacce_[0]);
			//std::cout << rotations_[0] << std::endl;

// #pragma omp parallel for
			for (int i = 0; i < Config::kTotalCount; ++i) {
				const int inv_ind = inverse_ind_[i];
				Eigen::Matrix<double, 3, 1> corrected_acce =
						linacce_[i] + Eigen::Matrix<double, 3, 1>(alpha_[i] * bx[inv_ind],
						                                          alpha_[i] * by[inv_ind],
						                                          alpha_[i] * bz[inv_ind]);
				if (inv_ind > 0) {
					corrected_acce = corrected_acce +
					                 Eigen::Matrix<double, 3, 1>((1.0 - alpha_[i]) * bx[inv_ind - 1],
					                                             (1.0 - alpha_[i]) * by[inv_ind - 1],
					                                             (1.0 - alpha_[i]) * bz[inv_ind - 1]);
				}
				if (i > 0) {
					directed_acce[i] = rotations_[i] * corrected_acce;
					speed[i] = speed[i - 1] + (directed_acce[i - 1]) * dt_[i - 1];
//					printf("(%f,%f,%f) + (%f,%f,%f) * %f = (%f,%f,%f)\n",
//					       speed[i - 1][0], speed[i - 1][1], speed[i - 1][2],
//					       directed_acce[i - 1][0], directed_acce[i - 1][1], directed_acce[i - 1][2],
//					       dt_[i - 1], speed[i][0], speed[i][1], speed[i][2]);
				}
			}

			for (int cid = 0; cid < constraint_ind_.size(); ++cid) {
				const int ind = constraint_ind_[cid];
				residual[cid] = weight_sm_ * (speed[ind].norm() - target_speed_mag_[cid]);
				// printf("%d\t (%f, %f, %f), target_vspeed: %.9f\n", ind, speed[ind][0], speed[ind][1], speed[ind][2], target_vspeed_[cid]);
				residual[cid + Config::kConstriantPoints] = weight_vs_ * (speed[ind][2] - target_vspeed_[cid]);
			}
			return true;
		}
#else

		template<typename T>
		bool operator()(const T *const bx, const T *const by, const T *const bz, T *residual) const {
			for (int i = 0; i < Config::kConstriantPoints * 2; ++i) {
				residual[i] = (T) 0.0;
			}

			std::vector<Eigen::Matrix<T, 3, 1> > directed_acce((size_t) Config::kTotalCount);
			std::vector<Eigen::Matrix<T, 3, 1> > speed((size_t) Config::kTotalCount);
			speed[0] = init_speed_ + Eigen::Matrix<T, 3, 1>((T) std::numeric_limits<double>::epsilon(),
			                                                (T) std::numeric_limits<double>::epsilon(),
			                                                (T) std::numeric_limits<double>::epsilon());

			directed_acce[0] = (rotations_[0] * linacce_[0]).cast<T>();
// #pragma omp parallel for
			for (int i = 0; i < Config::kTotalCount; ++i) {
				const int inv_ind = inverse_ind_[i];
				Eigen::Matrix<T, 3, 1> corrected_acce =
						linacce_[i] + Eigen::Matrix<T, 3, 1>(alpha_[i] * bx[inv_ind],
						                                     alpha_[i] * by[inv_ind],
						                                     alpha_[i] * bz[inv_ind]);
				if (inv_ind > 0) {
					corrected_acce = corrected_acce +
					                 Eigen::Matrix<T, 3, 1>((1.0 - alpha_[i]) * bx[inv_ind - 1],
					                                        (1.0 - alpha_[i]) * by[inv_ind - 1],
					                                        (1.0 - alpha_[i]) * bz[inv_ind - 1]);
				}
				if (i > 0) {
					directed_acce[i] = rotations_[i] * corrected_acce;
					speed[i] = speed[i - 1] + (directed_acce[i - 1]) * dt_[i - 1];
				}
			}

			for (int cid = 0; cid < constraint_ind_.size(); ++cid) {
				const int ind = constraint_ind_[cid];
				residual[cid] = weight_sm_ * (speed[ind].norm() - target_speed_mag_[cid]);
				residual[cid + Config::kConstriantPoints] = weight_vs_ * (speed[ind][2] - target_vspeed_[cid]);
			}
			return true;
		}
#endif
		std::vector<Eigen::Vector3d> correcte_acceleration(const std::vector<Eigen::Vector3d> &input,
														   const std::vector<double> &bx,
														   const std::vector<double> &by,
														   const std::vector<double> &bz);

	private:
		const std::vector<Eigen::Vector3d> &linacce_;
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


	struct WeightDecay {
	public:
		WeightDecay(const double weight) : weight_(std::sqrt(weight)) {}

		template<typename T>
		bool operator()(const T *const x, T *residual) const {
			for (int i = 0; i < Config::kSparsePoints; ++i) {
				residual[i] = weight_ * x[i];
			}
			return true;
		}

//        bool operator () (const double* const x, double* residual) const{
//            for(int i=0; i<Config::kSparsePoints; ++i){
//                residual[i] = weight_ * x[i];
//            }
//            return true;
//        }

	private:
		const double weight_;
	};


}//IMUProject
#endif //PROJECT_IMU_OPTIMIZATION_H
