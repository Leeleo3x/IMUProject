//
// Created by Yan Hang on 1/11/17.
//

#ifndef PROJECT_TEMPORAL_FUNCTION_H
#define PROJECT_TEMPORAL_FUNCTION_H

#include <vector>
#include <glog/logging.h>

/***
 * Base class for temporal function
 */

namespace IMUProject {
    template<typename T>
    class TemporalFunction {
    public:
        enum InterpolateMethod{
            Linear,
            BSpline
        };

        TemporalFunction(const std::vector<double>& times, const std::vector<T>& values):
                times_(times), values_(values){
            // Perform sanity checks
            CHECK_EQ(times_.size(), values_.size());
            CHECK_GT(times_.size(), 5);
            for(int i=0; i<times_.size()-1; ++i){
                CHECK_LE(times_[i], times_[i+1]);
            }
            dt_ = (times_.back() - times_[0]) / static_cast<double>(times_.size() - 1);
            values_.push_back(values_.back());
            values_.push_back(values_.back());
            values_.push_back(values_[0]);
            values_.push_back(values_[0]);
            valid_from_ = times_[2];
            valid_to_ = times_[times_.size() - 2];
        }

        virtual T GetValueAtTime(const double t, const InterpolateMethod method = BSpline) const = 0;
        virtual void GetValueAtTimes(const std::vector<double> input_t, std::vector<T> &values,
                                     const InterpolateMethod method = BSpline) const = 0;

        virtual T InterpolateAtInterval(const int index, const double ratio,
                                        const InterpolateMethod method = BSpline) = 0;

        double BFunction(const int i, const int k, const double t){
            if (k == 1) {
                if (t >= i && t < i + 1) {
                    return 1;
                } else {
                    return 0;
                }
            }
            return ((t - i / (k - 1)) * BFunction(i, k - 1, t) + ((i + k - t) / (k - 1)) * BFunction(i + 1, k - 1, t));
        }
        inline double BPrimeFunction(const int i, const int k, const double t) {
            CHECK_GT(times_.size(), 1);
            return (1 / dt_) * (BFunction(i, k - 1, t) - BFunction(i + 1, k - 1, t));
        }
        double TildeBFunction(const int i, const int k, const double t){
            if(t >= i+k-1){
                return 1;
            }
            if(t <= i){
                return 0;
            }
            double sum = 0.0;
            for(int j=i; j<i+k+1; ++j){
                sum += BFunction(j,k,t);
            }
            return sum;
        }

        inline double GetValidFrom() const{
            return valid_from_;
        }

        inline double GetValidTo() const{
            return valid_to_;
        }
    protected:
        std::vector<double> times_;
        std::vector<T> values_;
        double dt_;
        double valid_from_;
        double valid_to_;
    };

} //namespace IMUProject
#endif //PROJECT_TEMPORAL_FUNCTION_H
