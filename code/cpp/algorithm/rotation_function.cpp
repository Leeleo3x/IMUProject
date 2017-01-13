//
// Created by Yan Hang on 1/11/17.
//

#include "rotation_function.h"
#include <limits>

namespace IMUProject{

    RotationFunction::RotationFunction(const std::vector<double>& times, const std::vector<Quaternion_T>& quats):
            TemporalFunction<Quaternion_T >(times, quats ){
        for(int i=0; i<values_.size(); ++i){
            const Quaternion_T & q1 = values_[(i-1) % (int)values_.size()];
            const Quaternion_T & q2 = values_[i];
            w_[i] = QuaternionLog(q1.inverse() * q2);
        }
    }

    Eigen::Quaternion<double> RotationFunction::InterpolateAtInterval(const int index, const double ratio,
                                                                      const InterpolateMethod method) {
        CHECK_GE(index, 0);
        CHECK_LT(index, times_.size() - 1);

        Quaternion_T result;
        const Quaternion_T &q1 = values_[index];
        const Quaternion_T &q2 = values_[index + 1];

        if (method == InterpolateMethod::Linear) {
            result = q1.slerp(ratio, q2);
        }else if(method == InterpolateMethod::BSpline){
            CHECK_GT(times_.size(), 3) << "Can not use Spline interpolation with array size less than 3";

        }else{
            CHECK(true) << "Unrecognized interpolation method";
        }
        return result;
    }

    Eigen::Quaternion<double> RotationFunction::GetValueAtTime(const double t, const InterpolateMethod method) const {
        int index = 0;
        while (index < values_.size() - 1) {
            if (t >= times_[index] && t <= times_[index + 1]) {
                break;
            }
            index++;
        }
        CHECK_LT(index, times_.size() - 1) << "Invalid input";

        if (std::fabs(t - times_[index]) < std::numeric_limits<double>::epsilon()) {
            return values_[index];
        } else if (std::fabs(t - times_[index + 1]) < std::numeric_limits<double>::epsilon()) {
            return values_[index + 1];
        }

        const double ratio = (t - times_[index]) / (times_[index+1] - times_[index]);
        return InterpolateAtInterval(index, ratio, method);
    }


    void RotationFunction::GetValueAtTimes(const std::vector<double> input_t, std::vector<Quaternion_T> &values,
                                           const InterpolateMethod method) const {

    }

}//namespace IMUProject