//
// Created by Yan Hang on 1/11/17.
//

#ifndef PROJECT_ROTATIONFUNCTION_H
#define PROJECT_ROTATIONFUNCTION_H

#include <algorithm/temporal_function.h>

#include <limits>
#include <vector>
#include <glog/logging.h>
#include <Eigen/Eigen>


namespace IMUProject {

    inline Eigen::Quaternion<double> QuaternionLog(const Eigen::Quaternion<double>& q) {
        Eigen::Quaternion<double> result(0,0,0,0);
        const double qn = std::sqrt(q.x() * q.x() + q.y() * q.y() + q.z() * q.z());
        if (qn > std::numeric_limits<double>::epsilon()) {
            const double m = std::sqrt(q.w() * q.w() + qn * qn);
            const double a = std::acos(q.w() / m);
            const double s = a / qn;
            result.w() = std::log(std::sqrt(q.w() * q.w() + qn * qn));
            result.x() = q.x() * s;
            result.y() = q.y() * s;
            result.z() = q.z() * s;
        }
        return result;
    }

    inline Eigen::Quaternion<double> QuaternionExp(const Eigen::Quaternion<double>& q){
        Eigen::Quaternion<double> result(std::exp(q.w()), 0, 0, 0);
        const double qn = std::sqrt(q.x() * q.x() + q.y() * q.y() + q.z() * q.z());
        if(qn > std::numeric_limits<double>::epsilon()){
            const double s = std::sin(qn) / qn;
            const double e = std::exp(q.w());
            result.w() = std::cos(qn) * e;
            result.x() = q.x() * s * e;
            result.y() = q.y() * s * e;
            result.z() = q.z() * s * e;
        }
        return result;
    }

    class RotationFunction : public TemporalFunction<Eigen::Quaternion<double> > {
    public:
        RotationFunction(const std::vector<double>& times, const std::vector<Quaternion_T>& quats);
        virtual Quaternion_T GetValueAtTime(const double t, const InterpolateMethod method = BSpline) const;
        virtual void GetValueAtTimes(const std::vector<double> input_t, std::vector<Quaternion_T > &values,
                                     const InterpolateMethod method = BSpline) const;

        virtual Quaternion_T InterpolateAtInterval(const int index, const double ratio,
                                                   const InterpolateMethod method = BSpline);
    private:
        using Quaternion_T = Eigen::Quaternion<double>;
        std::vector<Quaternion_T> w_;
    };

} //namespace IMUProject
#endif //PROJECT_ROTATIONFUNCTION_H
