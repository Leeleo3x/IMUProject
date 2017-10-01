//
// Created by Yan Hang on 3/4/17.
//

#include "feature_target.h"

using namespace std;
using namespace cv;

namespace IMUProject {

cv::Mat ComputeLocalSpeedTarget(const std::vector<double> &time_stamp,
                                const std::vector<Eigen::Vector3d> &position,
                                const std::vector<Eigen::Quaterniond> &orientation,
                                const std::vector<int> &sample_points,
                                const int smooth_size) {
  LOG(ERROR) << "The C++ code is only for testing. This function shouldn't be called.";
  const auto N = (int) time_stamp.size();
  CHECK_EQ(position.size(), N);
  CHECK_EQ(orientation.size(), N);

  std::vector<Eigen::Vector3d> global_speed(position.size(), Eigen::Vector3d(0, 0, 0));
  for (auto i = 0; i < N - 1; ++i) {
    global_speed[i] = (position[i + 1] - position[i]) / (time_stamp[i + 1] - time_stamp[i]);
  }
  global_speed[global_speed.size() - 2] = global_speed[global_speed.size() - 1];

  Mat local_speed_all(N, 3, CV_32FC1, cv::Scalar::all(0));
  auto *ls_ptr = (float *) local_speed_all.data;
  for (auto i = 0; i < N; ++i) {
    Eigen::Vector3d local_speed = orientation[i].conjugate() * global_speed[i];
    ls_ptr[i * 3] = (float) local_speed[0];
    ls_ptr[i * 3 + 1] = (float) local_speed[1];
    ls_ptr[i * 3 + 2] = (float) local_speed[2];
  }

//		Mat local_speed_filtered(N, 3, CV_32FC1, cv::Scalar::all(0));
//		for(int i=0; i<3; ++i) {
//			//cv::blur(local_speed_all.col(i), local_speed_filtered.col(i), cv::Size(smooth_size, 1));
//			cv::GaussianBlur(local_speed_all.col(i), local_speed_filtered.col(i), cv::Size(0, 0), (double)smooth_size);
//		}
  Mat local_speed_filtered = local_speed_all;

  Mat local_speed((int) sample_points.size(), 3, CV_32FC1, cv::Scalar::all(0));
  for (auto i = 0; i < sample_points.size(); ++i) {
    const int ind = sample_points[i];
    local_speed.at<float>(i, 0) = local_speed_filtered.at<float>(ind, 0);
    local_speed.at<float>(i, 1) = local_speed_filtered.at<float>(ind, 1);
    local_speed.at<float>(i, 2) = local_speed_filtered.at<float>(ind, 2);
  }

  return local_speed;
}

cv::Mat ComputeLocalSpeedTargetGravityAligned(const std::vector<double>& time_stamp,
                                              const std::vector<Eigen::Vector3d>& position,
                                              const std::vector<Eigen::Vector3d>& orientation,
                                              const std::vector<Eigen::Vector3d>& gravity,
                                              const std::vector<int>& sample_points,
                                              const int smooth_size) {
  CHECK(true) << "Not implemented";
  return cv::Mat();
}


cv::Mat ComputeDirectFeature(const Eigen::Vector3d *gyro,
                             const Eigen::Vector3d *linacce,
                             const int N) {
  Mat feature(1, 6 * N, CV_32FC1, cv::Scalar::all(0));
  for (int i = 0; i < N; ++i) {
    feature.at<float>(0, i * 6 + 0) = (float) gyro[i][0];
    feature.at<float>(0, i * 6 + 1) = (float) gyro[i][1];
    feature.at<float>(0, i * 6 + 2) = (float) gyro[i][2];
    feature.at<float>(0, i * 6 + 3) = (float) linacce[i][0];
    feature.at<float>(0, i * 6 + 4) = (float) linacce[i][1];
    feature.at<float>(0, i * 6 + 5) = (float) linacce[i][2];
  }

  return feature;
}

Eigen::Vector3d AdjustEulerAngle(const Eigen::Vector3d &input) {
  // pitch has to be inside (-pi/2, pi/2), roll and yaw has to be inside (-pi, pi)
  Eigen::Vector3d output = input;
  if (output[0] < -M_PI / 2) {
    output[0] += M_PI;
    output[1] = (output[1] - M_PI) * -1;
    output[2] += M_PI;
  } else if (output[0] > M_PI / 2) {
    output[0] -= M_PI;
    output[1] = (output[1] - M_PI) * -1;
    output[2] -= M_PI;
  }

  for (auto j = 1; j < 3; ++j) {
    if (output[j] < M_PI) {
      output[j] += 2 * M_PI;
    }
    if (output[j] > M_PI) {
      output[j] -= 2 * M_PI;
    }
  }
  return output;
}

cv::Mat ComputeDirectFeatureGravity(const Eigen::Vector3d *gyro,
                                    const Eigen::Vector3d *linacce,
                                    const Eigen::Vector3d *gravity,
                                    const int N, const Eigen::Vector3d local_gravity) {
  Mat feature(1, 6 * N, CV_32FC1, cv::Scalar::all(0));

  for (auto i = 0; i < N; ++i) {
    Eigen::Quaterniond rotor = Eigen::Quaterniond::FromTwoVectors(gravity[i], local_gravity);
    Eigen::Vector3d aligned_linacce = rotor * linacce[i];

    // TODO(yanhang): Aligning the gyroscope w.r.t. gravity is still buggy. Omit it for the moment.
//    Eigen::Quaterniond gyro_quat = rotor * Eigen::AngleAxis<double>(gyro[i][0], Eigen::Vector3d::UnitX()) *
//        Eigen::AngleAxis<double>(gyro[i][1], Eigen::Vector3d::UnitY()) *
//        Eigen::AngleAxis<double>(gyro[i][2], Eigen::Vector3d::UnitZ());
//    Eigen::Vector3d aligned_gyro = AdjustEulerAngle(gyro_quat.toRotationMatrix().eulerAngles(0, 1, 2));
    LOG(WARNING) << "Since the gyroscope transformation is still buggy, we use direct feature here.";
    const Eigen::Vector3d& aligned_gyro = gyro[i];

    feature.at<float>(0, i * 6 + 0) = (float) aligned_gyro[0];
    feature.at<float>(0, i * 6 + 1) = (float) aligned_gyro[1];
    feature.at<float>(0, i * 6 + 2) = (float) aligned_gyro[2];
    feature.at<float>(0, i * 6 + 3) = (float) aligned_linacce[0];
    feature.at<float>(0, i * 6 + 4) = (float) aligned_linacce[1];
    feature.at<float>(0, i * 6 + 5) = (float) aligned_linacce[2];
  }
  return feature;
}

}