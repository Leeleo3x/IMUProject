//
// Created by yanhang on 10/9/17.
//

#include "algorithm/geometry.h"

#include <iostream>
#include <vector>
#include <Eigen/Eigen>

#include "googletest/include/gtest/gtest.h"

namespace IMUProject{

TEST(Geometry, EstimateTransformation){
  const int kPoints = 100;

  // Get a random rigid transformation;
  Eigen::Vector3d axis = Eigen::Vector3d::Random();
  axis.normalize();
  const Eigen::AngleAxisd rotor(0.25 * M_PI, axis);
  const Eigen::Vector3d translation = Eigen::Vector3d::Random();
  std::vector<Eigen::Vector3d> source(kPoints);
  std::vector<Eigen::Vector3d> target(kPoints);
  for (int i=0; i<source.size(); ++i){
    source[i] = Eigen::Vector3d::Random();
    target[i] = rotor * source[i] + translation;
  }

  Eigen::Matrix4d estimated_transformation;
  Eigen::Matrix3d estimated_rotation;
  Eigen::Vector3d estimated_translation;

  EstimateTransformation<3>(source, target, &estimated_transformation, &estimated_rotation, &estimated_translation);

  constexpr double kTol = 1e-05;
  for (int i=0; i<source.size(); ++i){
    auto source_transformed = estimated_rotation * source[i] + estimated_translation;
    auto diff = (source_transformed - target[i]).norm();
    EXPECT_NEAR(diff, 0, kTol);
  }
}

}  // namespace IMUProject