//
// Created by yanhang on 10/9/17.
//

#include "geometry.h"

#include <vector>
#include <Eigen/Eigen>

#include "googletest/include/gtest/gtest.h"

namespace IMUProject{

TEST(Geometry, EstimateTransformation){
  const int kPoints = 100;

  std::vector<Eigen::Vector3d> source(kPoints);
  std::vector<Eigen::Vector3d> target(kPoints);
}

}  // namespace IMUProject