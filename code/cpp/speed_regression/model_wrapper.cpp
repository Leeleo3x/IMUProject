//
// Created by yanhang on 9/30/17.
//

#include <fstream>
#include "speed_regression/model_wrapper.h"

namespace IMUProject{

std::istream& operator >> (std::istream& stream, SVRCascadeOption& option){

}

static std::string SVRCascadeOption::kVersionTag_ = "v1.0";

bool SVRCascade::LoadFromFile(const std::string &path) {
  std::ifstream option_in(path + "/option.txt");
  if (!option_in.is_open()){
    LOG(ERROR) << "Can not open option file: " << path + "/option.txt";
    return false;
  }

  SVRCascadeOption option;
  option_in >> option;
}

void SVRCascade::Predict(const cv::Mat &feature, cv::Mat *predicted) const {
  cv::Mat label;
  Predict(feature, &label, predicted);
}

void SVRCascade::Predict(const cv::Mat &feature, cv::Mat *label, cv::Mat *response) const {

}

bool CVModel::LoadFromFile(const std::string &path) {

}

void CVModel::Predict(const cv::Mat &feature, cv::Mat *predicted) const {

}

}  // namespace IMUProject