//
// Created by yanhang on 9/30/17.
//

#ifndef CODE_SPEED_REGRESSION_MODEL_WRAPPER_H
#define CODE_SPEED_REGRESSION_MODEL_WRAPPER_H

#include <string>
#include <glog/logging.h>
#include <opencv2/opencv.hpp>

// This class provide unified predicting interface for regression model. The training of these models are done through
// python codes. See code/python/speed_regression for details.
namespace IMUProject{

class ModelWrapper{
 public:
  virtual bool LoadFromFile(const std::string& path) = 0;
  virtual void Predict(const cv::Mat& feature, cv::Mat* predicted) const = 0;
};

struct SVRCascadeOption{
  int num_classes = 0;
  int num_channels = 0;
  const static std::string kVersionTag_;
};

std::istream& operator >> (std::istream& stream, SVRCascadeOption& option);

// This class defines the cascading model.
class SVRCascade: public ModelWrapper{
 public:
  explicit SVRCascade(const std::string& path){
    CHECK(LoadFromFile(path)) << "Can not load SVRCascade model from " << path;
  }
  bool LoadFromFile(const std::string& path) override;
  void Predict(const cv::Mat& feature, cv::Mat* predicted) const override;
  void Predict(const cv::Mat& feature, cv::Mat* label, cv::Mat* response) const;

  inline int GetNumClasses() const{
    return option_.num_classes;
  }

  inline int GetNumChannels() const{
    return option_.num_channels;
  }

  SVRCascade(const SVRCascade& model) = delete;
  bool operator = (const SVRCascade& model) = delete;
 private:
  SVRCascadeOption option_;

  std::vector<cv::Ptr<cv::ml::SVM>> regressors_;
  cv::Ptr<cv::ml::SVM> classifier_;
};

class CVModel: public ModelWrapper{
 public:
  bool LoadFromFile(const std::string& path) override;
  void Predict(const cv::Mat& feature, cv::Mat* predicted) const override;
 private:
  cv::Ptr<cv::ml::SVM> regressor_;
};

}  // namespace IMUProject

#endif // CODE_SPEED_REGRESSION_MODEL_WRAPPER_H
