//
// Created by yanhang on 9/30/17.
//

#include "speed_regression/model_wrapper.h"

#include <fstream>

namespace IMUProject {

const std::string SVRCascadeOption::kVersionTag = "v1.0";

std::istream &operator>>(std::istream &stream, SVRCascadeOption &option) {
  std::string version_tag;
  stream >> version_tag;
  CHECK_EQ(version_tag, option.kVersionTag) << "The version of the file " << version_tag
                                            << " doesn't match the current version " << option.kVersionTag;
  stream >> option.num_classes >> option.num_channels;
  return stream;
}

bool SVRCascade::LoadFromFile(const std::string &path) {
  std::ifstream option_in(path + "/option.txt");
  if (!option_in.is_open()) {
    LOG(ERROR) << "Can not open option file: " << path + "/option.txt";
    return false;
  }

  // Load the option
  option_in >> option_;

  // Load the class map
  class_names_.resize(GetNumClasses());
  std::ifstream classmap_in(path + "/class_map.txt");
  if (!classmap_in.is_open()){
    LOG(ERROR) << "Can not open class map file " << path + "/class_map.txt";
    return false;
  }
  std::string name;
  int number;
  classmap_in >> number;
  if (number != GetNumClasses()){
    LOG(ERROR) << "The number of classes in the class map file doesn't match the one in the option file: "
        << number << " vs " << GetNumClasses();
    return false;
  }
  for (int i=0; i<GetNumClasses(); ++i){
    classmap_in >> name >> number;
    if (number < 0 || number > GetNumClasses()){
      LOG(ERROR) << "Invalid class number encountered: " << number;
      return false;
    }
    class_names_[number] = name;
  }

  // Load the classifier
  classifier_ = cv::ml::SVM::load(path + "/classifier.yaml");
  if (!classifier_.get()) {
    LOG(ERROR) << "Can not read the classifier from " << path + "/classifier.yaml";
    return false;
  }
  LOG(INFO) << "Classifier " << path + "/classifier.yaml loaded";

  regressors_.resize(GetNumChannels() * GetNumClasses());
  char buffer[128] = {};
  for (int cls = 0; cls < GetNumClasses(); ++cls) {
    for (int chn = 0; chn < GetNumChannels(); ++chn) {
      int rid = cls * GetNumChannels() + chn;
      sprintf(buffer, "%s/regressor_%d_%d.yaml", path.c_str(), cls, chn);
      regressors_[rid] = cv::ml::SVM::load(buffer);
      if (!regressors_[rid].get()){
        LOG(ERROR) << "Can not load regressor " << buffer;
        return false;
      }
      LOG(INFO) << "Regressor " << rid << ':' << buffer << " loaded";
    }
  }
  return true;
}

void SVRCascade::Predict(const cv::Mat &feature, cv::Mat *predicted) const {
  cv::Mat label;
  Predict(feature, &label, predicted);
}

void SVRCascade::Predict(const cv::Mat &feature, cv::Mat *label, cv::Mat *response) const {
  // Predict the label
  CHECK(label) << "The provided output label matrix is empty";
  CHECK_NOTNULL(classifier_.get())->predict(feature, *label);

  printf("label->rows: %d", label->rows);
  std::vector<int> num_sample_in_class(GetNumClasses());
  for (int i=0; i<label->rows; ++i){

  }
  // For each class, copy the corresponding samples to a seperate Mat.
  for (int cls = 0; cls < GetNumClasses(); ++cls){

    for (int chn = 0; chn < GetNumChannels(); ++chn){

    }
  }
}

bool CVModel::LoadFromFile(const std::string &path) {
  regressor_ = cv::ml::SVM::load(path);
  if (!regressor_.get()){
    LOG(ERROR) << "Can not open regressioni model: " << path;
    return false;
  }
  return true;
}

void CVModel::Predict(const cv::Mat &feature, cv::Mat *predicted) const {

}

}  // namespace IMUProject