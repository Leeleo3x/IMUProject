//
// Created by yanhang on 2/2/17.
//

#include <iostream>

#include <gflags/gflags.h>
#include <glog/logging.h>
#include <opencv2/opencv.hpp>

int main(int argc, char** argv){
    google::InitGoogleLogging(argv[0]);
    CHECK_GE(argc, 2) << "Usage: ./VisualizeTrajectory <path-to-csv-file>";
    google::ParseCommandLineFlags(&argc, &argv, true);

    // Read data
    cv::Ptr<cv::ml::TrainData> data_all = cv::ml::TrainData::loadFromCSV(std::string(argv[1]), 1);

}