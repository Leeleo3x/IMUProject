//
// Created by yanhang on 2/6/17.
//
#include <vector>
#include <string>
#include <memory>

#include <glog/logging.h>
#include <gflags/gflags.h>

#include <opencv2/opencv.hpp>

#include "imu_optimization.h"

using namespace std;

int main(int argc, char** argv){
    if(argc < 3){
        cerr << "Usage: ./IMUOptimization <path-to-datasets> <path-to-output-ply>" << endl;
        return 1;
    }

    google::InitGoogleLogging(argv[0]);
    google::ParseCommandLineFlags(&argc, &argv, true);

    cv::Ptr<cv::ml::TrainData> dataset = cv::ml::TrainData::loadFromCSV(string(argv[1]) + "/processed/data.csv", 1);
    return 0;
}