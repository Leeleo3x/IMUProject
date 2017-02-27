//
// Created by yanhang on 2/27/17.
//

#include <memory>
#include <gflags/gflags.h>
#include <glog/logging.h>

#include "data_io.h"

using namespace std;
using namespace cv;
using namespace IMUProject;

int main(int argc, char** argv){
    if(argc < 2){
        std::cerr << "Usage: ./IMUUtility_cli <path-to-data>" << std::endl;
        return 1;
    }
    google::ParseCommandLineFlags(&argc, &argv, true);
    FLAGS_logtostderr = true;
    google::InitGoogleLogging(argv[0]);

    IMUProject::IMUDataset dataset(argv[1]);
    cout << "Number of samples:" << dataset.GetPosition().rows << endl;

    cout << dataset.GetPosition().rowRange(0, 10);
    return 0;
}
