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

using namespace std;
int main(int argc, char** argv){
    if(argc < 2){
        std::cerr << "Usage: ./IMUUtility_cli <path-to-data>" << std::endl;
        return 1;
    }
    google::ParseCommandLineFlags(&argc, &argv, true);
    FLAGS_logtostderr = true;
    google::InitGoogleLogging(argv[0]);

    std::string test_string = "0,6806.688065023,0.006109,-0.007963,0.004010,0.000000,0.000000,0.000000,0.000000,0.000000,0.000000";
	std::vector<double> parsed = IMUProject::ParseCommaSeparatedLine(test_string);
	cout << "Parsed line: " << endl;
	for(const auto v: parsed){
		cout << v << ' ';
	}
	cout << endl;

    return 0;
}
