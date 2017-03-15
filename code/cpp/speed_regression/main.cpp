//
// Created by Yan Hang on 3/4/17.
//


#include <gflags/gflags.h>

#include <iostream>
#include <fstream>

#include <utility/data_io.h>

#include "speed_regression.h"

using namespace std;

DEFINE_int32(step, 10, "interval between two samples");
DEFINE_int32(window, 200, "Window size");
DEFINE_int32(smooth_size, 10, "smooth size");
DEFINE_string(output, "../../../../models", "output folder for models");

int main(int argc, char ** argv) {
	if (argc < 2) {
		cerr << "Usage: ./SpeedRegression_cli <path-to-list>" << endl;
		return -1;
	}

	google::InitGoogleLogging(argv[0]);
	google::ParseCommandLineFlags(&argc, &argv, true);
	FLAGS_logtostderr = true;

	char buffer[128] = {};

	IMUProject::IMUDataset dataset(argv[1]);

	// test the conversion between euler angle and quaternion
	std::vector<Eigen::Vector3d> gyro = dataset.GetGyro();

	double diff = 0;
	double diff_v = 0;
	for(auto i=0; i<gyro.size(); ++i){
		Eigen::Quaterniond quat = Eigen::AngleAxis<double>(gyro[i][0], Eigen::Vector3d::UnitX()) *
		                          Eigen::AngleAxis<double>(gyro[i][1], Eigen::Vector3d::UnitY()) *
		                          Eigen::AngleAxis<double>(gyro[i][2], Eigen::Vector3d::UnitZ());
		Eigen::Vector3d gyro2 = quat.toRotationMatrix().eulerAngles(0, 1, 2);

		Eigen::Quaterniond quat2 = Eigen::AngleAxis<double>(gyro2[0], Eigen::Vector3d::UnitX()) *
		                           Eigen::AngleAxis<double>(gyro2[1], Eigen::Vector3d::UnitY()) *
		                           Eigen::AngleAxis<double>(gyro2[2], Eigen::Vector3d::UnitZ());

//		cout << "--------------------\n" << flush;
//		cout << (quat2.conjugate() * quat).toRotationMatrix() << endl << flush;
//		cout << "--------------------\n" << flush;

		Eigen::Vector3d gyro3 = IMUProject::AdjustEulerAngle(gyro2);

		Eigen::Quaterniond quat3 = Eigen::AngleAxis<double>(gyro3[0], Eigen::Vector3d::UnitX()) *
		                           Eigen::AngleAxis<double>(gyro3[1], Eigen::Vector3d::UnitY()) *
		                           Eigen::AngleAxis<double>(gyro3[2], Eigen::Vector3d::UnitZ());

//		cout << "--------------------\n" << flush;
//		cout << (quat3.conjugate() * quat).toRotationMatrix() << endl << flush;
//		cout << "--------------------\n" << flush;

//        Eigen::Vector3d gyro4(gyro[i][0]+M_PI, -gyro[i][1]+M_PI, gyro[i][2]+M_PI);
//        Eigen::Quaterniond quat4 = Eigen::AngleAxis<double>(gyro4[0], Eigen::Vector3d::UnitX()) *
//                                   Eigen::AngleAxis<double>(gyro4[1], Eigen::Vector3d::UnitY()) *
//                                   Eigen::AngleAxis<double>(gyro4[2], Eigen::Vector3d::UnitZ());
//        cout << "--------------------\n" << flush;
//        cout << (quat4.conjugate() * quat).toRotationMatrix() << endl << flush;
//        cout << "--------------------\n" << flush;

//		printf("%d %.6f, %.6f, %.6f | %.6f, %.6f, %.6f | %.6f, %.6f, %.6f | %.6f, %.6f, %.6f\n",
//		       i, gyro[i][0], gyro[i][1], gyro[i][2], gyro2[0], gyro2[1], gyro2[2], gyro3[0], gyro3[1], gyro3[2],
//               gyro4[0], gyro4[1], gyro4[2]);
		double cur_diff = ((quat3.conjugate() * quat).toRotationMatrix() - Eigen::Matrix3d::Identity()).norm();
		if(cur_diff > 1e-9){
			cout << "Fatal diff!\n";
		}

		double cur_diff_v = (gyro3 - gyro[i]).norm();
		if(cur_diff_v > 1e-9){
			cout << "Different value!\n";
		}
		diff += cur_diff;
		diff_v += cur_diff_v;
	}

	printf("diff: %.5f\n", diff);
	printf("diff_v: %.5f\n", diff_v);

	return 0;
}