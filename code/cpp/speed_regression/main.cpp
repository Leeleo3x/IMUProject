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

	string list_path(argv[1]);
	string root_dir = list_path.substr(0, list_path.find_last_of('/'));
	cout << "Root directory: " << root_dir << endl;

	// Prepare training data


	IMUProject::TrainingDataOption option(FLAGS_step, FLAGS_window);

	cv::Mat feature_mat_all;
	cv::Mat target_mat_all;

	// Load all training data

	ifstream list_in(argv[1]);
	CHECK(list_in.is_open()) << "Can not open list file " << argv[1];

	string folder;
	while (list_in >> folder) {
		if(folder.empty()){
			continue;
		}
		if (folder[0] == '#') {
			continue;
		}
		IMUProject::IMUDataset dataset(root_dir + "/" + folder);
		const int N = (int) dataset.GetTimeStamp().size();
		printf("%s loaded. Number of samples: %d\n", folder.c_str(), N);

		vector<int> sample_points{};
		for (int i = FLAGS_window; i < N; i += FLAGS_step) {
			sample_points.push_back(i);
		}

		const std::vector<double> &time_stamp = dataset.GetTimeStamp();
		const std::vector<Eigen::Vector3d> &gyro = dataset.GetGyro();
		const std::vector<Eigen::Vector3d> &linacce = dataset.GetLinearAcceleration();

		cv::Mat feature_mat((int) sample_points.size(), 6 * option.window_, CV_32FC1, cv::Scalar::all(0));
		for (int i = 0; i < sample_points.size(); ++i) {
			const int ind = sample_points[i];
			CHECK_GE(ind - option.window_, 0);
			CHECK_LT(ind, N);
			IMUProject::ComputeDirectFeature(&gyro[ind - option.window_], &linacce[ind - option.window_], option.window_).copyTo(feature_mat.row(i));
		}

		cv::Mat target_mat = IMUProject::ComputeLocalSpeedTarget(dataset.GetTimeStamp(),
		                                                         dataset.GetPosition(),
		                                                         dataset.GetOrientation(), sample_points,
		                                                         FLAGS_smooth_size);

		if(!feature_mat_all.data){
			feature_mat_all = feature_mat.clone();
			target_mat_all = target_mat.clone();
		}else{
			cv::vconcat(feature_mat_all, feature_mat, feature_mat_all);
			cv::vconcat(target_mat_all, target_mat, target_mat_all);
		}
	}

	printf("Number of samples: %d, feature dimension: %d, target dimension: %d\n",
	       feature_mat_all.rows, feature_mat_all.cols, target_mat_all.cols);

	std::vector<double> c_param{10.0, 1.0, 10.0};
	std::vector<double> p_param{0.01, 0.001, 0.01};
	cv::TermCriteria term_criteria(cv::TermCriteria::EPS | cv::TermCriteria::MAX_ITER, 5000, 1e-09);
#pragma omp parallel for
	for (int i = 0; i < target_mat_all.cols; ++i) {
		cv::Ptr<cv::ml::TrainData> train_data = cv::ml::TrainData::create(feature_mat_all, cv::ml::ROW_SAMPLE,
		                                                                  target_mat_all.col(i).clone());

		// Configure the parameter grid
		cv::ml::ParamGrid c_grid = cv::ml::SVM::getDefaultGrid(cv::ml::SVM::C);
		c_grid.minVal = 0.01;
		c_grid.maxVal = 100.0;
		cv::ml::ParamGrid p_grid = cv::ml::SVM::getDefaultGrid(cv::ml::SVM::P);
		p_grid.minVal = 0.001;
		p_grid.maxVal = 10.0;

		// disable the search for parameter gamma, nu, coef and degree
		cv::ml::ParamGrid gamma_grid = cv::ml::SVM::getDefaultGrid(cv::ml::SVM::GAMMA);
		gamma_grid.logStep = -1;
		cv::ml::ParamGrid nu_grid = cv::ml::SVM::getDefaultGrid(cv::ml::SVM::NU);
		nu_grid.logStep = -1;
		cv::ml::ParamGrid coef_grid = cv::ml::SVM::getDefaultGrid(cv::ml::SVM::COEF);
		coef_grid.logStep = -1;
		cv::ml::ParamGrid degree_grid = cv::ml::SVM::getDefaultGrid(cv::ml::SVM::DEGREE);
		degree_grid.logStep = -1;

		cv::Ptr<cv::ml::SVM> regressor = cv::ml::SVM::create();
		regressor->setType(cv::ml::SVM::RBF);
		regressor->setP(p_param[i]);
		regressor->setC(c_param[i]);
		regressor->setGamma(1.0 / (double)feature_mat_all.cols);
		regressor->setTermCriteria(term_criteria);

		regressor->setType(cv::ml::SVM::EPS_SVR);
		float start_t = (float) cv::getTickCount();
		cout << "Training for channel " << i << endl;
		// regressor->trainAuto(train_data, 3, c_grid, gamma_grid, p_grid, nu_grid, coef_grid, degree_grid);
		regressor->train(train_data);
		printf("Done. Number of support vectors: %d. Time usage %.2fs\n",
		       regressor->getSupportVectors().rows,
		       ((float) cv::getTickCount() - start_t) / (float) cv::getTickFrequency());
		printf("Parameter: C: %.6f, p: %.6f\n", regressor->getC(), regressor->getP());

		sprintf(buffer, "%s/model_direct_local_speed_w200_s10_%d.yml", FLAGS_output.c_str(), i);
		regressor->save(std::string(buffer));
		cout << "Model saved to " << buffer << endl;
	}
	return 0;
}