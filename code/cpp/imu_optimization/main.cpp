//
// Created by yanhang on 2/6/17.
//
#include <vector>
#include <string>
#include <memory>
#include <fstream>
#include <random>

#include <glog/logging.h>
#include <gflags/gflags.h>

#include <opencv2/opencv.hpp>

#include <utility/data_io.h>
#include <utility/utility.h>
#include <speed_regression/speed_regression.h>

#include "imu_optimization.h"

using namespace std;

DEFINE_int32(max_iter, 500, "maximum iteration");
DEFINE_int32(window, 200, "Window size");
DEFINE_string(model_path, "../../../../models", "Path to models");

int main(int argc, char** argv) {
    if (argc < 2) {
        cerr << "Usage: ./IMUOptimization <path-to-datasets>" << endl;
        return 1;
    }

    google::ParseCommandLineFlags(&argc, &argv, true);
    FLAGS_logtostderr = true;
    google::InitGoogleLogging(argv[0]);

    char buffer[128] = {};

    printf("Loading...\n");
    IMUProject::IMUDataset dataset(argv[1]);
	printf("Total count: %d\n", (int)dataset.GetTimeStamp().size());
	CHECK_GT(dataset.GetTimeStamp().size(), IMUProject::Config::kTotalCount);
	const std::vector<double> ts(dataset.GetTimeStamp().begin(),
	                             dataset.GetTimeStamp().begin() + IMUProject::Config::kTotalCount);
	const std::vector<Eigen::Vector3d> gyro(dataset.GetGyro().begin(),
	                                        dataset.GetGyro().begin() + IMUProject::Config::kTotalCount);
	const std::vector<Eigen::Vector3d> linacce(dataset.GetLinearAcceleration().begin(),
	                                           dataset.GetLinearAcceleration().begin() + IMUProject::Config::kTotalCount);
	const std::vector<Eigen::Quaterniond> orientation(dataset.GetOrientation().begin(),
	                                                  dataset.GetOrientation().begin() + IMUProject::Config::kTotalCount);

	// Load regressors
	std::vector<cv::Ptr<cv::ml::SVM> > regressors(3);
	for(int i = 0; i<3; ++i){
		sprintf(buffer, "%s/model_direct_local_speed_w200_s10_%d_cv.yml", FLAGS_model_path.c_str(), i);
		regressors[i] = cv::ml::SVM::load(buffer);
		cout << buffer << " loaded" << endl;
		CHECK(regressors[i].get()) << "Can not load " << buffer;
	}

	// Load constraints
	std::vector<int> constraint_ind;
	std::vector<Eigen::Vector3d> local_speed;

	{
		// regress local speed
		constraint_ind.resize(IMUProject::Config::kConstriantPoints);
		local_speed.resize(constraint_ind.size(), Eigen::Vector3d(0, 0, 0));
		constraint_ind[0] = FLAGS_window;
		const int constraint_interval = (IMUProject::Config::kTotalCount - FLAGS_window) /
				(IMUProject::Config::kConstriantPoints - 1);

		for(int i=1; i<constraint_ind.size(); ++i){
			constraint_ind[i] = constraint_ind[i-1] + constraint_interval;
		}

		printf("Regressing local speed...\n");
		for(int i=0; i<constraint_ind.size(); ++i){
			const int sid = constraint_ind[i] - FLAGS_window;
			const int eid = constraint_ind[i];
			cv::Mat feature = IMUProject::ComputeDirectFeature(&gyro[sid], &linacce[sid], FLAGS_window);
			for(int j=0; j<regressors.size(); ++j){
				local_speed[i][j] = static_cast<double>(regressors[j]->predict(feature));
			}
		}
	}


	// Formulate problem
    printf("Constructing problem...\n");
    ceres::Problem problem;
    constexpr int kResiduals = IMUProject::Config::kConstriantPoints;
    constexpr int kSparsePoint = IMUProject::Config::kSparsePoints;
    // Initialize bias with gaussian distribution
    std::vector<double> bx((size_t)kSparsePoint, 0.0), by((size_t)kSparsePoint, 0.0), bz((size_t)kSparsePoint,0.0);
    std::default_random_engine generator;
    std::normal_distribution<double> distribution(0.0, 0.01);
    for(int i=0; i<kSparsePoint; ++i){
        bx[i] = distribution(generator);
        by[i] = distribution(generator);
        bz[i] = distribution(generator);
    }

	IMUProject::LocalSpeedFunctor<kSparsePoint, kResiduals>* functor =
			new IMUProject::LocalSpeedFunctor<kSparsePoint, kResiduals>(ts, linacce, orientation, constraint_ind, local_speed, Eigen::Vector3d(0, 0, 0));

	constexpr int kVar = IMUProject::Config::kSparsePoints;
	problem.AddResidualBlock(new ceres::AutoDiffCostFunction<IMUProject::LocalSpeedFunctor<kSparsePoint, kResiduals>, kResiduals * 3, kSparsePoint, kSparsePoint, kSparsePoint>(
			functor), nullptr, bx.data(), by.data(), bz.data());

    problem.AddResidualBlock(new ceres::AutoDiffCostFunction<IMUProject::WeightDecay, kSparsePoint, kSparsePoint>(
            new IMUProject::WeightDecay(1.0)
    ), nullptr, bx.data());

    problem.AddResidualBlock(new ceres::AutoDiffCostFunction<IMUProject::WeightDecay, kSparsePoint, kSparsePoint>(
            new IMUProject::WeightDecay(1.0)
    ), nullptr, by.data());

    problem.AddResidualBlock(new ceres::AutoDiffCostFunction<IMUProject::WeightDecay, kSparsePoint, kSparsePoint>(
            new IMUProject::WeightDecay(1.0)
    ), nullptr, bz.data());


    float start_t = cv::getTickCount();
    ceres::Solver::Options options;
    options.linear_solver_type = ceres::DENSE_QR;
	options.num_threads = 6;
    options.minimizer_progress_to_stdout = true;
    options.max_num_iterations = FLAGS_max_iter;
    ceres::Solver::Summary summary;

    printf("Solving...\n");
    ceres::Solve(options, &problem, &summary);

    std::cout << summary.BriefReport() << endl;
    printf("Time usage: %.3fs\n", ((float)cv::getTickCount() - start_t) / cv::getTickFrequency());

	// Compute corrected trajectory
	printf("Computing trajectory...\n");

	std::vector<Eigen::Vector3d> corrected_linacce = linacce;
	functor->GetGrid()->correct_bias<double>(corrected_linacce.data(), bx.data(), by.data(), bz.data());
	std::vector<Eigen::Vector3d> corrected_position =
			IMUProject::Integration(ts,
			                        IMUProject::Integration(ts,
			                                                IMUProject::Rotate3DVector(corrected_linacce, orientation)));
	sprintf(buffer, "%s/optimized_cpp.ply", argv[1]);
	IMUProject::WriteToPly(std::string(buffer), corrected_position, orientation);

    return 0;
}