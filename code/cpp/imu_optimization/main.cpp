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
DEFINE_string(model_path, "../../../../models/model_0308_gaussian_w200_s10", "Path to models");
DEFINE_bool(gt, false, "Use ground truth");
DEFINE_double(feature_smooth_alpha, -1, "cut-off threshold for ");
DEFINE_double(weight_ls, 1.0, "The weight of local speed residual");
DEFINE_double(weight_vs, 1.0, "The weight of vertical speed residual");

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
	const int kTotalCount = (int)dataset.GetTimeStamp().size();
//    const int kTotalCount = 3000;

	printf("Total count: %d\n", kTotalCount);
	const std::vector<double> ts(dataset.GetTimeStamp().begin(),
	                             dataset.GetTimeStamp().begin() + kTotalCount);
	std::vector<Eigen::Vector3d> gyro(dataset.GetGyro().begin(),
	                                        dataset.GetGyro().begin() + kTotalCount);
	std::vector<Eigen::Vector3d> linacce(dataset.GetLinearAcceleration().begin(),
	                                           dataset.GetLinearAcceleration().begin() + kTotalCount);
//	const std::vector<Eigen::Quaterniond> orientation(dataset.GetOrientation().begin(),
//	                                                  dataset.GetOrientation().begin() + kTotalCount);

	std::vector<Eigen::Quaterniond> orientation(dataset.GetRotationVector().begin(),
												dataset.GetRotationVector().begin() + kTotalCount);

	const Eigen::Quaterniond& tango_init_ori = dataset.GetOrientation()[0];
	const Eigen::Quaterniond imu_init_ori = orientation[0];
	auto align_ori = tango_init_ori * imu_init_ori.conjugate();
	for(auto i=0; i<orientation.size(); ++i){
		orientation[i] = align_ori * orientation[i];
	}


	const double feature_smooth_sigma = 2.0;
	IMUProject::GaussianFilter(&gyro[0], (int)gyro.size(), feature_smooth_sigma);
	IMUProject::GaussianFilter(&linacce[0], (int)linacce.size(), feature_smooth_sigma);

	// Load constraints
	std::vector<int> constraint_ind;
	std::vector<Eigen::Vector3d> local_speed;

	// regress local speed
	constraint_ind.resize(IMUProject::Config::kConstriantPoints);
	local_speed.resize(constraint_ind.size(), Eigen::Vector3d(0, 0, 0));
	constraint_ind[0] = FLAGS_window;
	const int constraint_interval = (kTotalCount - FLAGS_window) /
									(IMUProject::Config::kConstriantPoints - 1);

	for(int i=1; i<constraint_ind.size(); ++i){
		constraint_ind[i] = constraint_ind[i-1] + constraint_interval;
	}

	if(FLAGS_gt){
		LOG(WARNING) << "Using ground truth as constraint";
		const std::vector<Eigen::Vector3d> positions(
				dataset.GetPosition().begin(),
				dataset.GetPosition().begin() + kTotalCount);
		cv::Mat local_speed_mat = IMUProject::ComputeLocalSpeedTarget(ts, positions, orientation, constraint_ind, 10);
		CHECK_EQ(local_speed_mat.rows, local_speed.size());
		for(auto i=0; i<local_speed.size(); ++i){
			local_speed[i][0] = (double)local_speed_mat.at<float>(i, 0);
			local_speed[i][1] = (double)local_speed_mat.at<float>(i, 1);
			local_speed[i][2] = (double)local_speed_mat.at<float>(i, 2);
		}
	}else{
		printf("Regressing local speed...\n");
		// Load regressors
		std::vector<cv::Ptr<cv::ml::SVM> > regressors(3);
		for(auto i: {0, 2}){
			sprintf(buffer, "%s_%d.yml", FLAGS_model_path.c_str(), i);
			regressors[i] = cv::ml::SVM::load(buffer);
			cout << buffer << " loaded" << endl;
			CHECK(regressors[i].get()) << "Can not load " << buffer;
		}

#pragma omp parallel for
		for(int i=0; i<constraint_ind.size(); ++i){
			const int sid = constraint_ind[i] - FLAGS_window;
			const int eid = constraint_ind[i];
			cv::Mat feature = IMUProject::ComputeDirectFeature(&gyro[sid], &linacce[sid], FLAGS_window);
			for(auto j: {0, 2}){
				local_speed[i][j] = static_cast<double>(regressors[j]->predict(feature));
			}
		}
	}



	// Formulate problem
    printf("Constructing problem...\n");
    ceres::Problem problem;
    constexpr int kResiduals = IMUProject::Config::kConstriantPoints;
	constexpr int kOriSparsePoint = IMUProject::Config::kOriSparsePoint;
    constexpr int kSparsePoint = IMUProject::Config::kSparsePoints;
    // Initialize bias with gaussian distribution
    std::vector<double> bx((size_t)kSparsePoint, 0.0), by((size_t)kSparsePoint, 0.0), bz((size_t)kSparsePoint,0.0);

//	std::vector<double> rz((size_t)kOriSparsePoint, 0.0);
//    std::default_random_engine generator;
//    std::normal_distribution<double> distribution(0.0, 0.001);
//    for(int i=0; i<kSparsePoint; ++i){
//        bx[i] = distribution(generator);
//        by[i] = distribution(generator);
//        bz[i] = distribution(generator);
//    }



//	using FunctorType = IMUProject::LocalSpeedAndOrientationFunctor<kSparsePoint, kOriSparsePoint, kResiduals>;
	using FunctorType = IMUProject::LocalSpeedFunctor<kSparsePoint, kResiduals>;

	FunctorType* functor = new FunctorType(ts, linacce, orientation, constraint_ind, local_speed, Eigen::Vector3d(0, 0, 0), FLAGS_weight_ls, FLAGS_weight_vs);
	problem.AddResidualBlock(new ceres::AutoDiffCostFunction<FunctorType, 3 * kResiduals, kSparsePoint, kSparsePoint, kSparsePoint>(
			functor), nullptr, bx.data(), by.data(), bz.data());


	// Use LocalSpeedAndOrientation functor
//	FunctorType* functor = new FunctorType(ts, linacce, orientation, constraint_ind, local_speed, Eigen::Vector3d(0,0,0), FLAGS_weight_ls, FLAGS_weight_vs);
//	problem.AddResidualBlock(new ceres::NumericDiffCostFunction<FunctorType, ceres::CENTRAL, 3 * kResiduals, kSparsePoint, kSparsePoint, kSparsePoint, kOriSparsePoint>(
//			functor), nullptr, bx.data(), by.data(), bz.data(), rz.data());


    problem.AddResidualBlock(new ceres::AutoDiffCostFunction<IMUProject::WeightDecay<kSparsePoint>, kSparsePoint, kSparsePoint>(
            new IMUProject::WeightDecay<kSparsePoint>(1.0)
    ), nullptr, bx.data());

	problem.AddResidualBlock(new ceres::AutoDiffCostFunction<IMUProject::WeightDecay<kSparsePoint>, kSparsePoint, kSparsePoint>(
			new IMUProject::WeightDecay<kSparsePoint>(1.0)
	), nullptr, by.data());

	problem.AddResidualBlock(new ceres::AutoDiffCostFunction<IMUProject::WeightDecay<kSparsePoint>, kSparsePoint, kSparsePoint>(
			new IMUProject::WeightDecay<kSparsePoint>(1.0)
	), nullptr, bz.data());

//	problem.AddResidualBlock(new ceres::AutoDiffCostFunction<IMUProject::WeightDecay<kOriSparsePoint>, kOriSparsePoint, kOriSparsePoint>(
//			new IMUProject::WeightDecay<kOriSparsePoint>(1.0)
//	), nullptr, rz.data());

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
	std::vector<Eigen::Quaterniond> corrected_orientation = orientation;

	functor->GetLinacceGrid()->correct_linacce_bias<double>(corrected_linacce.data(), bx.data(), by.data(), bz.data());
//	functor->GetOrientationGrid()->correct_orientation(corrected_orientation.data(), rz.data());

	std::vector<Eigen::Vector3d> directed_corrected_linacce = IMUProject::Rotate3DVector(corrected_linacce, corrected_orientation);
	std::vector<Eigen::Vector3d> corrected_speed = IMUProject::Integration(ts, directed_corrected_linacce);
	std::vector<Eigen::Vector3d> corrected_position = IMUProject::Integration(ts, corrected_speed, dataset.GetPosition()[0]);

	sprintf(buffer, "%s/optimized_cpp_gaussian.ply", argv[1]);
	IMUProject::WriteToPly(std::string(buffer), corrected_position, orientation);

	std::vector<Eigen::Vector3d> directed_linacce = IMUProject::Rotate3DVector(linacce, orientation);
	std::vector<Eigen::Vector3d> speed = IMUProject::Integration(ts, directed_linacce);
	std::vector<Eigen::Vector3d> raw_position = IMUProject::Integration(ts, speed, dataset.GetPosition()[0]);

	sprintf(buffer, "%s/raw.ply", argv[1]);
	IMUProject::WriteToPly(std::string(buffer), raw_position, orientation);

    return 0;
}