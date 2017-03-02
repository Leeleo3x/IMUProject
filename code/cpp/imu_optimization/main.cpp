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

#include "imu_optimization.h"

using namespace std;

DEFINE_int32(max_iter, 500, "maximum iteration");

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
	const std::vector<double> ts(dataset.GetTimeStamp().begin(),
	                             dataset.GetTimeStamp().begin() + IMUProject::Config::kTotalCount);
	const std::vector<Eigen::Vector3d> linacce(dataset.GetLinearAcceleration().begin(),
	                                           dataset.GetLinearAcceleration().begin() + IMUProject::Config::kTotalCount);
	const std::vector<Eigen::Quaterniond> orientation(dataset.GetOrientation().begin(),
	                                                  dataset.GetOrientation().begin() + IMUProject::Config::kTotalCount);

    // Load constraints
    std::vector<double> target_speed_mag;
    std::vector<double> target_vspeed;
    std::vector<int> constraint_ind;

    {
        // load speed magnitude
        sprintf(buffer, "%s/processed/speed_magnitude.txt", argv[1]);
        ifstream sm_in(buffer);
        CHECK(sm_in.is_open()) << "Can not open file " << buffer;
        int kSMConstraints;
        sm_in >> kSMConstraints;
	    CHECK_GE(kSMConstraints, IMUProject::Config::kConstriantPoints);
        target_speed_mag.resize((size_t) IMUProject::Config::kConstriantPoints);
        constraint_ind.resize((size_t) IMUProject::Config::kConstriantPoints);
        for (int i = 0; i < IMUProject::Config::kConstriantPoints; ++i) {
            sm_in >> constraint_ind[i] >> target_speed_mag[i];
        }
    }

    {
        // load vertical speed
        sprintf(buffer, "%s/processed/vertical_speed.txt", argv[1]);
        ifstream vs_in(buffer);
        CHECK(vs_in.is_open()) << "Can not open file " << buffer;
        int kVSConstraints;
        vs_in >> kVSConstraints;
        target_vspeed.resize((size_t) IMUProject::Config::kConstriantPoints);
        int cid;
        for (int i = 0; i < IMUProject::Config::kConstriantPoints; ++i) {
            vs_in >> cid;
            CHECK_EQ(cid, constraint_ind[i]);
            vs_in >> target_vspeed[i];
        }
    }


    // Formulate problem
    printf("Constructing problem...\n");
    ceres::Problem problem;
    constexpr int kResiduals = IMUProject::Config::kConstriantPoints * 2;
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

    IMUProject::SizedSharedSpeedFunctor* functor = new IMUProject::SizedSharedSpeedFunctor(ts, linacce, orientation,
                                                                                           constraint_ind, target_speed_mag, target_vspeed,
                                                                                           Eigen::Vector3d(0, 0, 0), 1.0);
//    ceres::CostFunction *cost_function =
//            new ceres::NumericDiffCostFunction<IMUProject::SizedSharedSpeedFunctor, ceres::CENTRAL, kResiduals, kSparsePoint, kSparsePoint, kSparsePoint>(
//                    functor);

	ceres::CostFunction *cost_function =
			new ceres::AutoDiffCostFunction<IMUProject::SizedSharedSpeedFunctor, kResiduals, kSparsePoint, kSparsePoint, kSparsePoint>(functor);
    problem.AddResidualBlock(cost_function, nullptr, bx.data(), by.data(), bz.data());

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
    options.minimizer_progress_to_stdout = true;
    options.max_num_iterations = FLAGS_max_iter;
    ceres::Solver::Summary summary;

    printf("Solving...\n");
    ceres::Solve(options, &problem, &summary);

    std::cout << summary.BriefReport() << endl;
    printf("Time usage: %.3fs\n", ((float)cv::getTickCount() - start_t) / cv::getTickFrequency());

	// Compute corrected trajectory
	printf("Computing trajectory...\n");
	std::vector<Eigen::Vector3d> corrected_linacce = functor->correcte_acceleration(linacce, bx, by, bz);

	std::vector<Eigen::Vector3d> corrected_position =
			IMUProject::Integration(ts,
			                        IMUProject::Integration(ts,
			                                                IMUProject::Rotate3DVector(corrected_linacce, orientation)));
	sprintf(buffer, "%s/optimized_cpp.ply", argv[1]);
	IMUProject::WriteToPly(std::string(buffer), corrected_position, orientation);

    return 0;
}