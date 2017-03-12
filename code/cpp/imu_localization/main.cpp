//
// Created by yanhang on 3/5/17.
//

#include <vector>
#include <string>

#include <opencv2/opencv.hpp>
#include <Eigen/Eigen>
#include <gflags/gflags.h>

#include <utility/data_io.h>

#include "imu_localization.h"

DEFINE_string(model_path, "../../../../models/model_0309_body_w200_s10", "Path to model");
DEFINE_int32(log_interval, 1000, "logging interval");

using namespace std;

int main(int argc, char** argv){
    if(argc < 2){
        cerr << "Usage: ./IMULocalization_cli <path-to-data>" << endl;
        return 1;
    }

    google::InitGoogleLogging(argv[0]);
    google::ParseCommandLineFlags(&argc, &argv, true);
    FLAGS_logtostderr = true;

    char buffer[128] = {};

    LOG(INFO) << "Initializing...";
    // load data
    IMUProject::IMUDataset dataset(argv[1]);

    // load regressor
    std::vector<cv::Ptr<cv::ml::SVM> > regressors(3);
    for(int chn: {0, 2}){
        sprintf(buffer, "%s_%d.yml", FLAGS_model_path.c_str(), chn);
        regressors[chn] = cv::ml::SVM::load(std::string(buffer));
        LOG(INFO) << buffer << " loaded";
    }

    // crete trajectory instance
    IMUProject::IMULocalizationOption option;
    const double sigma = 0.2;
    IMUProject::IMUTrajectory trajectory(Eigen::Vector3d(0, 0, 0), dataset.GetPosition()[0], regressors, sigma, option);

    // Run the system
    const int N = (int)dataset.GetTimeStamp().size();
    const std::vector<double>& ts = dataset.GetTimeStamp();
    const std::vector<Eigen::Vector3d>& gyro = dataset.GetGyro();
    const std::vector<Eigen::Vector3d>& linacce = dataset.GetLinearAcceleration();
    std::vector<Eigen::Quaterniond> orientation = dataset.GetRotationVector();
    Eigen::Quaterniond rv_to_tango = dataset.GetOrientation()[0] * dataset.GetRotationVector()[0].conjugate();
    for(auto& v: orientation){
        v = rv_to_tango * v;
    }

    float start_t = (float)cv::getTickCount();

    constexpr int init_capacity = 20000;
    std::vector<Eigen::Vector3d> positions_opt;
    std::vector<Eigen::Quaterniond> orientations_opt;
//    positions_opt.reserve(init_capacity);
//    orientations_opt.reserve(init_capacity);

    for(int i=0; i < N; ++i){
        if(i > 1000){
            if(i % option.global_opt_interval_ == 0) {
                LOG(INFO) << "Running global optimzation at frame " << i;
//                trajectory.RunOptimization(0, trajectory.GetNumFrames());

                //block the execution is there are too many tasks in the background thread
                while(true) {
                    if(trajectory.CanAdd()){
                        break;
                    }
                }
                trajectory.ScheduleOptimization(0, trajectory.GetNumFrames());
            }else if(i % option.opt_interval_ == 0){
//                while(true) {
//                    if(trajectory.CanAdd()){
//                        break;
//                    }
//                }
//                LOG(INFO) << "Running local optimzation at frame " << i;
//                //trajectory.RunOptimization(i - 1000, 1000);
//                trajectory.ScheduleOptimization(i - 1000, 800);
            }
            trajectory.AddRecord(ts[i], gyro[i], linacce[i], orientation[i]);
        }
        if(FLAGS_log_interval > 0 && i > 0 && i % FLAGS_log_interval == 0){
            const float time_passage = std::max(((float)cv::getTickCount() - start_t) / (float)cv::getTickFrequency(),
                                                std::numeric_limits<float>::epsilon());
            sprintf(buffer, "%d records added in %.5fs, fps=%.2fHz\n", i, time_passage, (float) i / time_passage);
            LOG(INFO) << buffer;
        }
    }

    trajectory.EndTrajectory();
    printf("Running global optimization on the whole sequence...\n");
    trajectory.RunOptimization(0, trajectory.GetNumFrames());

    printf("All done. Number of points on trajectory: %d\n", trajectory.GetNumFrames());
    const float fps_all = (float)trajectory.GetNumFrames() / (((float)cv::getTickCount() - start_t) / (float)cv::getTickFrequency());
    printf("Overall framerate: %.3f\n", fps_all);

    sprintf(buffer, "%s/result_trajectory.ply", argv[1]);
    // IMUProject::WriteToPly(std::string(buffer), positions_opt.data(), orientations_opt.data(), (int)positions_opt.size());
    IMUProject::WriteToPly(std::string(buffer), trajectory.GetPositions().data(),
                           trajectory.GetOrientations().data(), trajectory.GetNumFrames());
    LOG(INFO) << "Optimized trajectory written to " << buffer;

    return 0;
}