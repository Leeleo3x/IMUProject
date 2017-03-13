//
// Created by yanhang on 3/5/17.
//

#include <vector>
#include <string>

#include <opencv2/opencv.hpp>
#include <Eigen/Eigen>
#include <gflags/gflags.h>

#include <utility/data_io.h>
#include <utility/utility.h>

#include "imu_localization.h"

DEFINE_string(model_path, "../../../../models/model_0312_body_w200_s10", "Path to model");
DEFINE_string(mapinfo_path, "default", "path to map info");
DEFINE_int32(log_interval, 1000, "logging interval");
DEFINE_bool(run_global, true, "Run global optimization at the end");
DEFINE_bool(tango_ori, false, "Use ground truth orientation");

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
    std::vector<Eigen::Quaterniond> orientation;
    if(FLAGS_tango_ori){
        orientation = dataset.GetOrientation();
    }else {
        orientation = dataset.GetRotationVector();
        Eigen::Quaterniond rv_to_tango = dataset.GetOrientation()[0] * dataset.GetRotationVector()[0].conjugate();
        for(auto& v: orientation){
            v = rv_to_tango * v;
        }
    }

    float start_t = (float)cv::getTickCount();

    constexpr int init_capacity = 20000;
    std::vector<Eigen::Vector3d> positions_opt;
    std::vector<Eigen::Quaterniond> orientations_opt;
//    positions_opt.reserve(init_capacity);
//    orientations_opt.reserve(init_capacity);

    for(int i = 0; i < N; ++i){
	    trajectory.AddRecord(ts[i], gyro[i], linacce[i], orientation[i]);

	    if(i > option.local_opt_window_){
            if(i % option.global_opt_interval_ == 0) {
//                LOG(INFO) << "Running global optimzation at frame " << i;
////                trajectory.RunOptimization(0, trajectory.GetNumFrames());
//
//                //block the execution is there are too many tasks in the background thread
//                while(true) {
//                    if(trajectory.CanAdd()){
//                        break;
//                    }
//                }
//                trajectory.ScheduleOptimization(0, trajectory.GetNumFrames());
            }else if(i % option.local_opt_interval_ == 0){
	            LOG(INFO) << "Running local optimzation at frame " << i;
                while(true) {
                    if(trajectory.CanAdd()){
                        break;
                    }
                }
	            trajectory.ScheduleOptimization(i - option.local_opt_window_, option.local_opt_window_);
//	            trajectory.RunOptimization(i  - option.local_opt_window_, option.local_opt_window_);
            }
        }
        if(FLAGS_log_interval > 0 && i > 0 && i % FLAGS_log_interval == 0){
            const float time_passage = std::max(((float)cv::getTickCount() - start_t) / (float)cv::getTickFrequency(),
                                                std::numeric_limits<float>::epsilon());
            sprintf(buffer, "%d records added in %.5fs, fps=%.2fHz\n", i, time_passage, (float) i / time_passage);
            LOG(INFO) << buffer;
        }

    }

    trajectory.EndTrajectory();
	if(FLAGS_run_global) {
		printf("Running global optimization on the whole sequence...\n");
		trajectory.RunOptimization(0, trajectory.GetNumFrames());
	}

    printf("All done. Number of points on trajectory: %d\n", trajectory.GetNumFrames());
    const float fps_all = (float)trajectory.GetNumFrames() / (((float)cv::getTickCount() - start_t) / (float)cv::getTickFrequency());
    printf("Overall framerate: %.3f\n", fps_all);

    sprintf(buffer, "%s/result_trajectory.ply", argv[1]);
    IMUProject::WriteToPly(std::string(buffer), trajectory.GetPositions().data(),
                           trajectory.GetOrientations().data(), trajectory.GetNumFrames(),
                           true, Eigen::Vector3i(0, 0, 255), 0, 0, 0);

	sprintf(buffer, "%s/tango_trajectory.ply", argv[1]);
	IMUProject::WriteToPly(std::string(buffer), dataset.GetPosition().data(),
	                       dataset.GetOrientation().data(), dataset.GetPosition().size(),
	                       true, Eigen::Vector3i(255, 0, 0), 0, 0, 0);

	LOG(INFO) << "Optimized trajectory written to " << buffer;

	if(FLAGS_mapinfo_path == "default"){
		sprintf(buffer, "%s/map.txt", argv[1]);
	}else{
		sprintf(buffer, "%s/%s", argv[1], FLAGS_mapinfo_path.c_str());
	}
	ifstream map_in(buffer);
	if(map_in.is_open()){
		LOG(INFO) << "Found map info file, creating overlay";
		string line;
		map_in >> line;
		sprintf(buffer, "%s/%s", argv[1], line.c_str());

		cv::Mat map_img = cv::imread(buffer);
		CHECK(map_img.data) << "Can not open image file: " << buffer;

		Eigen::Vector2d sp1, sp2;
		Eigen::Vector3d op1(0, 0, 0), op2(0, 0, 0);
		double scale_length;

		map_in >> sp1[0] >> sp1[1] >> sp2[0] >> sp2[1];
		map_in >> scale_length;
		map_in >> op1[0] >> op1[1] >> op2[0] >> op2[1];

		Eigen::Vector2d start_pix(op1[0], op1[1]);

		const double pixel_length = scale_length / (sp2 - sp1).norm();

		IMUProject::TrajectoryOverlay(pixel_length, start_pix, op2-op1, trajectory.GetPositions(),
		                              Eigen::Vector3i(255, 0, 0), map_img);

		IMUProject::TrajectoryOverlay(pixel_length, start_pix, op2-op1, dataset.GetPosition(),
		                              Eigen::Vector3i(0, 0, 255), map_img);
		sprintf(buffer, "%s/overlay.png", argv[1]);
		cv::imwrite(buffer, map_img);
	}



    return 0;
}