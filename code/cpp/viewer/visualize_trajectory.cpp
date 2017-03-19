//
// Created by yanhang on 2/2/17.
//

#include <iostream>

#include "renderable.h"

#include <gflags/gflags.h>
#include <glog/logging.h>
#include <opencv2/opencv.hpp>

#include <QApplication>
#include <QBoxLayout>

#ifndef QT_NO_OPENGL
#include <QWidget>
#endif

#include <utility/data_io.h>

int main(int argc, char** argv){
    google::InitGoogleLogging(argv[0]);
    CHECK_GE(argc, 2) << "Usage: ./VisualizeTrajectory <path-to-csv-file>";
    google::ParseCommandLineFlags(&argc, &argv, true);

    QApplication app(argc, argv);
    app.setApplicationName("IMU Trajectory Viewer");

#ifndef QT_NO_OPENGL
    QWidget main_window;
    main_window.resize(1280, 720);

#else
#endif
