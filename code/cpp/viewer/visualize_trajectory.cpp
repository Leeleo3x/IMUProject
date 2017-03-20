//
// Created by yanhang on 2/2/17.
//

#include <iostream>

#include "renderable.h"
#include "main_widget.h"

#include <gflags/gflags.h>
#include <glog/logging.h>
#include <opencv2/opencv.hpp>

#include <QApplication>
#include <QBoxLayout>
#include <QLabel>

#ifndef QT_NO_OPENGL
#include <QWidget>
#endif

#include <utility/data_io.h>

int main(int argc, char** argv) {
	google::InitGoogleLogging(argv[0]);
	CHECK_GE(argc, 2) << "Usage: ./VisualizeTrajectory <path-to-csv-file>";
	google::ParseCommandLineFlags(&argc, &argv, true);

	QApplication app(argc, argv);
	app.setApplicationName("IMU Trajectory Viewer");

#ifndef QT_NO_OPENGL
	QWidget main_window;
	main_window.resize(800, 800);

	IMUProject::MainWidget *main_widget = new IMUProject::MainWidget(std::string(argv[1]));
	QHBoxLayout *layout = new QHBoxLayout();
	layout->addWidget(main_widget);

	main_window.setLayout(layout);
	main_window.show();
#else
	QLabel note("OpenGL support required")
	note.show();
#endif

	return app.exec();
}
