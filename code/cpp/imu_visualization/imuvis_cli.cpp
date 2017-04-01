//
// Created by Yan Hang on 3/30/17.
//

#include "main_widget.h"

#include <iostream>

#include <gflags/gflags.h>
#include <glog/logging.h>

#include <QApplication>
#include <QBoxLayout>
#include <QLabel>

#ifndef QT_NO_OPENGL
#include <QWidget>
#endif
#include <utility/data_io.h>

DEFINE_int32(width, 800, "width of the graph");
DEFINE_int32(height, 400, "height of the graph");

int main(int argc, char** argv){
    google::InitGoogleLogging(argv[0]);
    CHECK_GE(argc, 2) << "Usage: ./IMU_visualization <path-to-csv-file>";
    google::ParseCommandLineFlags(&argc, &argv, true);

    QApplication app(argc, argv);
    app.setApplicationName("IMU visualization");

#ifndef QT_NO_OPENGL
    QWidget main_window;
    main_window.resize(FLAGS_width, FLAGS_height);

    IMUProject::MainWidget *main_widget = new IMUProject::MainWidget(std::string(argv[1]), FLAGS_width, FLAGS_height);
    QHBoxLayout *layout = new QHBoxLayout();
    layout->addWidget(main_widget);

    main_window.setLayout(layout);
    main_window.show();
#else
    QLabel note("OpenGL support required")
	note.show();
#endif

    return app.exec();

	return 0;
}