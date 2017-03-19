//
// Created by yanhang on 3/19/17.
//

#ifndef PROJECT_MAIN_WINDOW_H
#define PROJECT_MAIN_WINDOW_H

#include "renderable.h"

#include <QTimerEvent>
#include <QOpenGLWidget>
#include <opencv2/opencv.hpp>

#include <utility/data_io.h>

namespace IMUProject {

    class MainWidget: public QOpenGLWidget, protected QOpenGLFunctions{
        Q_OBJECT
    public:
        explicit MainWidget(const std::string& path, QWidget* parent = 0);
        ~MainWidget();

    public slots:
        void start();
        void stop();

    protected:
        void initializedGL() Q_DECL_OVERRIDE;
        void resizeGL(int w, int h) Q_DECL_OVERRIDE;
        void paintGL() Q_DECL_OVERRIDE;

        void timerEvent(QTimerEvent* event) Q_DECL_OVERRIDE;

    private:
        void InitalizeShader();
        void AllocateRecourse();
        void FreeResource();
    };

}//namespace IMUProject

#endif //PROJECT_MAIN_WINDOW_H
