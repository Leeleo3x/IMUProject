//
// Created by yanhang on 3/19/17.
//

#ifndef PROJECT_MAIN_WINDOW_H
#define PROJECT_MAIN_WINDOW_H

#include "renderable.h"

#include <memory>

#include <QTimerEvent>
#include <QOpenGLWidget>
#include <opencv2/opencv.hpp>

#include <utility/data_io.h>

namespace IMUProject {

    class MainWidget: public QOpenGLWidget, protected QOpenGLFunctions{
        Q_OBJECT
    public:
        explicit MainWidget(const std::string& path, QWidget* parent = 0);
        ~MainWidget(){
	        FreeResource();
        }

    public slots:
        void Start();
        void Stop();

    protected:
        void initializeGL() Q_DECL_OVERRIDE;
        void resizeGL(int w, int h) Q_DECL_OVERRIDE;
        void paintGL() Q_DECL_OVERRIDE;

        void timerEvent(QTimerEvent* event) Q_DECL_OVERRIDE;

    private:
	    void InitShaders();
        void AllocateRecourse();
        void FreeResource();

	    std::shared_ptr<Canvas> canvas_;
	    std::shared_ptr<Navigation> navigation_;
    };

}//namespace IMUProject

#endif //PROJECT_MAIN_WINDOW_H
