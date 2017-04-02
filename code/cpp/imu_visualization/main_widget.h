//
// Created by Yan Hang on 3/31/17.
//

#ifndef PROJECT_MAIN_WIDGET_H
#define PROJECT_MAIN_WIDGET_H

#include "renderable.h"

#include <QTimerEvent>
#include <QBasicTimer>
#include <QKeyEvent>
#include <QOpenGLWidget>

#include <utility/data_io.h>

namespace IMUProject {
    class MainWidget: public QOpenGLWidget, protected QOpenGLFunctions{
        Q_OBJECT
    public:
        explicit MainWidget(const std::string& path,
                            const int graph_width,
                            const int graph_height,
                            const int frame_interval = 5,
                            QWidget* parent = 0);
        ~MainWidget(){

        }

    protected:
        void initializeGL() Q_DECL_OVERRIDE;
        void resizeGL(int w, int h) Q_DECL_OVERRIDE{
            glViewport(0, 0, w, h);
        }

        void paintGL() Q_DECL_OVERRIDE;
        void keyPressEvent(QKeyEvent* e) Q_DECL_OVERRIDE;
        void timerEvent(QTimerEvent* event) Q_DECL_OVERRIDE;

    private:
        std::vector<std::shared_ptr<GraphRenderer> > graph_renderers_;
        QBasicTimer timer_;

	    int counter_;
        std::vector<double> ts_;
        std::vector<Eigen::Vector3d> data_;

        const int graph_width_;
        const int graph_height_;
        const int frame_interval_;
    };

}// namespace IMUProject
#endif //PROJECT_MAIN_WIDGET_H
