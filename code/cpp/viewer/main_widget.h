//
// Created by yanhang on 3/19/17.
//

#ifndef PROJECT_MAIN_WINDOW_H
#define PROJECT_MAIN_WINDOW_H

#include "renderable.h"

#include <memory>

#include <QTimerEvent>
#include <QBasicTimer>
#include <QKeyEvent>
#include <QOpenGLWidget>

#include <opencv2/opencv.hpp>

#include <utility/data_io.h>

namespace IMUProject {

	class MainWidget: public QOpenGLWidget, protected QOpenGLFunctions{
        Q_OBJECT
    public:
        explicit MainWidget(const std::string& path,
							const int canvas_width = 50,
							const int convas_height = 50,
							QWidget* parent = 0);
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
	    void keyPressEvent(QKeyEvent *e) Q_DECL_OVERRIDE;

        void timerEvent(QTimerEvent* event) Q_DECL_OVERRIDE;

    private:
        void AllocateRecourse();
        void FreeResource();

        std::vector<Eigen::Vector3d> gt_pos_;
        std::vector<Eigen::Quaterniond> gt_orientation_;

	    std::shared_ptr<Canvas> canvas_;
        std::shared_ptr<OfflineTrajectory> gt_trajectory_;
	    std::shared_ptr<ViewFrustum> view_frustum_;

        int render_count_;
		CameraMode camera_mode_;

        QBasicTimer render_timer_;

	    std::shared_ptr<Navigation> navigation_;
    };


    void AdjustPositionToCanvas(std::vector<Eigen::Vector3d>& position,
                                const int canvas_width, const int canvas_height);

}//namespace IMUProject

#endif //PROJECT_MAIN_WINDOW_H
