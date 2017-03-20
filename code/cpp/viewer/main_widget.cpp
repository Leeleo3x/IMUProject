//
// Created by yanhang on 3/19/17.
//

#include "main_widget.h"

using namespace std;

namespace IMUProject{

	MainWidget::MainWidget(const std::string &path,
                           const int canvas_width,
                           const int canvas_height,
                           QWidget *parent): render_count_(0){
		navigation_.reset(new Navigation());
        canvas_.reset(new Canvas(canvas_width, canvas_height));

        IMUDataset dataset(path);
        for(auto i=0; i<dataset.GetPosition().size(); i+=3){
            gt_pos_.push_back(dataset.GetPosition()[i]);
            gt_orientation_.push_back(dataset.GetOrientation()[i]);
        }
        AdjustPositionToCanvas(gt_pos_, canvas_width, canvas_height);

        gt_trajectory_.reset(new OfflineTrajectory(gt_pos_, Eigen::Vector3f(1.0, 0.0, 0.0)));
        gt_trajectory_->SetRenderLength(0);

        // Set a toy camera for debugging
        QMatrix4x4 projection;
        projection.setToIdentity();
        projection.perspective(60.0f, 1.0f, 0.001f, 500.0f);
        navigation_->SetProjection(projection);
	}

	void MainWidget::initializeGL() {
		initializeOpenGLFunctions();

		canvas_->InitGL();
        gt_trajectory_->InitGL();

		glClearColor(1.f, 1.f, 1.f, 1.f);
		glEnable(GL_DEPTH);
		//glClearDepth(0.0);
		glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

		AllocateRecourse();

		glViewport(0, 0, width(), height());

        render_timer_.start(20, this);
	}

	void MainWidget::resizeGL(int w, int h) {
        glViewport(0, 0, w, h);
	}

	void MainWidget::paintGL() {
		canvas_->Render(*navigation_);
        gt_trajectory_->Render(*navigation_);
        glFlush();
	}

	void MainWidget::timerEvent(QTimerEvent *event) {
        render_count_++;
        if(render_count_ == (int)gt_pos_.size()){
            return;
        }
        gt_trajectory_->SetRenderLength(render_count_);
//        QMatrix4x4 modelview;
//        modelview.setToIdentity();
//        modelview.lookAt(QVector3D(0, 5.0f, 0.5f * canvas_->GetHeight()),
//                         QVector3D((float)gt_pos_[render_count_][0], (float)gt_pos_[render_count_][2], -1*(float)gt_pos_[render_count_][1]),
//                         QVector3D(0.0f, 1.0f, 0.0f));
//        navigation_->SetModelView(modelview);

        navigation_->UpdateCamera(gt_pos_[render_count_], gt_orientation_[render_count_]);
        update();
	}

	void MainWidget::Start() {

	}

	void MainWidget::Stop(){

	}

	void MainWidget::AllocateRecourse() {

	}

	void MainWidget::FreeResource() {

	}

    void AdjustPositionToCanvas(std::vector<Eigen::Vector3d>& position,
                                const int canvas_width, const int canvas_height){
        CHECK_GT(position.size(), 0);
        Eigen::Vector3d centroid = std::accumulate(position.begin(), position.end(), Eigen::Vector3d(0, 0, 0))
                                   / (double)position.size();
        double max_distance = -1;
        for(const auto& pos: position){
            const double d = (pos-centroid).norm();
            if(d > max_distance){
                max_distance = d;
            }
        }
        const double larger_dim = std::max((double)canvas_width/2.0, (double)canvas_height/2.0);
        for(auto& pos: position){
            pos = (pos - centroid) / max_distance * larger_dim * 0.7;
            pos[1] -= (double)canvas_height / 2.0;
        }
    }
}//namespace IMUProject