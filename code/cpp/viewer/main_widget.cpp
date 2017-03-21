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
		setFocusPolicy(Qt::StrongFocus);
        canvas_.reset(new Canvas(canvas_width, canvas_height));
		navigation_.reset(new Navigation(50.f, (float)width(), (float)height()));

        IMUDataset dataset(path);
        for(auto i=0; i<dataset.GetPosition().size(); i+=3){
            gt_pos_.push_back(dataset.GetPosition()[i]);
            gt_orientation_.push_back(dataset.GetOrientation()[i]);
        }
        AdjustPositionToCanvas(gt_pos_, canvas_width, canvas_height);

        gt_trajectory_.reset(new OfflineTrajectory(gt_pos_, Eigen::Vector3f(1.0, 0.0, 0.0)));
        gt_trajectory_->SetRenderLength(0);

		view_frustum_.reset(new ViewFrustum(1.0));
		camera_mode_ = BACK;
	}

	void MainWidget::initializeGL() {
		initializeOpenGLFunctions();

		canvas_->InitGL();
        gt_trajectory_->InitGL();
		view_frustum_->InitGL();

		glClearColor(1.f, 1.f, 1.f, 1.f);
		glEnable(GL_DEPTH);
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
		view_frustum_->Render(*navigation_);
        glFlush();
	}

	void MainWidget::timerEvent(QTimerEvent *event) {
        if(render_count_ >= (int)gt_pos_.size()){
	        render_count_ = 0;
        }
		render_count_++;
        gt_trajectory_->SetRenderLength(render_count_);

		view_frustum_->UpdateCameraPose(gt_pos_[render_count_], gt_orientation_[render_count_]);

		if(camera_mode_ == BACK){
			navigation_->UpdateCameraBack(gt_pos_[render_count_], gt_orientation_[render_count_]);
		}else if(camera_mode_ == CENTER){
			navigation_->UpdateCameraCenter(gt_pos_[render_count_],
			                                Eigen::Vector3d(0.0, 5.0f, -0.5 * (double)canvas_->GetHeight()));
		}

        update();
	}

	void MainWidget::keyPressEvent(QKeyEvent *e) {
		switch(e->key()){
			case(Qt::Key_C):{
				if(camera_mode_ == BACK){
					camera_mode_ = CENTER;
				}else{
					camera_mode_ = BACK;
				}
				break;
			}

			case(Qt::Key_S):{
				Start();
				break;
			}

			case(Qt::Key_P):{
				Stop();
				break;
			}

			default:
				break;
		}
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
            pos[1] += (double)canvas_height / 2.0;
        }
    }
}//namespace IMUProject