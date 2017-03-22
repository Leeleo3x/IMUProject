//
// Created by yanhang on 3/19/17.
//

#include "main_widget.h"

using namespace std;

namespace IMUProject{

	MainWidget::MainWidget(const std::string &path,
                           const int canvas_width,
                           const int canvas_height,
                           QWidget *parent): render_count_(0),
	                                         panel_border_margin_(10), panel_size_(300){
		setFocusPolicy(Qt::StrongFocus);
        canvas_.reset(new Canvas(canvas_width, canvas_height));
		navigation_.reset(new Navigation(50.f, (float)width(), (float)height()));

        IMUDataset dataset(path);
        for(auto i=0; i<dataset.GetPosition().size(); i+=frame_interval_){
            gt_pos_.push_back(dataset.GetPosition()[i]);
            gt_orientation_.push_back(dataset.GetOrientation()[i]);
	        ts_.push_back(dataset.GetTimeStamp()[i]);
        }
        AdjustPositionToCanvas(gt_pos_, canvas_width, canvas_height);

        gt_trajectory_.reset(new OfflineTrajectory(gt_pos_, Eigen::Vector3f(1.0, 0.0, 0.0)));
        gt_trajectory_->SetRenderLength(0);

        view_frustum_.reset(new ViewFrustum(1.0));

        speed_panel_.reset(new OfflineSpeedPanel(gt_pos_, gt_orientation_, gt_orientation_[0]));
		camera_mode_ = BACK;
	}

	void MainWidget::initializeGL() {
		initializeOpenGLFunctions();

		canvas_->InitGL();
        gt_trajectory_->InitGL();
		view_frustum_->InitGL();
        speed_panel_->InitGL();

        UpdateCameraInfo(0);

		glClearColor(1.f, 1.f, 1.f, 1.f);
		glClear(GL_COLOR_BUFFER_BIT);
		glViewport(0, 0, width(), height());
	}

	void MainWidget::resizeGL(int w, int h) {
        glViewport(0, 0, w, h);
	}

	void MainWidget::paintGL() {
		canvas_->Render(*navigation_);
        gt_trajectory_->Render(*navigation_);
		view_frustum_->Render(*navigation_);

        // Render the speed panel
        glViewport(panel_border_margin_, panel_border_margin_, width() / 4, width() / 4);
        speed_panel_->Render(*navigation_);
        glViewport(0, 0, width(), height());
        glFlush();
	}

    void MainWidget::UpdateCameraInfo(const int ind){
        gt_trajectory_->SetRenderLength(ind);

        // update camera pose and heading
        view_frustum_->UpdateCameraPose(gt_pos_[ind], gt_orientation_[ind]);
        if(ind > 0){
            Eigen::Vector3d forward_dir = (gt_pos_[ind] - gt_pos_[ind - 1])
                                          / (ts_[ind] - ts_[ind-1]);
            forward_dir[2] = 0.0;
            Eigen::Vector3d device_dir = gt_orientation_[ind] * Eigen::Vector3d(0, 0, -1);
            device_dir[2] = 0.0;
            speed_panel_->UpdateDirection(forward_dir, device_dir);
        }else{
            speed_panel_->UpdateDirection(Eigen::Vector3d(0, 1, 0), Eigen::Vector3d(0, 1, 0));
        }

        if(camera_mode_ == BACK){
            navigation_->UpdateCameraBack(gt_pos_[ind], gt_orientation_[ind]);
        }else if(camera_mode_ == CENTER){
            navigation_->UpdateCameraCenter(gt_pos_[ind],
                                            Eigen::Vector3d(0.0, 5.0f, 0.0f));
        }
    }

	void MainWidget::timerEvent(QTimerEvent *event) {
        if(render_count_ >= (int)gt_pos_.size()){
	        render_count_ = 0;
        }
        UpdateCameraInfo(render_count_);
		render_count_++;
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
                UpdateCameraInfo(render_count_);
                update();
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
        if(render_timer_.isActive()){
            render_timer_.stop();
        }
        render_timer_.start(frame_interval_ * 5, this);
	}

	void MainWidget::Stop(){
        if(render_timer_.isActive()){
            render_timer_.stop();
        }
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
        }
    }
}//namespace IMUProject