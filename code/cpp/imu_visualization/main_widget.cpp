//
// Created by Yan Hang on 3/31/17.
//

#include "main_widget.h"

#include <fstream>

namespace IMUProject{

    MainWidget::MainWidget(const std::string &path, const int graph_width, const int graph_height,
                           const int frame_interval, QWidget *parent)
            :graph_width_(graph_width), graph_height_(graph_height), frame_interval_(frame_interval), counter_(0),
             is_rendering_(false){
        setFocusPolicy(Qt::StrongFocus);

        const float x_scale = 750 / frame_interval_;

        graph_renderers_.emplace_back(new GraphRenderer(Eigen::Vector3f(1.0f, 0.0f, 0.0f), graph_width_, graph_height_));
        graph_renderers_.emplace_back(new GraphRenderer(Eigen::Vector3f(0.0f, 1.0f, 0.0f), graph_width_, graph_height_));
        graph_renderers_.emplace_back(new GraphRenderer(Eigen::Vector3f(0.0f, 0.0f, 1.0f), graph_width_, graph_height_));

        constexpr double nano_to_sec = 1e09;

        std::ifstream data_in(path.c_str());
        CHECK(data_in.is_open()) << "Can not open file " << path;

        std::string line;

        // skip the first line
        std::getline(data_in, line);

        int count = 0;
        double v0, v1, v2, v3;
        while(data_in >> v0){
            data_in >> v1 >> v2 >> v3;
            if(count % frame_interval_ == 0) {
                ts_.push_back(v0 / nano_to_sec);
                data_.push_back(Eigen::Vector3d(v1, v2, v3));
            }
            ++count;
        }

	    // apply low pass filter
	    const double alpha = 0.9;
	    data_[0] *= 1.0 - alpha;
	    for(auto i=1; i<data_.size(); ++i){
		    data_[i] = alpha * data_[i-1] + (1.0 - alpha) * data_[i];
	    }
        // normalize data
        const double init_t = ts_[0];
        double max_v = -1;
        for(auto i=0; i<data_.size(); ++i){
            for(auto j=0; j<3; ++j){
                max_v = std::max(max_v, std::fabs(data_[i][j]));
            }
        }

	    // scale data and apply low pass filter

        for(auto i=0; i<ts_.size(); ++i){
            ts_[i] = (ts_[i] - init_t) * x_scale;
            data_[i] = data_[i] / max_v * graph_height / 2.0;
        }

        LOG(INFO) << count << " data read";
    }

    void MainWidget::initializeGL() {
        initializeOpenGLFunctions();
        for(auto& graph: graph_renderers_){
            graph->initGL();
        }

        glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
        glViewport(0, 0, graph_width_, graph_height_);

        timer_.start(frame_interval_ * 5, this);
    }

    void MainWidget::paintGL() {
        glClear(GL_COLOR_BUFFER_BIT);
        for(auto& graph: graph_renderers_){
            graph->Render();
        }
	    graph_renderers_[0]->Render();

        glFlush();
    }

    void MainWidget::keyPressEvent(QKeyEvent *e) {
        switch(e->key()) {
            case (Qt::Key_S): {
                is_rendering_ = !is_rendering_;
                break;
            }
            default:
                break;
        }
    }

    void MainWidget::timerEvent(QTimerEvent *event) {
	    if(counter_ == ts_.size()){
		    counter_ = 0;
		    timer_.stop();
		    return;
	    }
        if(is_rendering_) {
            for (auto i = 0; i < 3; ++i) {
                graph_renderers_[i]->AppendData(ts_[counter_], data_[counter_][i]);
            }
            counter_++;
        }
        update();
    }

} // namespace IMUProject