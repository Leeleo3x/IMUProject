//
// Created by Yan Hang on 3/31/17.
//

#ifndef PROJECT_RENDERABLE_H
#define PROJECT_RENDERABLE_H

#include <memory>
#include <vector>

#include <QOpenGLFunctions>
#include <QOpenGLBuffer>
#include <QOpenGLShaderProgram>
#include <QMatrix4x4>
#include <glog/logging.h>
#include <Eigen/Eigen>

namespace IMUProject {

    class GraphRenderer: protected QOpenGLFunctions{
    public:
        GraphRenderer(const Eigen::Vector3f color,
                      const int graph_width,
                      const int graph_height,
                      const int kMaxPoints=200);

        void AppendData(const double t, const double v);

        void initGL();
        void Render();

    private:
        const int kMaxPoints_;
        const int graph_width_;
        const int graph_height_;
        int render_pointer_;
	    int insert_pointer_;

	    std::vector<double> ts_;
        std::vector<GLfloat> vertex_data_;
        std::shared_ptr<QOpenGLShaderProgram> shader_;

        std::vector<GLfloat> grid_vertex_data_;
        GLfloat grid_color_data_[3];
        QOpenGLBuffer grid_vertex_buffer_;
	    QOpenGLBuffer line_vertex_buffer_;

	    GLfloat line_color_data_[3];
        QMatrix4x4 projection_;
        QMatrix4x4 modelview_;

        const float left_border_;
        static const float z_pos_;
    };

}// namespace IMUProject


#endif //PROJECT_RENDERABLE_H
