//
// Created by Yan Hang on 3/31/17.
//

#include "renderable.h"

namespace IMUProject{

    const float GraphRenderer::z_pos_ = 0.5f;

    GraphRenderer::GraphRenderer(const Eigen::Vector3f color,
                                 const int graph_width,
                                 const int graph_height,
                                 const int kMaxPoints)
            :graph_width_(graph_width), graph_height_(graph_height), kMaxPoints_(kMaxPoints),
             left_border_(10.0f), render_pointer_(0), insert_pointer_(0){

        const float wf = (float)graph_width_;
        const float hf = (float)graph_height_;

        projection_.setToIdentity();
        projection_.ortho(0.0f, wf, hf/2.0f, -hf/2.0f, 0.0f, 1.0f);
        modelview_.setToIdentity();
        modelview_.lookAt(QVector3D(0.0f, 0.0f, 0.0f), QVector3D(0.0f, 0.0f, z_pos_), QVector3D(0.0f, -1.0f, 0.0f));

        grid_vertex_data_ = {left_border_, -hf/2.0f, z_pos_, left_border_, hf/2.0f, z_pos_,
                             left_border_, 0.0f, z_pos_, wf, 0.0f, z_pos_};
        grid_color_data_[0] = 0.8f;
        grid_color_data_[1] = 0.8f;
        grid_color_data_[2] = 0.8f;

        line_color_data_[0] = color[0];
        line_color_data_[1] = color[1];
        line_color_data_[2] = color[2];

	    constexpr int capacity_ratio = 10;
        vertex_data_.resize((size_t)kMaxPoints_ * 3 * capacity_ratio, 0.0f);
	    ts_.resize((size_t)kMaxPoints_ * capacity_ratio, 0.0);
    }

    void GraphRenderer::AppendData(const double t, const double v){
	    CHECK(line_vertex_buffer_.bind()) << "Can not bind line_vertex_buffer";
	    GLfloat* buffer_data = (GLfloat*)line_vertex_buffer_.map(QOpenGLBuffer::ReadWrite);

	    CHECK(buffer_data) << "Can not map line_vertex_buffer";
	    if(insert_pointer_ >= vertex_data_.size()) {
		    const int sid = insert_pointer_ - kMaxPoints_ * 3;
		    for (auto i = sid; i < vertex_data_.size(); ++i) {
			    ts_[(i - sid) / 3] = ts_[i / 3];
			    buffer_data[i - sid] = buffer_data[i];
		    }
		    insert_pointer_ = kMaxPoints_ * 3;
	    }
	    ts_[insert_pointer_ / 3] = t;
	    buffer_data[insert_pointer_] = (float)t;
	    buffer_data[insert_pointer_ + 1] = (float)v;
	    buffer_data[insert_pointer_ + 2] = z_pos_;

	    insert_pointer_ += 3;
	    render_pointer_ = std::max(insert_pointer_ - kMaxPoints_ * 3, 0);

	    float start_t = (float)ts_[render_pointer_ / 3];

	    for(auto i=render_pointer_; i < insert_pointer_; i+=3){
		    buffer_data[i] = (float)ts_[i/3] - start_t + left_border_;
	    }

	    line_vertex_buffer_.unmap();
	    line_vertex_buffer_.release();
    }

    void GraphRenderer::initGL(){
        initializeOpenGLFunctions();

        const char *v_shader = "uniform mat4 m_mat;\n"
                "uniform mat4 p_mat;\n"
                "attribute vec3 pos;\n"
                "void main(void){\n"
                "gl_Position = p_mat * m_mat * vec4(pos, 1.0);\n"
                "}\n";

        const char* f_shader = "uniform vec4 color;\n"
                "void main(void){\n"
                "gl_FragColor = color;\n"
                "}\n";

        shader_.reset(new QOpenGLShaderProgram());
        shader_->addShaderFromSourceCode(QOpenGLShader::Vertex, v_shader);
        shader_->addShaderFromSourceCode(QOpenGLShader::Fragment, f_shader);
        CHECK(shader_->link()) << "Unable to compile and link shaders";
        CHECK(shader_->bind()) << "Unable to bind shader";
        shader_->enableAttributeArray("pos");
        shader_->release();

        grid_vertex_buffer_.create();
        grid_vertex_buffer_.bind();
        grid_vertex_buffer_.allocate(grid_vertex_data_.data(), (int)grid_vertex_data_.size() * sizeof(GLfloat));
        grid_vertex_buffer_.release();

	    line_vertex_buffer_.create();
	    line_vertex_buffer_.bind();
	    line_vertex_buffer_.setUsagePattern(QOpenGLBuffer::DynamicDraw);
	    line_vertex_buffer_.allocate(vertex_data_.data(), (int)vertex_data_.size() * sizeof(GLfloat));
	    line_vertex_buffer_.release();
    }

    void GraphRenderer::Render(){
        // first render grid lines
        const float wf = (float) graph_width_;
        const float hf = (float) graph_height_;

        CHECK(shader_->bind());

        // draw grid
        shader_->setUniformValue("p_mat", projection_);
        shader_->setUniformValue("m_mat", modelview_);
        shader_->setUniformValue("color", grid_color_data_[0], grid_color_data_[1], grid_color_data_[2], 1.0f);
        grid_vertex_buffer_.bind();
        shader_->setAttributeBuffer("pos", GL_FLOAT, 0, 3);
        glLineWidth(3.0f);
        glDrawArrays(GL_LINES, 0, 4);
        grid_vertex_buffer_.release();

        //draw graph
        CHECK_GE(render_pointer_, 0);
        shader_->setUniformValue("m_mat", modelview_);
        shader_->setUniformValue("color", line_color_data_[0], line_color_data_[1], line_color_data_[2], 1.0f);
	    line_vertex_buffer_.bind();
        shader_->setAttributeBuffer("pos", GL_FLOAT, 0, 3);
        glLineWidth(2.0f);
//         glDrawArrays(GL_LINE_STRIP, start_pos * 3, (GLsizei)((int)vertex_data_.size() - start_pos * 3));
        glDrawArrays(GL_LINE_STRIP, render_pointer_ / 3, (insert_pointer_ - render_pointer_) / 3);
	    line_vertex_buffer_.release();
        shader_->release();
    }

}// namespace IMUProject