//
// Created by yanhang on 3/19/17.
//

#include "renderable.h"

#include <glog/logging.h>

namespace IMUProject{

    ////////////////////////////////////
    // Implementation of canvas
    Canvas::Canvas(const float width, const float height,
                   const float grid_size, const Eigen::Vector3f grid_color, const cv::Mat *texture)
			:width_(width), height_(height), line_alpha_(0.5f){
	    is_shader_init_ = false;
	    vertex_data_ = {-width/2.0f, 0.0f, -height,
	                    width/2.0f, 0.0f, -height,
	                    width/2.0f, 0.0f, 0,
	                    -width/2.0f, 0.0f, 0};
	    index_data_ = {0, 1, 2, 2, 3, 0};
	    texcoord_data_ = {0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0};

        // Add grid lines
        GLuint p_counter = 0;
        for(float x=-width/2.0f + grid_size; x <= width/2.0f - grid_size; x += grid_size){
            grid_vertex_data_.push_back(x);
            grid_vertex_data_.push_back(0.0f);
            grid_vertex_data_.push_back(-height_);
            grid_vertex_data_.push_back(x);
            grid_vertex_data_.push_back(0.0f);
            grid_vertex_data_.push_back(0);

            grid_color_data_.push_back(grid_color[0]);
            grid_color_data_.push_back(grid_color[1]);
            grid_color_data_.push_back(grid_color[2]);
            grid_color_data_.push_back(line_alpha_);
	        grid_color_data_.push_back(grid_color[0]);
	        grid_color_data_.push_back(grid_color[1]);
	        grid_color_data_.push_back(grid_color[2]);
	        grid_color_data_.push_back(line_alpha_);


	        grid_index_data_.push_back(p_counter);
            grid_index_data_.push_back(p_counter+1);
            p_counter += (GLuint)2;
        }
        for(float y=-height + grid_size; y < 0; y += grid_size){
            grid_vertex_data_.push_back(width/2.0f);
            grid_vertex_data_.push_back(0.0f);
            grid_vertex_data_.push_back(y);
            grid_vertex_data_.push_back(-width/2.0f);
            grid_vertex_data_.push_back(0.0f);
            grid_vertex_data_.push_back(y);

	        //printf("(%.6f,%.6f,%.6f), (%.6f,%.6f,%.6f)\n", -width/2.0f, 0.0f, y, width/2.0f, 0.0f, y);
            grid_color_data_.push_back(grid_color[0]);
            grid_color_data_.push_back(grid_color[1]);
            grid_color_data_.push_back(grid_color[2]);
            grid_color_data_.push_back(line_alpha_);
	        grid_color_data_.push_back(grid_color[0]);
	        grid_color_data_.push_back(grid_color[1]);
	        grid_color_data_.push_back(grid_color[2]);
	        grid_color_data_.push_back(line_alpha_);

            grid_index_data_.push_back(p_counter);
            grid_index_data_.push_back(p_counter+1);
            p_counter += (GLuint)2;
        }

	    if(texture == nullptr){
		    texture_img_.load(":/images/iccv_texture.png");
		    printf("Default texture image loaded. Width: %d, height: %d\n",
		           texture_img_.width(), texture_img_.height());
	    }else{
		    QImage::Format tex_format = QImage::Format_RGB888;
		    CHECK(texture->data) << "Empty texture image";
		    if(texture->type() == CV_8UC3){
			    tex_format = QImage::Format_RGB888;
		    }else if(texture->type() == CV_8UC4){
			    tex_format = QImage::Format_RGBA8888;
		    }else{
			    CHECK(true) << "Unsupported pixel format:" << texture->type();
		    }

		    texture_img_ = QImage(texture->data, texture->cols, texture->rows, tex_format);
	    }

    }

	Canvas::~Canvas(){
		glDeleteBuffers(1, &vertex_buffer_);
		glDeleteBuffers(1, &index_buffer_);
		glDeleteBuffers(1, &texcoord_buffer_);
        glDeleteBuffers(1, &grid_vertex_buffer_);
        glDeleteBuffers(1, &grid_color_buffer_);
        glDeleteBuffers(1, &grid_index_buffer_);
	}

	void Canvas::InitGL(){
        initializeOpenGLFunctions();
        tex_shader_.reset(new QOpenGLShaderProgram());
		CHECK(tex_shader_->addShaderFromSourceFile(QOpenGLShader::Vertex, ":/shaders/canvas_shader.vert"));
		CHECK(tex_shader_->addShaderFromSourceFile(QOpenGLShader::Fragment, ":/shaders/canvas_shader.frag"));
		CHECK(tex_shader_->link()) << "Canvas: can not link texture shader";
		CHECK(tex_shader_->bind()) << "Canvas: can not bind texture shader";

		tex_shader_->enableAttributeArray("pos");
		tex_shader_->enableAttributeArray("texcoord");
		tex_shader_->release();

        line_shader_.reset(new QOpenGLShaderProgram());
        CHECK(line_shader_->addShaderFromSourceFile(QOpenGLShader::Vertex, ":/shaders/line_shader.vert"));
        CHECK(line_shader_->addShaderFromSourceFile(QOpenGLShader::Fragment, ":/shaders/line_shader.frag"));
        CHECK(line_shader_->link()) << "Canvas: can not link line shader";
        CHECK(line_shader_->bind());
        line_shader_->enableAttributeArray("pos");
        line_shader_->enableAttributeArray("v_color");
        line_shader_->release();

		is_shader_init_ = true;

        glGenBuffers(1, &vertex_buffer_);
        glBindBuffer(GL_ARRAY_BUFFER, vertex_buffer_);
        glBufferData(GL_ARRAY_BUFFER, vertex_data_.size() * sizeof(GLfloat),
                     vertex_data_.data(), GL_STATIC_DRAW);

        glGenBuffers(1, &texcoord_buffer_);
        glBindBuffer(GL_ARRAY_BUFFER, texcoord_buffer_);
        glBufferData(GL_ARRAY_BUFFER, texcoord_data_.size() * sizeof(GLfloat),
                     texcoord_data_.data(), GL_STATIC_DRAW);

        glGenBuffers(1, &index_buffer_);
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, index_buffer_);
        glBufferData(GL_ELEMENT_ARRAY_BUFFER, index_data_.size() * sizeof(GLuint),
                     index_data_.data(), GL_STATIC_DRAW);

        glGenBuffers(1, &grid_vertex_buffer_);
        glBindBuffer(GL_ARRAY_BUFFER, grid_vertex_buffer_);
        glBufferData(GL_ARRAY_BUFFER, grid_vertex_data_.size() * sizeof(GLfloat),
                     grid_vertex_data_.data(), GL_STATIC_DRAW);

        glGenBuffers(1, &grid_color_buffer_);
        glBindBuffer(GL_ARRAY_BUFFER, grid_color_buffer_);
        glBufferData(GL_ARRAY_BUFFER, grid_color_data_.size() * sizeof(GLfloat),
                     grid_color_data_.data(), GL_STATIC_DRAW);

        glGenBuffers(1, &grid_index_buffer_);
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, grid_index_buffer_);
        glBufferData(GL_ELEMENT_ARRAY_BUFFER, grid_index_data_.size() * sizeof(GLuint),
                     grid_index_data_.data(), GL_STATIC_DRAW);

        glEnable(GL_TEXTURE_2D);
        canvas_texture_.reset(new QOpenGLTexture(texture_img_));
        glBindTexture(GL_TEXTURE_2D, 0);
        glDisable(GL_TEXTURE_2D);
    }

    void Canvas::Render(const Navigation& navigation) {
	    CHECK(tex_shader_->bind());

	    glEnable(GL_TEXTURE_2D);
	    canvas_texture_->bind();
	    CHECK(canvas_texture_->isBound()) << "Can not bind canvas texture";
	    tex_shader_->setUniformValue("tex_sampler", 0);
	    tex_shader_->setUniformValue("m_mat", navigation.GetModelViewMatrix());
	    tex_shader_->setUniformValue("p_mat", navigation.GetProjectionMatrix());

	    glBindBuffer(GL_ARRAY_BUFFER, vertex_buffer_);
	    tex_shader_->setAttributeArray("pos", GL_FLOAT, 0, 3);
	    glBindBuffer(GL_ARRAY_BUFFER, texcoord_buffer_);
	    tex_shader_->setAttributeArray("texcoord", GL_FLOAT, 0, 2);

	    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, index_buffer_);
	    glDrawElements(GL_TRIANGLES, (GLsizei)index_data_.size(), GL_UNSIGNED_INT, 0);
	    glDisable(GL_TEXTURE_2D);
	    tex_shader_->release();

        glEnable(GL_BLEND);
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
        CHECK(line_shader_->bind());
        glLineWidth(1.0f);
        line_shader_->setUniformValue("m_mat", navigation.GetModelViewMatrix());
        line_shader_->setUniformValue("p_mat", navigation.GetProjectionMatrix());
        glBindBuffer(GL_ARRAY_BUFFER, grid_vertex_buffer_);
        line_shader_->setAttributeArray("pos", GL_FLOAT, 0, 3);
        glBindBuffer(GL_ARRAY_BUFFER, grid_color_buffer_);
        line_shader_->setAttributeArray("v_color", GL_FLOAT, 0, 4);
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, grid_index_buffer_);
        glDrawElements(GL_LINES, (GLsizei)grid_index_data_.size(), GL_UNSIGNED_INT, 0);
        line_shader_->release();
        glDisable(GL_BLEND);
    }

	///////////////////////////////////
	// Implementation of ViewFrustum
	ViewFrustum::ViewFrustum(const float length, const bool with_axes, const float default_height)
			:default_height_(default_height){
		vertex_data_ = {0.0f, 0.0f, 0.0f,
		                -length/2.0f, length/2.0f, -length * 0.8f,
		                length/2.0f, length/2.0f, -length * 0.8f,
		                length/2.0f, -length/2.0f, -length * 0.8f,
		                -length/2.0f, -length/2.0f, -length * 0.8f};
		index_data_ = {0, 1, 0, 2, 0, 3, 0, 4,
		               1, 2, 2, 3, 3, 4, 4, 1};
		color_data_ = {0.0f, 0.0f, 0.0f, 1.0f,
		               0.0f, 0.0f, 0.0f, 1.0f,
		               0.0f, 0.0f, 0.0f, 1.0f,
		               0.0f, 0.0f, 0.0f, 1.0f,
		               0.0f, 0.0f, 0.0f, 1.0f};

		if(with_axes){
			vertex_data_.insert(vertex_data_.end(), {0.0f, 0.0f, 0.0f,
			                                         length, 0.0f, 0.0f,
			                                         0.0f, 0.0f, 0.0f,
			                                         0.0f, length, 0.0f,
			                                         0.0f, 0.0f, 0.0f,
			                                         0.0f, 0.0f, length});
			index_data_.insert(index_data_.end(), {5,6,7,8,9,10});
			color_data_.insert(color_data_.end(), {1.0f, 0.0f, 0.0f, 1.0f,
			                                       1.0f, 0.0f, 0.0f, 1.0f,
			                                       0.0f, 1.0f, 0.0f, 1.0f,
			                                       0.0f, 1.0f, 0.0f, 1.0f,
			                                       0.0f, 0.0f, 1.0f, 1.0f,
			                                       0.0f, 0.0f, 1.0f, 1.0f});
		}

		float mat3[9] = {1.0, 0.0, 0.0,
		                 0.0, 0.0, 1.0,
		                 0.0, -1.0f, 0.0};
		QMatrix3x3 l_to_g(mat3);
		local_to_global_ = QQuaternion::fromRotationMatrix(l_to_g);

	}

	void ViewFrustum::InitGL(){
		initializeOpenGLFunctions();

		line_shader_.reset(new QOpenGLShaderProgram());
		CHECK(line_shader_->addShaderFromSourceFile(QOpenGLShader::Vertex, ":/shaders/line_shader.vert"));
		CHECK(line_shader_->addShaderFromSourceFile(QOpenGLShader::Fragment, ":/shaders/line_shader.frag"));
		CHECK(line_shader_->link()) << "ViewFrustum: can not link line shader";
		CHECK(line_shader_->bind());
		line_shader_->enableAttributeArray("pos");
		line_shader_->enableAttributeArray("v_color");
		line_shader_->release();

		is_shader_init_ = true;

		glGenBuffers(1, &vertex_buffer_);
		glBindBuffer(GL_ARRAY_BUFFER, vertex_buffer_);
		glBufferData(GL_ARRAY_BUFFER, vertex_data_.size() * sizeof(GLfloat),
		             vertex_data_.data(), GL_STATIC_DRAW);

		glGenBuffers(1, &color_buffer_);
		glBindBuffer(GL_ARRAY_BUFFER, color_buffer_);
		glBufferData(GL_ARRAY_BUFFER, color_data_.size() * sizeof(GLfloat),
		             color_data_.data(), GL_STATIC_DRAW);

		glGenBuffers(1, &index_buffer_);
		glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, index_buffer_);
		glBufferData(GL_ELEMENT_ARRAY_BUFFER, index_data_.size() * sizeof(GLuint),
		             index_data_.data(), GL_STATIC_DRAW);
	}

    void ViewFrustum::Render(const Navigation& navigation) {
	    CHECK(line_shader_->bind());

	    QMatrix4x4 modelview;
	    modelview.setToIdentity();
	    modelview.translate(position_);
	    modelview.rotate(local_to_global_);
	    modelview.rotate(orientation_);
	    modelview = navigation.GetModelViewMatrix() * modelview;

	    line_shader_->setUniformValue("m_mat", modelview);
	    line_shader_->setUniformValue("p_mat", navigation.GetProjectionMatrix());

	    glBindBuffer(GL_ARRAY_BUFFER, vertex_buffer_);
	    line_shader_->setAttributeArray("pos", GL_FLOAT, 0, 3);
	    glBindBuffer(GL_ARRAY_BUFFER, color_buffer_);
	    line_shader_->setAttributeArray("v_color", GL_FLOAT, 0, 4);

	    glLineWidth(2.0f);
	    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, index_buffer_);
	    glDrawElements(GL_LINES, (GLsizei)index_data_.size(), GL_UNSIGNED_INT, 0);
	    line_shader_->release();
    }


    ///////////////////////////////////
    // Implementation of OfflineTrajectory
    OfflineTrajectory::OfflineTrajectory(const std::vector<Eigen::Vector3d> &trajectory,
                                         const Eigen::Vector3f& color,
                                         const float default_height) {
        is_shader_init_ = false;
        vertex_data_.resize(trajectory.size() * 3);
        color_data_.resize(trajectory.size() * 4);
        index_data_.resize(trajectory.size());

        for(auto i=0; i<trajectory.size(); ++i){
            vertex_data_[3*i] = (GLfloat)trajectory[i][0];
            vertex_data_[3*i+1] = (GLfloat)trajectory[i][2] + default_height;
            vertex_data_[3*i+2] = -1 * (GLfloat)trajectory[i][1];

            color_data_[4*i] = color[0];
            color_data_[4*i+1] = color[1];
            color_data_[4*i+2] = color[2];
            color_data_[4*i+3] = 1.0f;

            index_data_[i] = (GLuint)i;
        }

//        for(auto i=0; i<trajectory.size(); i+=100){
//            printf("i %d, pos: (%.6f, %.6f, %.6f)\n", i, vertex_data_[3*i], vertex_data_[3*i+1], vertex_data_[3*i+2]);
//        }
    }

	void OfflineTrajectory::InitGL(){
        initializeOpenGLFunctions();

        line_shader_.reset(new QOpenGLShaderProgram());
        CHECK(line_shader_->addShaderFromSourceFile(QOpenGLShader::Vertex, ":/shaders/line_shader.vert"));
        CHECK(line_shader_->addShaderFromSourceFile(QOpenGLShader::Fragment, ":/shaders/line_shader.frag"));
        CHECK(line_shader_->link()) << "OfflineTrajectory: can not link line shader";
        CHECK(line_shader_->bind());
        line_shader_->enableAttributeArray("pos");
        line_shader_->enableAttributeArray("v_color");
        line_shader_->release();

        is_shader_init_ = true;

        glGenBuffers(1, &vertex_buffer_);
        glBindBuffer(GL_ARRAY_BUFFER, vertex_buffer_);
        glBufferData(GL_ARRAY_BUFFER, vertex_data_.size() * sizeof(GLfloat),
                     vertex_data_.data(), GL_STATIC_DRAW);

        glGenBuffers(1, &color_buffer_);
        glBindBuffer(GL_ARRAY_BUFFER, color_buffer_);
        glBufferData(GL_ARRAY_BUFFER, color_data_.size() * sizeof(GLfloat),
                     color_data_.data(), GL_STATIC_DRAW);

        glGenBuffers(1, &index_buffer_);
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, index_buffer_);
        glBufferData(GL_ELEMENT_ARRAY_BUFFER, index_data_.size() * sizeof(GLuint),
                     index_data_.data(), GL_STATIC_DRAW);
	}

    void OfflineTrajectory::Render(const Navigation& navigation){
        CHECK(line_shader_->bind());
        line_shader_->setUniformValue("m_mat", navigation.GetModelViewMatrix());
        line_shader_->setUniformValue("p_mat", navigation.GetProjectionMatrix());

        glLineWidth(2.0f);
        glBindBuffer(GL_ARRAY_BUFFER, vertex_buffer_);
        line_shader_->setAttributeArray("pos", GL_FLOAT, 0, 3);

        glBindBuffer(GL_ARRAY_BUFFER, color_buffer_);
        line_shader_->setAttributeArray("v_color", GL_FLOAT, 0, 4);

        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, index_buffer_);
        glDrawElements(GL_LINE_STRIP, (GLsizei)render_length_, GL_UNSIGNED_INT, 0);

        line_shader_->release();
    }


}//namespace IMUProject