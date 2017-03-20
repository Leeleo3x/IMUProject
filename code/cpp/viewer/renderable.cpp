//
// Created by yanhang on 3/19/17.
//

#include "renderable.h"

#include <glog/logging.h>

namespace IMUProject{

    ////////////////////////////////////
    // Implementation of canvas
    Canvas::Canvas(const float width, const float height, const cv::Mat *texture){
	    initializeOpenGLFunctions();
	    is_shader_init_ = false;
	    vertex_data_ = {-width/2.0f, -height/2.0f, 0.0f,
	                    width/2.0f, -height/2.0f, 0.0f,
	                    width/2.0f, height/2.0f, 0.0f,
	                    -width/2.0f, height/2.0f, 0.0f};
	    index_data_ = {0, 1, 2, 1, 2, 3};
	    texcoord_data_ = {0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0};

	    if(texture == nullptr){
		    texture_img_.load(":images/iccv_texture.png");
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

	    canvas_texture_.reset(new QOpenGLTexture(texture_img_));
    }

	Canvas::~Canvas(){
		glDeleteBuffers(1, &vertex_buffer_);
		glDeleteBuffers(1, &index_buffer_);
		glDeleteBuffers(1, &texcoord_buffer_);
	}

	void Canvas::Init(){
		tex_shader_.reset(new QOpenGLShaderProgram());
		CHECK(tex_shader_->addShaderFromSourceCode(QOpenGLShader::Vertex, ":shaders/canvas_shader.vert"));
		CHECK(tex_shader_->addShaderFromSourceCode(QOpenGLShader::Fragment, ":shaders/canvas_shader.frag"));
		CHECK(tex_shader_->link()) << "Canvas: can not link shader";
		CHECK(tex_shader_->bind()) << "Canvas: can not bind shader";

		tex_shader_->enableAttributeArray("pos");
		tex_shader_->enableAttributeArray("texcoord");
		tex_shader_->release();

		is_shader_init_ = true;
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
    }

	///////////////////////////////////
	// Implementation of ViewFrustum
	ViewFrustum::ViewFrustum(const double length, const bool with_axes) {

	}

	void ViewFrustum::Init(){

	}
    void ViewFrustum::Render(const Navigation& navigation) {

    }


    ///////////////////////////////////
    // Implementation of OfflineTrajectory
    OfflineTrajectory::OfflineTrajectory(const std::vector<double> &trajectory) {

    }

	void OfflineTrajectory::Init(){

	}

    void OfflineTrajectory::Render(const Navigation& navigation){

    }


}//namespace IMUProject