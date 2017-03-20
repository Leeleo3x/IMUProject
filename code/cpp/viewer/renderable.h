//
// Created by yanhang on 3/19/17.
//

#ifndef PROJECT_RENDERABLE_H
#define PROJECT_RENDERABLE_H

#include "navigation.h"

#include <memory>
#include <vector>

#include <opencv2/opencv.hpp>
#include <QOpenGLFunctions>
#include <QOpenGLBuffer>
#include <QOpenGLTexture>
#include <QOpenGLShaderProgram>
#include <QOpenGLContext>
#include <QMatrix4x4>
#include <QQuaternion>
#include <QImage>
#include <Eigen/Eigen>

namespace IMUProject{

    class Renderable: protected QOpenGLFunctions{
    public:
	    inline bool IsShaderInit() const{
		    return is_shader_init_;
	    }
	    virtual void Init() = 0;
        virtual void Render(const Navigation& navigation) = 0;
    protected:
	    bool is_shader_init_;
    };


    class Canvas: public Renderable{
    public:
        Canvas(const float width, const float height, const cv::Mat* texture = nullptr);
	    ~Canvas();
        virtual void Render(const Navigation& navigation);
	    virtual void Init();

    private:
        std::vector<GLfloat> vertex_data_;
        std::vector<GLuint> index_data_;
        std::vector<GLfloat> texcoord_data_;
        QImage texture_img_;

        std::shared_ptr<QOpenGLTexture> canvas_texture_;
        GLuint vertex_buffer_;
        GLuint index_buffer_;
        GLuint texcoord_buffer_;

	    std::shared_ptr<QOpenGLShaderProgram> tex_shader_;
	    std::shared_ptr<QOpenGLShaderProgram> line_shader_;
    };

    class ViewFrustum: public Renderable{
    public:
        ViewFrustum(const double length=1.0, const bool with_axes=false);
        virtual void Render(const Navigation& navigation);
	    virtual void Init();
    private:
        std::vector<GLfloat> vertex_data_;
        std::vector<GLuint> index_data_;
        GLuint vertex_buffer_;
        GLuint index_buffer_;

    };

    class OfflineTrajectory: public Renderable{
    public:
        OfflineTrajectory(const std::vector<double>& trajectory);
        inline void SetRenderLength(const int length){
            render_length_ = length;
        }
        virtual void Render(const Navigation& navigation);
	    virtual void Init();
    private:
        int render_length_;
        std::vector<GLfloat> vertex_data_;
        std::vector<GLuint> index_data_;
	    GLuint vertex_buffer_;
        GLuint index_buffer_;
    };


} //namespace IMUProject



#endif //PROJECT_RENDERABLE_H
