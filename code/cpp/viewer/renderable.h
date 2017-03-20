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
	    virtual void InitGL() = 0;
        virtual void Render(const Navigation& navigation) = 0;
    protected:
	    bool is_shader_init_;
    };


    class Canvas: public Renderable{
    public:
        Canvas(const float width, const float height,
               const float grid_size = 1.0f,
               const Eigen::Vector3f grid_color=Eigen::Vector3f(0.7, 0.7, 0.7),
               const cv::Mat* texture = nullptr);
	    ~Canvas();
        virtual void Render(const Navigation& navigation);
	    virtual void InitGL();

        float GetWidth() const {return width_;}
        float GetHeight() const {return height_;}

    private:
        std::vector<GLfloat> vertex_data_;
        std::vector<GLuint> index_data_;
        std::vector<GLfloat> grid_vertex_data_;
        std::vector<GLuint> grid_index_data_;
        std::vector<GLfloat> texcoord_data_;
        std::vector<GLfloat> grid_color_data_;
        QImage texture_img_;

        const float width_;
        const float height_;
        const float line_alpha_;

        std::shared_ptr<QOpenGLTexture> canvas_texture_;
        GLuint vertex_buffer_;
        GLuint index_buffer_;
        GLuint texcoord_buffer_;
        GLuint grid_vertex_buffer_;
        GLuint grid_index_buffer_;
        GLuint grid_color_buffer_;

	    std::shared_ptr<QOpenGLShaderProgram> tex_shader_;
	    std::shared_ptr<QOpenGLShaderProgram> line_shader_;
    };

    class ViewFrustum: public Renderable{
    public:
        ViewFrustum(const double length=1.0, const bool with_axes=false);
        virtual void Render(const Navigation& navigation);
	    virtual void InitGL();
    private:
        std::vector<GLfloat> vertex_data_;
        std::vector<GLuint> index_data_;
        GLuint vertex_buffer_;
        GLuint index_buffer_;

    };

    class OfflineTrajectory: public Renderable{
    public:
        OfflineTrajectory(const std::vector<Eigen::Vector3d>& trajectory, const Eigen::Vector3f& color,
                          const float default_height = 1.7);
        inline void SetRenderLength(const int length){
            render_length_ = length;
        }
        virtual void Render(const Navigation& navigation);
	    virtual void InitGL();
    private:
        int render_length_;
        std::vector<GLfloat> vertex_data_;
        std::vector<GLfloat> color_data_;
        std::vector<GLuint> index_data_;
	    GLuint vertex_buffer_;
        GLuint color_buffer_;
        GLuint index_buffer_;

        std::shared_ptr<QOpenGLShaderProgram> line_shader_;
    };


} //namespace IMUProject



#endif //PROJECT_RENDERABLE_H
