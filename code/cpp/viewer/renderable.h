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
#include <QMatrix3x3>
#include <QQuaternion>
#include <QImage>
#include <Eigen/Eigen>
#include <glog/logging.h>

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
               const Eigen::Vector3f grid_color=Eigen::Vector3f(0.0, 0.0, 0.0),
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
        ViewFrustum(const float length=1.0, const bool with_axes=true, const float default_height = 1.7);
        virtual void Render(const Navigation& navigation);
	    inline void UpdateCameraPose(const Eigen::Vector3d& position,
	                                 const Eigen::Quaterniond& orientation){
		    position_ = QVector3D((float)position[0], (float)position[2] + default_height_,
		                          -1.0f * (float)position[1]);
		    orientation_ = QQuaternion((float)orientation.w(), (float)orientation.x(),
		                               (float)orientation.y(), (float)orientation.z());
	    }
	    virtual void InitGL();
    private:
	    QVector3D position_;
	    QQuaternion orientation_;
	    QQuaternion local_to_global_;
        std::vector<GLfloat> vertex_data_;
        std::vector<GLuint> index_data_;
	    std::vector<GLfloat> color_data_;
        GLuint vertex_buffer_;
        GLuint index_buffer_;
	    GLuint color_buffer_;

	    const float default_height_;

	    std::shared_ptr<QOpenGLShaderProgram> line_shader_;

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

    class OfflineSpeedPanel: public Renderable{
    public:
        OfflineSpeedPanel(const std::vector<Eigen::Vector3d>& positions,
                          const std::vector<Eigen::Quaterniond>& orientation,
                          const Eigen::Quaterniond& init_dir, const float radius = 1.0f,
                          const Eigen::Vector3f fcolor=Eigen::Vector3f(1.0f, 0.0f, 0.0f),
                          const Eigen::Vector3f dcolor=Eigen::Vector3f(0.0f, 0.0f, 1.0f));

        virtual void InitGL();
        virtual void Render(const Navigation& navigation);

        inline void UpdateDirection(const Eigen::Vector3d& forward_dir,
                                 const Eigen::Vector3d& device_dir){
	        CHECK(pointer_vertex_buffer_.bind()) << "Can not bind pointer_vertex_buffer_";
	        GLfloat* vertex_data = (GLfloat *)pointer_vertex_buffer_.map(QOpenGLBuffer::WriteOnly);
	        CHECK(vertex_data) << "Can not map pointer_vertex_buffer_";
            vertex_data[3] = static_cast<float>(forward_dir[0] / max_speed_ * radius_);
            vertex_data[4] = static_cast<float>(-forward_dir[1] / max_speed_ * radius_);
            vertex_data[9] = static_cast<float>(device_dir[0] / max_speed_ * radius_);
            vertex_data[10] = static_cast<float>(-device_dir[1] / max_speed_ * radius_);
	        pointer_vertex_buffer_.unmap();
	        pointer_vertex_buffer_.release();
        }
    private:
        const float radius_;
        const Eigen::Vector3f fcolor_;
        const Eigen::Vector3f dcolor_;

	    std::shared_ptr<QOpenGLTexture> panel_texture_;
        std::vector<GLfloat> panel_vertex_data_;
        std::vector<GLfloat> panel_color_data_;
	    std::vector<GLfloat> panel_texcoord_data_;
        std::vector<GLuint> panel_index_data_;

        GLuint panel_vertex_buffer_;
        GLuint panel_color_buffer_;
	    GLuint panel_texcoord_buffer_;
        GLuint panel_index_buffer_;

        std::vector<GLfloat> pointer_vertex_data_;
        std::vector<GLfloat> pointer_color_data_;
	    std::vector<GLuint> pointer_index_data_;

	    QOpenGLBuffer pointer_vertex_buffer_;
	    //GLuint pointer_vertex_buffer_;
	    GLuint pointer_color_buffer_;
	    GLuint pointer_index_buffer_;

        static constexpr double max_speed_ = 2.5;
	    const float panel_alpha_;

	    std::shared_ptr<QOpenGLShaderProgram> tex_shader_;
        std::shared_ptr<QOpenGLShaderProgram> line_shader_;
    };


} //namespace IMUProject



#endif //PROJECT_RENDERABLE_H
