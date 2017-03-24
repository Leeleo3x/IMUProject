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
        OfflineSpeedPanel(const float radius = 1.0f,
                          const Eigen::Vector3f fcolor=Eigen::Vector3f(0.7f, 0.0f, 1.0f),
                          const Eigen::Vector3f dcolor=Eigen::Vector3f(0.0f, 0.8f, 1.0f));

        virtual void InitGL();
        virtual void Render(const Navigation& navigation);

        inline void UpdateDirection(const Eigen::Vector3d& forward_dir,
                                 const Eigen::Vector3d& device_dir) {
            constexpr double min_distance = 0.01;
            if(forward_dir.norm() < min_distance){
                return;
            }
			Eigen::Quaternionf device_rot = Eigen::Quaterniond::FromTwoVectors(forward_dir, device_dir).cast<float>();
			Eigen::Quaternionf panel_rot = Eigen::Quaterniond::FromTwoVectors(forward_dir,
																			  Eigen::Vector3d(0, 1, 0)).cast<float>();

			panel_modelview_.setToIdentity();
			//panel_modelview_.scale(1.0f, (float)forward_dir.norm() / max_speed_ * radius_, 1.0f);
			panel_modelview_.rotate(QQuaternion(panel_rot.w(), panel_rot.x(), panel_rot.y(), panel_rot.z()));
			panel_modelview_ = panel_view_matrix_ * panel_modelview_;

			device_pointer_modelview_.setToIdentity();
			//device_pointer_modelview_.scale(1.0f, static_cast<float>(forward_dir.norm() / max_speed_), 1.0f);
			device_pointer_modelview_.rotate(
					QQuaternion(device_rot.w(), device_rot.x(), device_rot.y(), device_rot.z()));
			device_pointer_modelview_ = panel_view_matrix_ * device_pointer_modelview_;
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

		std::vector<GLfloat> device_vertex_data_;
		std::vector<GLfloat> device_color_data_;
		std::vector<GLuint> device_index_data_;

		GLuint device_vertex_buffer_;
		GLuint device_color_buffer_;
		GLuint device_index_buffer_;

		QMatrix4x4 panel_modelview_;
        QMatrix4x4 device_pointer_modelview_;

        QMatrix4x4 panel_view_matrix_;
        QMatrix4x4 panel_projection_matrix_;

	    //QOpenGLBuffer pointer_vertex_buffer_;
	    GLuint pointer_vertex_buffer_;
	    GLuint pointer_color_buffer_;
	    GLuint pointer_index_buffer_;

        static constexpr double max_speed_ = 2.5;
        const float z_pos_;
	    const float panel_alpha_;

	    std::shared_ptr<QOpenGLShaderProgram> tex_shader_;
        std::shared_ptr<QOpenGLShaderProgram> line_shader_;
    };

    class LegendRenderer: public Renderable{
    public:
        LegendRenderer(const int width, const int height, const QImage& texture_img);
        virtual void InitGL();
        virtual void Render(const Navigation& navigation);

    private:
        const float width_;
        const float height_;
        const float z_pos_;

        std::vector<GLfloat> vertex_data_;
        std::vector<GLfloat> texcoord_data_;
        std::vector<GLuint> index_data_;

        QImage texture_img_;

        GLuint vertex_buffer_;
        GLuint texcoord_buffer_;
        GLuint index_buffer_;

        QMatrix4x4 modelview_;
        QMatrix4x4 projection_;

        std::shared_ptr<QOpenGLTexture> texture_;
        std::shared_ptr<QOpenGLShaderProgram> tex_shader_;
    };

} //namespace IMUProject



#endif //PROJECT_RENDERABLE_H
