//
// Created by yanhang on 3/19/17.
//

#ifndef PROJECT_RENDERABLE_H
#define PROJECT_RENDERABLE_H

#include <memory>
#include <vector>

#include <opencv2/opencv.hpp>
#include <QOpenGLFunctions>
#include <QOpenGLBuffer>
#include <QOpenGLTexture>
#include <QOpenGLShader>
#include <QOpenGLContext>
#include <QMatrix4x4>
#include <QQuaternion>
#include <QImage>
#include <Eigen/Eigen>

namespace IMUProject{

    class Renderable: protected QOpenGLFunctions{
    public:
        inline void UpdateModelView(const QMatrix4x4& model_view){
            model_view_ = model_view;
        }
        inline QMatrix4x4 GetModelView() const{
            return model_view_;
        }
        virtual void Render() = 0;
    protected:
        QMatrix4x4 model_view_;
    };

    class Scene: public Renderable{
    public:
        Scene();

    private:
        std::vector<std::shared_ptr<Renderable> > elements_;

    };

    class Canvas: public Renderable{
    public:
        Canvas(const int width, const int height, const cv::Mat* texture = nullptr);
        void Render();
    private:
        std::vector<GLfloat> vertex_data_;
        std::vector<GLuint> index_data_;
        std::vector<GLfloat> texcoord_data_;
        QImage texture_img_;

        QOpenGLTexture canvas_texture_;
        QOpenGLBuffer vertex_buffer_;
        QOpenGLBuffer index_buffer_;
        QOpenGLBuffer texcoord_buffer_;
    };

    class ViewFrustum: public Renderable{
    public:
        ViewFrustum(const double length=1.0, const bool with_axes=false);
        virtual void Render();
    private:
        std::vector<GLfloat> vertex_data_;
        std::vector<GLuint> index_data_;
        QOpenGLBuffer vertex_buffer_;
        QOpenGLBuffer index_buffer_;
    };

    class OfflineTrajectory: public Renderable{
    public:
        OfflineTrajectory(const std::vector<double>& trajectory);
        inline void SetRenderLength(const int length){
            render_length_ = length;
        }
        virtual void Render();
    private:
        int render_length_;
        std::vector<GLfloat> vertex_data_;
        std::vector<GLuint> index_data_;
        QOpenGLBuffer index_buffer_;
    };


} //namespace IMUProject



#endif //PROJECT_RENDERABLE_H
