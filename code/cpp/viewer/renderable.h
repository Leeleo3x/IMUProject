//
// Created by yanhang on 3/19/17.
//

#ifndef PROJECT_RENDERABLE_H
#define PROJECT_RENDERABLE_H

#include <memory>
#include <vector>

#include <QOpenGLFunctions>
#include <QOpenGLBuffer>
#include <QOpenGLTexture>
#include <QOpenGLShader>
#include <QOpenGLContext>
#include <QMatrix4x4>
#include <QQuaternion>
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

    class ViewFrustum: public Renderable{
    public:
        ViewFrustum(const double length=1.0, const bool with_axes=false);
        virtual void Render();
    private:
        QOpenGLBuffer vertex_buffer_;
    };

    class OfflineTrajectory: public Renderable{
    public:
        OfflineTrajectory()
        virtual void Render();
    };


} //namespace IMUProject



#endif //PROJECT_RENDERABLE_H
