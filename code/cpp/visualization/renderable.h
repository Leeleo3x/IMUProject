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
        virtual void Render() = 0;
    };

    class ViewFrustum: public Renderable{
    public:
        ViewFrustum(const double length=1.0, const bool with_axes=false);
        virtual void Render();
    };

    class OfflineTrajectory: public Renderable{
    public:
        OfflineTrajectory()
        virtual void Render();
    };


} //namespace IMUProject



#endif //PROJECT_RENDERABLE_H
