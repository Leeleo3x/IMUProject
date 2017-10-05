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

namespace IMUProject {

class Renderable : protected QOpenGLFunctions {
 public:
  inline bool IsShaderInit() const {
    return is_shader_init_;
  }
  virtual void InitGL() = 0;
  virtual void Render(const Navigation &navigation) = 0;
 protected:
  bool is_shader_init_;
};

class Canvas : public Renderable {
 public:
  Canvas(const float width, const float height,
         const float grid_size = 1.0f,
         const Eigen::Vector3f grid_color = Eigen::Vector3f(0.0, 0.0, 0.0),
         const cv::Mat *texture = nullptr);
  ~Canvas();
  virtual void Render(const Navigation &navigation);
  virtual void InitGL();

  float GetWidth() const { return width_; }
  float GetHeight() const { return height_; }

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

class ViewFrustum : public Renderable {
 public:
  ViewFrustum(const float length = 1.0, const bool with_axes = true, const float default_height = 1.7);
  virtual void Render(const Navigation &navigation);
  inline void UpdateCameraPose(const Eigen::Vector3d &position,
                               const Eigen::Quaterniond &orientation) {
    position_ = QVector3D((float) position[0], (float) position[2] + default_height_,
                          -1.0f * (float) position[1]);
    orientation_ = QQuaternion((float) orientation.w(), (float) orientation.x(),
                               (float) orientation.y(), (float) orientation.z());
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

class OfflineTrajectory : public Renderable {
 public:
  OfflineTrajectory(const std::vector<Eigen::Vector3d> &trajectory, const Eigen::Vector3f &color,
                    const float default_height = 1.7);
  inline void SetRenderLength(const int length) {
    render_length_ = length;
  }
  virtual void Render(const Navigation &navigation);
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

class OfflineSpeedPanel : public Renderable {
 public:
  OfflineSpeedPanel(const int kTraj,
                    const std::vector<Eigen::Vector3f> &colors,
                    const float radius = 1.0f,
                    const float max_speed = 1.5f,
                    const QVector3D initial_dir = QVector3D(0.0f, 1.0f, 0.0f));

  virtual void InitGL();
  virtual void Render(const Navigation &navigation);

  inline void UpdateDirections(const std::vector<Eigen::Vector3d> &speeds) {
    CHECK_EQ(speeds.size(), kTraj_);
    for (auto i = 0; i < kTraj_; ++i) {
      Eigen::Vector3f s_f = speeds[i].cast<float>();
      QQuaternion q = QQuaternion::rotationTo(initial_dir_, QVector3D(s_f[0], s_f[1], s_f[2]));
      const float mag = s_f.norm();
      // apply a low pass filter
      speed_mags_[i] = filter_alpha_ * speed_mags_[i] + (1.0f - filter_alpha_) * mag;
      const float scale = speed_mags_[i] / max_speed_;
      //printf("speed mag: %.6f, scale: %.6f\n", speeds[i].norm(), scale);
      pointer_modelviews_[i].setToIdentity();
      pointer_modelviews_[i].scale(scale, scale, 1.0f);
      pointer_modelviews_[i].rotate(q);
      pointer_modelviews_[i] = panel_view_matrix_ * pointer_modelviews_[i];
    }
  }

  inline void ResetFilters() {
    for (auto &v: speed_mags_) {
      v = 0.0;
    }
  }
 private:
  const float radius_;
  const int kTraj_;
  const QVector3D initial_dir_;
  const float filter_alpha_;

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

  std::vector<QMatrix4x4> pointer_modelviews_;
  std::vector<float> speed_mags_;

  QMatrix4x4 panel_view_matrix_;
  QMatrix4x4 panel_projection_matrix_;

  //QOpenGLBuffer pointer_vertex_buffer_;
  GLuint pointer_vertex_buffer_;
  GLuint pointer_color_buffer_;
  GLuint pointer_index_buffer_;

  const float max_speed_;
  const float z_pos_;
  const float panel_alpha_;

  std::shared_ptr<QOpenGLShaderProgram> tex_shader_;
  std::shared_ptr<QOpenGLShaderProgram> line_shader_;
};

class LegendRenderer : public Renderable {
 public:
  LegendRenderer(const int width, const int height, const QImage &texture_img);
  virtual void InitGL();
  virtual void Render(const Navigation &navigation);

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
