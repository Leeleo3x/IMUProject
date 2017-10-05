//
// Created by Yan Hang on 3/20/17.
//

#ifndef PROJECT_NAVIGATION_H
#define PROJECT_NAVIGATION_H

#include <QMatrix4x4>
#include <Eigen/Eigen>

namespace IMUProject {

enum CameraMode {
  CENTER,
  BACK,
  TOP,
  PERSPECTIVE,
  SIDE,
  TRANSITION,
  NUM_MODE
};

inline float GetTransitionRatio(const int counter, const int total) {
  const float x = (float) counter;
  const float N = (float) total;
  const float v = (x - 0.5f * N) / N * (float) M_PI;
  return 0.5f * (1.0f + std::sin(v));
}

class Navigation {
 public:
  Navigation(const float fov,
             const float aspect_ratio,
             const float canvas_width,
             const float canvas_height,
             CameraMode start_mode = PERSPECTIVE,
             const float trajectory_height = 1.7f,
             const float center_height = 10.0f,
             const float perspective_height = 15.0f,
             const float back_height = 5.0f,
             const float top_height = 50.0f,
             const int transition_frames = 30) :
      fov_(fov), aspect_ratio_(aspect_ratio), canvas_width_(canvas_width), canvas_height_(canvas_height),
      trajectory_height_(trajectory_height), center_height_(center_height), top_height_(top_height),
      perspective_height_(perspective_height), back_height_(back_height),
      transition_frames_(transition_frames), render_mode_(start_mode), src_mode_(start_mode) {
    camera_centers_.resize(NUM_MODE);
    center_points_.resize(NUM_MODE);
    up_dirs_.resize(NUM_MODE);
    fovs_.resize(NUM_MODE);
  }

  inline QMatrix4x4 GetProjectionMatrix() const {
    return projection_;
  }

  inline QMatrix4x4 GetModelViewMatrix() const {
    return modelview_;
  }

  void UpdateNavitation(const Eigen::Vector3d &pos,
                        const Eigen::Quaterniond &orientation);

  inline void StartTransition(const CameraMode dst) {
    src_mode_ = render_mode_;
    dst_mode_ = dst;
    transition_counter_ = 0;
    render_mode_ = TRANSITION;
//			render_mode_ = dst;
//			src_mode_ = dst;
  }

  inline void SetModelView(QMatrix4x4 modelview) {
    modelview_ = modelview;
  }

  inline void SetProjection(QMatrix4x4 projection) {
    projection_ = projection;
  }

  inline void SetCameraMode(const CameraMode mode) {
    if (mode != render_mode_) {
      StartTransition(mode);
    }
  }
 private:
  const float fov_;
  const float aspect_ratio_;
  const float trajectory_height_;

  const float canvas_width_;
  const float canvas_height_;

  const float center_height_;
  const float back_height_;
  const float perspective_height_;
  const float top_height_;

  std::vector<QVector3D> camera_centers_;
  std::vector<QVector3D> center_points_;
  std::vector<QVector3D> up_dirs_;
  std::vector<float> fovs_;

  CameraMode render_mode_;

  CameraMode src_mode_;
  CameraMode dst_mode_;
  const int transition_frames_;
  int transition_counter_;

  QMatrix4x4 projection_;
  QMatrix4x4 modelview_;
};
}//namespace IMUProject

#endif //PROJECT_NAVIGATION_H
