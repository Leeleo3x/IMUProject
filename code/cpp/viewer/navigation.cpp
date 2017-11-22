//
// Created by Yan Hang on 3/20/17.
//

#include "navigation.h"
#include <iostream>
#include <QDebug>

namespace IMUProject {

void Navigation::UpdateNavitation(const Eigen::Vector3d &pos,
                                  const Eigen::Quaterniond &orientation) {
  Eigen::Vector3f pos_f = pos.cast<float>();
  Eigen::Quaternionf orientation_f = orientation.cast<float>();
  Eigen::Vector3f global_backward = orientation_f * Eigen::Vector3f(0, 0, 1);
  Eigen::Vector3f camera_center_back = pos_f + global_backward * 5.0f;

  camera_centers_[BACK] = QVector3D(camera_center_back[0], back_height_, -1.0f * camera_center_back[1]);
  camera_centers_[CENTER] = QVector3D(0.0, center_height_, 0.0f);
  camera_centers_[PERSPECTIVE] = QVector3D(0.0f, perspective_height_, (canvas_height_ - pos[1]) / 2.0f);
  camera_centers_[TOP] = QVector3D(0.0f, top_height_, 0.0f);
  camera_centers_[SIDE] = QVector3D((-pos[0] - canvas_width_) / 2.0f, perspective_height_, -1 * pos_f[1]);

  center_points_[BACK] = QVector3D((float) pos[0], 1.0f, -1 * (float) pos[1]);
  center_points_[CENTER] = QVector3D(pos[0], trajectory_height_, -1 * pos[1]);
  center_points_[PERSPECTIVE] = QVector3D(0.0f, 0.0f, -0.5f * pos[1]);
  center_points_[TOP] = QVector3D(0.0f, 0.0f, 0.0f);
  center_points_[SIDE] = QVector3D(0.0f, 0.0f, -1 * pos_f[1]);

  up_dirs_[BACK] = QVector3D(0.0f, 1.0f, 0.0f);
  up_dirs_[CENTER] = QVector3D(0.0f, 1.0f, 0.0f);
  up_dirs_[PERSPECTIVE] = QVector3D(0.0f, 1.0f, 0.0f);
  up_dirs_[TOP] = QVector3D(0.0f, 0.0f, -1.0f);
  up_dirs_[SIDE] = QVector3D(0.0f, 1.0f, 0.0f);

  if (render_mode_ == TRANSITION) {
    if (transition_counter_ == transition_frames_) {
      render_mode_ = dst_mode_;
      src_mode_ = render_mode_;
    } else {
      const float ratio = GetTransitionRatio(transition_counter_, transition_frames_);
      modelview_.setToIdentity();
      modelview_.lookAt((1.0f - ratio) * camera_centers_[src_mode_] + ratio * camera_centers_[dst_mode_],
                        (1.0f - ratio) * center_points_[src_mode_] + ratio * center_points_[dst_mode_],
                        (1.0f - ratio) * up_dirs_[src_mode_] + ratio * up_dirs_[dst_mode_]);

      float fov = fovs_[src_mode_] * (1.0f - ratio) + fovs_[dst_mode_] * ratio;
      //fov = std::atan(1.0f / fov) / std::atan(1.0f / fovs_[src_mode_]) * fovs_[dst_mode_];
      projection_.setToIdentity();
      projection_.perspective(fov, aspect_ratio_, 0.01f, 100.0f);

      transition_counter_++;
    }
  } else {
    modelview_.setToIdentity();
    modelview_.lookAt(camera_centers_[render_mode_],
                      center_points_[render_mode_],
                      up_dirs_[render_mode_]);
    projection_.setToIdentity();
    projection_.perspective(fovs_[render_mode_], aspect_ratio_, 0.01f, 100.0f);
  }
}

} //namespace IMUProject