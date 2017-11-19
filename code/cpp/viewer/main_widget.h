//
// Created by yanhang on 3/19/17.
//

#ifndef PROJECT_MAIN_WINDOW_H
#define PROJECT_MAIN_WINDOW_H

#include "renderable.h"

#include <memory>

#include <opencv2/opencv.hpp>
#include <QTimerEvent>
#include <QBasicTimer>
#include <QKeyEvent>
#include <QOpenGLWidget>

#include "utility/data_io.h"

namespace IMUProject {

class MainWidget : public QOpenGLWidget, protected QOpenGLFunctions {
 Q_OBJECT
 public:
  explicit MainWidget(const std::string &path,
                      const int canvas_width = 50,
                      const int convas_height = 50,
                      QWidget *parent = 0);
  ~MainWidget() {
  }

 protected:
  void initializeGL() Q_DECL_OVERRIDE;
  void resizeGL(int w, int h) Q_DECL_OVERRIDE;
  void paintGL() Q_DECL_OVERRIDE;
  void keyPressEvent(QKeyEvent *e) Q_DECL_OVERRIDE;

  void timerEvent(QTimerEvent *event) Q_DECL_OVERRIDE;

 private:
  void InitializeTrajectories(const std::string &path);
  void UpdateCameraInfo(const int ind);

  std::vector<double> ts_;
  std::vector<std::vector<Eigen::Vector3d> > positions_;
  std::vector<std::vector<Eigen::Quaterniond> > orientations_;

  std::shared_ptr<Canvas> canvas_;
  std::vector<std::shared_ptr<OfflineTrajectory> > trajectories_;
  std::vector<std::shared_ptr<ViewFrustum> > view_frustum_;
  std::shared_ptr<OfflineSpeedPanel> speed_panel_;

  std::vector<std::shared_ptr<LegendRenderer> > legends_;
  std::vector<QRect> legend_areas_;

  const Eigen::Vector3f const_traj_color;

  bool is_rendering_;

  int render_count_;
  QBasicTimer render_timer_;

  const int panel_border_margin_;
  const int panel_size_;
  std::shared_ptr<Navigation> navigation_;

  static constexpr int frame_interval_ = 5;
};

}//namespace IMUProject

#endif //PROJECT_MAIN_WINDOW_H
