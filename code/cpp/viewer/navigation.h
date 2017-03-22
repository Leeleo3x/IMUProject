//
// Created by Yan Hang on 3/20/17.
//

#ifndef PROJECT_NAVIGATION_H
#define PROJECT_NAVIGATION_H

#include <QMatrix4x4>
#include <Eigen/Eigen>

namespace IMUProject {

	enum CameraMode{
		CENTER,
		BACK,
		TOP
	};

	class Navigation {
	public:
		Navigation(const float fov,
		           const float width,
		           const float height):
				fov_(fov), width_(width), height_(height){
			QMatrix4x4 projection;
			projection.setToIdentity();
			projection.perspective(fov_, width_ / height_, 0.001f, 300.0f);
			SetProjection(projection);
		}

		inline QMatrix4x4 GetProjectionMatrix() const{
			return projection_;
		}

		inline QMatrix4x4 GetModelViewMatrix() const{
			return modelview_;
		}

		void UpdateCameraBack(const Eigen::Vector3d& pos,
		                      const Eigen::Quaterniond& orientation);
		void UpdateCameraCenter(const Eigen::Vector3d& pos,
		                        const Eigen::Vector3d& center);

		inline void SetModelView(QMatrix4x4 modelview){
			modelview_ = modelview;
		}

		inline void SetProjection(QMatrix4x4 projection){
			projection_ = projection;
		}

	private:
		const float fov_;
		const float width_;
		const float height_;
		QMatrix4x4 projection_;
		QMatrix4x4 modelview_;
	};
}//namespace IMUProject

#endif //PROJECT_NAVIGATION_H
