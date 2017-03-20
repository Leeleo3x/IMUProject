//
// Created by Yan Hang on 3/20/17.
//

#ifndef PROJECT_NAVIGATION_H
#define PROJECT_NAVIGATION_H

#include <QMatrix4x4>
#include <Eigen/Eigen>

namespace IMUProject {
	class Navigation {
	public:
		Navigation(){}

		inline QMatrix4x4 GetProjectionMatrix() const{
			return projection_;
		}

		inline QMatrix4x4 GetModelViewMatrix() const{
			return modelview_;
		}

		void UpdateCamera(const Eigen::Vector3d& pos,
		                  const Eigen::Quaterniond& orientation);


	private:
		QMatrix4x4 projection_;
		QMatrix4x4 modelview_;
	};
}//namespace IMUProject

#endif //PROJECT_NAVIGATION_H
