//
// Created by Yan Hang on 3/20/17.
//

#include "navigation.h"

namespace IMUProject{

	void Navigation::UpdateCamera(const Eigen::Vector3d &pos, const Eigen::Quaterniond &orientation) {
		QMatrix4x4 modelview;
		modelview.setToIdentity();
		Eigen::Vector3d global_backward = orientation * Eigen::Vector3d(0, 0, 1);
		Eigen::Vector3d camera_center = pos + global_backward * 5.0;
		camera_center[2] = 3.0;
		modelview.lookAt(QVector3D((float)camera_center[0], (float)camera_center[2], -1*(float)camera_center[1]),
						 QVector3D((float)pos[0], 1.0f, -1*(float)pos[1]),
						 QVector3D(0.0f, 1.0f, 0.0f));
		SetModelView(modelview);
	}

} //namespace IMUProject