//
// Created by yanhang on 9/26/17.
//

#ifndef PROJECT_SVR_CASCADE_H
#define PROJECT_SVR_CASCADE_H

#include <vector>
#include <glog/logging.h>

#include <opencv2/opencv.hpp>

namespace IMUProject {
    struct SVRCascadeOption {
        int num_classes;
        int num_channel;

    };

    class SVRCascade {
    public:
        SVRCascade();

    private:

    };

}  // namespace IMUProject
#endif //PROJECT_SVR_CASCADE_H
