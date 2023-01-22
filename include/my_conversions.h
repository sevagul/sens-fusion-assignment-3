
#ifndef MY_CONV_H
#define MY_CONV_H

#include <pcl/point_types.h>
#include <pcl/visualization/cloud_viewer.h>
#include <thread>
#include <cstdlib>
#include <ctime>
#include <iostream>
#include <Eigen/Dense>
#include "my_types.h"

namespace my_conv{
    template <typename Der>
    inline void EigenToPcl(const EigenPCLMat<Der> &pclIn, PclType &pclOut, int c1, int c2)
    {
        for (size_t i = 0; i < pclIn.rows(); i++)
        {
            auto row = pclIn.row(i);
            auto cur_pt = PointType(0, uint8_t(c1), uint8_t(c2));
            cur_pt.x = row(0);
            cur_pt.y = row(1);
            cur_pt.z = row(2);
            pclOut.push_back(cur_pt);
        }
    }
}

#endif