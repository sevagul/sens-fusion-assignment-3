
#ifndef MY_TYPES_H
#define MY_TYPES_H

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <Eigen/Dense>
#include "../nanoflann.hpp"




template <typename Der>
using Vector3 = Eigen::Matrix<Der, 3, 1>;
template <typename Der>
using Matrix4 = Eigen::Matrix<Der, 4, 4>;
template <typename Der>
using Matrix3 = Eigen::Matrix<Der, 3, 3>;

template <typename Der>
using EigenPCLMat = Eigen::Matrix<Der, Eigen::Dynamic, 3>;

template <typename Der>
using MatrixX = Eigen::Matrix<Der, Eigen::Dynamic, Eigen::Dynamic>;
template <typename PC_type>
using MyKdTree = nanoflann::KDTreeEigenMatrixAdaptor<PC_type, 3, nanoflann::metric_L2>;

using EigenPCL = Eigen::Matrix<double, Eigen::Dynamic, 3>;
using PointType = pcl::PointXYZRGB;
using PclType = pcl::PointCloud<PointType>;



#endif