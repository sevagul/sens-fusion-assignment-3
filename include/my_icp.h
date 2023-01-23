#ifndef MY_ICP_H
#define MY_ICP_H

#include "../nanoflann.hpp"
#include "my_visualizations.hpp"
#include "my_types.h"

#include <cstdlib>
#include <ctime>
#include <iostream>
#include <opencv2/opencv.hpp>

#include <boost/filesystem.hpp>
#include <Eigen/Dense>
#include <Eigen/Core>

#include <omp.h>

namespace fs = boost::filesystem;

using namespace Eigen;
using namespace std;
using namespace nanoflann;

namespace my_icp
{
    template <typename Der>
    int findClosestIndex(Vector3<Der> &query_pt, const MyKdTree<EigenPCLMat<Der>> &my_tree, Der &closest_sqdist);

    // template <typename Der>
    Matrix4<double> best_fit_transform(const EigenPCLMat<double> &A, const EigenPCLMat<double> &B);

    Matrix4<double> ICPiter(const EigenPCLMat<double> &PCL1, const EigenPCLMat<double> &PCL2, MyKdTree<EigenPCL> &my_tree, double& avg_distance);

    void applyTransformation(EigenPCLMat<double> &PCL, const Eigen::Matrix4d T);

    Matrix4<double> ICP(const EigenPCLMat<double> &PCL1, EigenPCLMat<double> &PCL2_in, int max_iter = 15);
}

#endif
