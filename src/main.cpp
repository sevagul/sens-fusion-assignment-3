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
#include "my_icp.h"

namespace fs = boost::filesystem;

using namespace Eigen;
using namespace std;
using namespace nanoflann;

// template <typename Der>
void Disparity2PointCloud(
    Eigen::Matrix<double, Eigen::Dynamic, 3> &pcl, cv::Mat &disparities,
    const int &dmin, const double &baseline = 160, const double &focal_length = 3740, int step = 10)
{
  int rows = disparities.rows;
  int cols = disparities.cols;
  std::vector<std::vector<double>> points_vec;
#pragma omp paraller for
  for (int r = 0; r < rows; r+=step)
  {
    // #pragma omp parallel for
    for (int c = 0; c < cols; c+=step)
    {
      if (disparities.at<uchar>(r, c) == 0)
        continue;

      int d = (int)disparities.at<uchar>(r, c) + dmin;
      int u1 = c - cols / 2;
      int u2 = c + d - cols / 2;
      int v1 = r - rows / 2;

      const double Z = baseline * focal_length / d;
      const double X = -0.5 * (baseline * (u1 + u2)) / d;
      const double Y = baseline * v1 / d;
      std::vector<double> point_vec{X, Y, Z};
      points_vec.push_back(point_vec);
    }
  }

  std::cout << std::endl;

  int N = points_vec.size();
  pcl.resize(N, 3);

  for (int i = 0; i < N; i++)
  {
    for (int j = 0; j < 3; j++)
    {
      pcl(i, j) = points_vec[i][j];
    }
  }
}

int main(int argc, char **argv)
{
  bool visualize = true;
  cv::Mat D1;
  cv::Mat D2;
  std::string datasetName = "Art";
  fs::path data_path("data");
  data_path = data_path / datasetName;

  fs::path d1_path = data_path / "disp1.png";
  fs::path d2_path = data_path / "disp5.png";

  D1 = cv::imread(d1_path.string(), 0);
  D2 = cv::imread(d2_path.string(), 0);

  EigenPCL PCL1;
  EigenPCL PCL2;

  Disparity2PointCloud(PCL1, D1, 200, 160, 3740);
  Disparity2PointCloud(PCL2, D2, 200, 160, 3740);

  EigenPCL PCL3(PCL2);

  Matrix4<double> T;
  T = my_icp::ICP(PCL1, PCL2);

  std::cout << T << std::endl;
  EigenPCL result(PCL2);
  my_icp::applyTransformation(result, T);

  my_vis::vis_point_clouds(PCL1, result);

  return 0;
}