#include "my_visualizations.hpp"
#include "my_types.h"

#include <cstdlib>
#include <ctime>
#include <iostream>
#include <opencv2/opencv.hpp>

#include <boost/filesystem.hpp>
#include <Eigen/Dense>
#include <Eigen/Core>
#include <Eigen/Geometry>

#include <omp.h>
#include "my_icp.h"

#include <random>

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
  for (int r = 0; r < rows; r += step)
  {
    // #pragma omp parallel for
    for (int c = 0; c < cols; c += step)
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

Eigen::Affine3d create_rotation_matrix(double ax, double ay, double az)
{
  Eigen::Affine3d rx =
      Eigen::Affine3d(Eigen::AngleAxisd(ax, Eigen::Vector3d(1, 0, 0)));
  Eigen::Affine3d ry =
      Eigen::Affine3d(Eigen::AngleAxisd(ay, Eigen::Vector3d(0, 1, 0)));
  Eigen::Affine3d rz =
      Eigen::Affine3d(Eigen::AngleAxisd(az, Eigen::Vector3d(0, 0, 1)));
  return rz * ry * rx;
}

void addGausiianNoise(EigenPCLMat<double> &PCL, double stddev, int seed)
{
  std::vector<double> data = {1., 2., 3., 4., 5., 6.};

  // Define random generator with Gaussian distribution
  const double mean = 0.0;
  std::default_random_engine generator;
  generator.seed(seed);
  std::normal_distribution<double> dist(mean, stddev);

  // Add Gaussian noise
  for (size_t i = 0; i < PCL.rows(); i++)
  {
    for (size_t j = 0; j < PCL.cols(); j++)
    {
      PCL(i, j) += dist(generator);
    }
  }

  return;
}

void normalizePCL(EigenPCL &PCL)
{
  Eigen::Vector3d S_A = PCL.colwise().mean();
  PCL = PCL.rowwise() - S_A.transpose();
}

void generatePair(EigenPCL &Original, EigenPCL &Output, Eigen::Matrix4d transformation, double noiseStd, double overlap)
{
}

int main(int argc, char **argv)
{
  bool visualize = true;
  double noiseStd = 5;
  double overlapInit = 0.9;
  double overlapICP = overlapInit;
  double dx = 40;
  double roll = 0.2;
  int maxIter = 100; 

  cv::Mat D1;
  cv::Mat D2;
  std::string datasetName = "Art";
  fs::path data_path("data");
  data_path = data_path / datasetName;

  fs::path d1_path = data_path / "disp1.png";
  // fs::path d2_path = data_path / "disp5.png";

  D1 = cv::imread(d1_path.string(), 0);
  // D2 = cv::imread(d2_path.string(), 0);

  EigenPCL PCL1;
  EigenPCL PCL2;

  Disparity2PointCloud(PCL1, D1, 200, 160, 3740);
  // Disparity2PointCloud(PCL2, D2, 200, 160, 3740);

  normalizePCL(PCL1);
  EigenPCL PCL3(PCL1);


  addGausiianNoise(PCL1, noiseStd, 1);
  // addGausiianNoise(PCL2, noiseStd);
  addGausiianNoise(PCL3, noiseStd, 2);

  // init PCL
  Eigen::Affine3d r = create_rotation_matrix(roll, 0, 0);
  Eigen::Affine3d t(Eigen::Translation3d(Eigen::Vector3d(dx, 0, 0)));
  Eigen::Matrix4d t_init = (t * r).matrix(); // Option 1
  
  std::cout << "True Transformation: " << std::endl;
  std::cout << t_init << std::endl;

  

  PCL3 = PCL3.topRows(PCL3.rows() * overlapInit).eval();
  PCL1 = PCL1.bottomRows(PCL1.rows() * overlapInit).eval();

  my_icp::applyTransformation(PCL3, t_init);

  Matrix4<double> T;
  T = my_icp::ICPtrimmed(PCL1, PCL3, overlapICP, maxIter);

  std::cout << "estimated Transformation: " << std::endl;
  std::cout << T.inverse() << std::endl;

  EigenPCL result(PCL3);
  my_icp::applyTransformation(result, T);

  if (visualize)
  {
    my_vis::vis_point_clouds(PCL1, result);
  }

  // my_vis::vis_point_clouds(a, result);

  return 0;
}