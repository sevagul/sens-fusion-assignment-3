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

Eigen::Affine3d create_rotation_matrix(double ax, double ay, double az) {
  Eigen::Affine3d rx =
      Eigen::Affine3d(Eigen::AngleAxisd(ax, Eigen::Vector3d(1, 0, 0)));
  Eigen::Affine3d ry =
      Eigen::Affine3d(Eigen::AngleAxisd(ay, Eigen::Vector3d(0, 1, 0)));
  Eigen::Affine3d rz =
      Eigen::Affine3d(Eigen::AngleAxisd(az, Eigen::Vector3d(0, 0, 1)));
  return rz * ry * rx;
}

void applyTransformation(const EigenPCLMat<double> &PCL, const Eigen::Matrix4d& T, EigenPCLMat<double> &out)
    {
        const int rows = int(PCL.rows());
        Eigen::Matrix<double, Eigen::Dynamic, -1> ones = Eigen::MatrixXd::Constant(rows, 1, 1.0);
        // ones.setConstant(1);
        Eigen::Matrix<double, Eigen::Dynamic, -1> PCL_hom(rows, 4);
        PCL_hom << PCL, ones;
        std::cout << PCL_hom.topRows(5) << std::endl;
        std::cout << T << std::endl;
        PCL_hom = PCL_hom * (T.transpose() );
        out << PCL_hom.leftCols<3>();
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

  // Matrix4<double> T_init;
  // Eigen::Affine3f m;     m  = Eigen::AngleAxisd(0.5, Eigen::Vector3d(1.0, 0.0, 0.0));
  Eigen::Affine3d r = create_rotation_matrix(0.00, 0, 0);
  Eigen::Affine3d t(Eigen::Translation3d(Eigen::Vector3d(3000,0,300)));

  Eigen::Matrix4d t_init = (t * r).matrix(); // Option 1
  std::cout << t_init << std::endl;

  // T_init.block<0, 0>(3, 3) = m.matrix();

  Disparity2PointCloud(PCL1, D1, 200, 160, 3740);
  Disparity2PointCloud(PCL2, D2, 200, 160, 3740);

  Eigen::Vector3d S_A = PCL1.colwise().mean();
  // std::cout << S_A << std::endl;

  PCL1 = PCL1.rowwise() - S_A.transpose();
  

  EigenPCL PCL3(PCL1);
  applyTransformation( PCL1, t_init, PCL3);
  // std::cout << PCL3.topRows(5) << std::endl;

  // auto a = PCL3.topRows(PCL3.rows()*1.0).eval();
  // auto b = PCL1.bottomRows(PCL1.rows()*1.0).eval();
  
  // PCL1 = PCL1.bottomRows(PCL1.rows()*0.9);


  Matrix4<double> T;
  T = my_icp::ICP(PCL1, PCL3, 100);
  // T = my_icp::best_fit_transform(PCL1, PCL3);
  std::cout << "Resulting: " << std::endl;
  std::cout << T << std::endl;
  //  MyKdTree<EigenPCL> my_tree(EigenPCL::ColsAtCompileTime, std::cref(PCL1));
  // my_tree.index->buildIndex();
  // double avg_dist = 0;
  // T = my_icp::ICPiter(PCL1, PCL3, my_tree, avg_dist);
  // std::cout << T << std::endl;

  EigenPCL result(PCL3);
  my_icp::applyTransformation(result, T);

  my_vis::vis_point_clouds(PCL1, result);

  
  // my_vis::vis_point_clouds(a, result);

  return 0;
}