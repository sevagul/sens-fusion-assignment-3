#include "../nanoflann.hpp"
// #include "include/visualizations.hpp"
#include "my_visualizations.hpp"
#include "my_types.h"

#include <cstdlib>
#include <ctime>
#include <iostream>
#include <opencv2/opencv.hpp>

#include <boost/filesystem.hpp>
#include <Eigen/Dense>


namespace fs = boost::filesystem;

using namespace Eigen;
using namespace std;
using namespace nanoflann;

const int SAMPLES_DIM = 15;
// using PCL_type = Eigen::Matrix<double, Eigen::Dynamic, 3>;

// template <typename Der>
// int findClosestIndex(Vector3<Der> &query_pt, const EigenPCLMat<Der> &pc)
// {
//   using PC_type = Eigen::Matrix<Der, Eigen::Dynamic, 3>;
//   using my_kd_tree_t = KDTreeEigenMatrixAdaptor<PC_type, 3, nanoflann::metric_L2>;


//   my_kd_tree_t my_tree(PC_type::ColsAtCompileTime, std::cref(pc));
//   my_tree.index->buildIndex();

//   size_t closest_index;
//   float closest_sqdist;

//   nanoflann::KNNResultSet<float> result(1);
//   // resultSet.init(&ret_indexes[0], &out_dists_sqr[0]);
//   result.init(&closest_index, &closest_sqdist);

//   my_tree.index->findNeighbors(result, &query_pt.x(),
//                                nanoflann::SearchParams());
//   std::cout << "closest sq_dist: " << closest_sqdist << std::endl;
//   return int(closest_index);
// }

template <typename Der>
int findClosestIndex(Vector3<Der> &query_pt, const MyKdTree<EigenPCLMat<Der>> &my_tree)
{

  size_t closest_index;
  Der closest_sqdist;

  nanoflann::KNNResultSet<Der> result(1);
  result.init(&closest_index, &closest_sqdist);

  my_tree.index->findNeighbors(result, &query_pt.x(),
                               nanoflann::SearchParams());
  std::cout << "closest sq_dist: " << closest_sqdist << std::endl;
  return int(closest_index);
}



template <typename Der>
Matrix4<Der> best_fit_transform(const EigenPCLMat<Der> &A, const EigenPCLMat<Der> &B)
{
  /*
  Notice:
  1/ JacobiSVD return U,S,V, S as a vector, "use U*S*Vt" to get original Matrix;
  2/ matrix type 'MatrixXd' or 'MatrixXf' matters.
  */
  Matrix4<Der> T = MatrixX<Der>::Identity(4, 4);
  Vector3<Der> centroid_A(0, 0, 0);
  Vector3<Der> centroid_B(0, 0, 0);
  MatrixX<Der> AA = A;
  MatrixX<Der> BB = B;
  int row = A.rows();

  for (int i = 0; i < row; i++)
  {
    centroid_A += (A.block<1, 3>(i, 0)).transpose();
    centroid_B += (B.block<1, 3>(i, 0)).transpose();
  }
  centroid_A /= row;
  centroid_B /= row;
  for (int i = 0; i < row; i++)
  {
    AA.block<1, 3>(i, 0) = A.block<1, 3>(i, 0) - centroid_A.transpose();
    BB.block<1, 3>(i, 0) = B.block<1, 3>(i, 0) - centroid_B.transpose();
  }

  MatrixX<Der> H = AA.transpose() * BB;
  MatrixX<Der> U;
  Eigen::VectorXd S;
  MatrixX<Der> V;
  MatrixX<Der> Vt;
  Matrix3<Der> R;
  Vector3<Der> t;

  JacobiSVD<MatrixX<Der>> svd(H, ComputeFullU | ComputeFullV);
  U = svd.matrixU();
  S = svd.singularValues();
  V = svd.matrixV();
  Vt = V.transpose();

  R = Vt.transpose() * U.transpose();

  if (R.determinant() < 0)
  {
    (Vt.block<1, 3>(2, 0)) *= -1;
    R = Vt.transpose() * U.transpose();
  }

  t = centroid_B - R * centroid_A;

  (T.block<3, 3>(0, 0)) = R;
  (T.block<3, 1>(0, 3)) = t;
  return T;
}

// template <typename Der>
void Disparity2PointCloud(
    Eigen::Matrix<double, Eigen::Dynamic, 3> &pcl, cv::Mat &disparities,
    const int &dmin, const double &baseline = 160, const double &focal_length = 3740)
{
  int rows = disparities.rows;
  int cols = disparities.cols;
  std::vector<std::vector<double>> points_vec;
  for (int r = 0; r < rows; ++r)
  {
    std::cout << "Reconstructing 3D point cloud from disparities... " << std::ceil(((r) / static_cast<double>(rows + 1)) * 100) << "%\r" << std::flush;
    // #pragma omp parallel for
    for (int c = 0; c < cols; ++c)
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
  bool visualize = false;
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

  // visdclouulaize pointcloud
  if (visualize)
  {
    my_vis::vis_point_clouds(PCL1, PCL2);
  }

  auto querySet = PCL2;
  MyKdTree<EigenPCL> my_tree(EigenPCL::ColsAtCompileTime, std::cref(PCL1));
  my_tree.index->buildIndex();


  for (size_t i = 0; i < 100; i++)
  {
    size_t query_index = 3;
    Eigen::Vector3d pt_query = PCL1.row(query_index);
    int closestIndex = findClosestIndex(pt_query, my_tree);
    std::cout << "closest_index: " << closestIndex << std::endl;
    std::cout << "point: " << pt_query << std::endl;
    std::cout << "closest point: " << querySet.row(closestIndex) << std::endl;
  }
  

  return 0;
}