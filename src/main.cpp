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

#include <boost/program_options.hpp>

#include <fstream>

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

float rotDiff(Matrix4<double> T1, Matrix4<double> T2)
{
  Quaterniond q1(T1.block<3, 3>(0, 0));
  Quaterniond q2(T2.block<3, 3>(0, 0));
  double diff = (q1.coeffs() - q2.coeffs()).norm();
  return diff;
}

double translationDiff(Matrix4<double> T1, Matrix4<double> T2)
{
  Matrix4<double> Tdiff = T1 * T2.inverse();
  double diff = (Tdiff.block<3, 1>(0, 3)).norm() / (T1.block<3, 1>(0, 3)).norm();
  return diff;
}

namespace po = boost::program_options;

int main(int argc, char **argv)
{
  bool visualize = true;
  bool visualizeSetup = false;
  double noiseStd = 5;
  double overlapInit = 0.9;
  double overlapICP = overlapInit;
  int dx = 40;
  double roll = 0.2;
  int maxIter = 100;
  std::string outputFileStr = "output.txt";

  po::options_description command_line_options("cli options");
  command_line_options.add_options()("help,h", "Produce help message")("shift-x,x", po::value<int>(&dx)->default_value(0), "shift in x to the original PCL")("visualize-setup,V", "Visualize generated pointclouds pointcloud")("visualize,v", "Visualize aligned pointcloud")("noise-std,n", po::value<double>(&noiseStd)->default_value(noiseStd), "Standard deviation of the applied noise")("roll,r", po::value<double>(&roll)->default_value(0), "roll applied to the original PCL")("max-iter,m", po::value<int>(&maxIter)->default_value(100), "maximum amount of iterations")("ovarlap-init,O", po::value<double>(&overlapInit)->default_value(1), "% of overlap for the generated PCL with the original")("ovarlap-icp,o", po::value<double>(&overlapICP)->default_value(1), "% of overlap for the trimmed ICP algo")("output,t", po::value<std::string>(&outputFileStr)->default_value("output.txt"), "name of the output file in output folder");

  po::variables_map vm;
  po::options_description cmd_opts;
  cmd_opts.add(command_line_options);
  po::store(po::command_line_parser(argc, argv).options(cmd_opts).run(), vm);
  po::notify(vm);

  if (vm.count("help"))
  {
    std::cout << "Usage: TrICP [<options>]\n";
    po::options_description help_opts;
    help_opts.add(command_line_options);
    std::cout << help_opts << "\n";
    return 1;
  }
  if (vm.count("visualize-setup"))
  {
    visualizeSetup = true;
  }
  else
  {
    visualizeSetup = false;
  }
  if (vm.count("visualize"))
  {
    visualize = true;
  }
  else
  {
    visualize = false;
  }
  std::cout << "Shift x: " << dx << std::endl;
  std::cout << "visualize: " << visualize << std::endl;
  std::cout << "noiseStd: " << noiseStd << std::endl;
  std::cout << "overlapInit: " << overlapInit << std::endl;
  std::cout << "overlapICP: " << overlapICP << std::endl;
  std::cout << "dx: " << dx << std::endl;
  std::cout << "roll: " << roll << std::endl;
  std::cout << "maxIter: " << maxIter << std::endl;

  fs::path outFile = ("output");
  outFile /= outputFileStr;

  std::cout << "Output File: " << outFile << std::endl;

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
  // addGausiianNoise(PCL2, noiseStd, 2);
  addGausiianNoise(PCL3, noiseStd, 3);

  // init PCL
  Eigen::Affine3d r = create_rotation_matrix(roll, 0, 0);
  Eigen::Affine3d t(Eigen::Translation3d(Eigen::Vector3d(dx, 0, 0)));
  Eigen::Matrix4d t_init = (t * r).matrix(); // Option 1

  std::cout << "True Transformation: " << std::endl;
  std::cout << t_init << std::endl;

  PCL3 = PCL3.topRows(PCL3.rows() * overlapInit).eval();
  PCL1 = PCL1.bottomRows(PCL1.rows() * overlapInit).eval();

  my_icp::applyTransformation(PCL3, t_init);

  if (visualizeSetup)
  {
    my_vis::vis_point_clouds(PCL1, PCL3);
  }

  double avgDistance = -1;

  Matrix4<double> T;
  int nIter=0;
  T = my_icp::ICPtrimmed(PCL1, PCL3, overlapICP, avgDistance, nIter, maxIter);

  std::cout << "Estimated Transformation: " << std::endl;
  std::cout << T.inverse() << std::endl;

  double rotErr = rotDiff(T.inverse(), t_init);
  double trErr = translationDiff(T.inverse(), t_init);

  std::cout << "Rotational Error: " << rotErr << std::endl;
  std::cout << "Translational Error: " << trErr << std::endl;

  EigenPCL result(PCL3);
  my_icp::applyTransformation(result, T);

  ofstream myfile;
  myfile.open(outFile.string());
  myfile << "GeneratingParams: " << noiseStd << ", " << dx << ", " << roll << ", " << overlapInit << std::endl;
  myfile << "ICP_params: " << overlapICP << ", " << maxIter << std::endl;
  myfile << "Metrics: " << rotErr << ", " << trErr << ", " << avgDistance << ", " << nIter << std::endl;
  myfile.close();

  if (visualize)
  {
    my_vis::vis_point_clouds(PCL1, result);
  }

  return 0;
}