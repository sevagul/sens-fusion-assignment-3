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
#include <my_icp.h>
#include <limits>

namespace fs = boost::filesystem;

using namespace Eigen;
using namespace std;
using namespace nanoflann;

namespace my_icp
{
    template <typename Der>
    int findClosestIndex(Vector3<Der> &query_pt, const MyKdTree<EigenPCLMat<Der>> &my_tree, Der &closest_sqdist)
    {
        size_t closest_index;

        nanoflann::KNNResultSet<Der> result(1);
        result.init(&closest_index, &closest_sqdist);

        my_tree.index->findNeighbors(result, &query_pt.x(),
                                     nanoflann::SearchParams());

        return int(closest_index);
    }

    // template <typename Der>
    Matrix4<double> best_fit_transform(const EigenPCLMat<double> &A, const EigenPCLMat<double> &B)
    {
        /*
        Notice:
        1/ JacobiSVD return U,S,V, S as a vector, "use U*S*Vt" to get original Matrix;
        2/ matrix type 'MatrixXd' or 'MatrixXf' matters.
        */
        Matrix4<double> T = MatrixX<double>::Identity(4, 4);
        Vector3<double> centroid_A(0, 0, 0);
        Vector3<double> centroid_B(0, 0, 0);
        MatrixX<double> AA = A;
        MatrixX<double> BB = B;
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

        MatrixX<double> H = AA.transpose() * BB;
        MatrixX<double> U;
        Eigen::VectorXd S;
        MatrixX<double> V;
        MatrixX<double> Vt;
        Matrix3<double> R;
        Vector3<double> t;

        JacobiSVD<MatrixX<double>> svd(H, ComputeFullU | ComputeFullV);
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

        std::cout << "Best transform: " << std::endl << T << std::endl;
        return T;
    }

    Matrix4<double> ICPiter(const EigenPCLMat<double> &PCL1, const EigenPCLMat<double> &PCL2, MyKdTree<EigenPCL> &my_tree, double &avg_distance)
    {
        auto querySet = PCL2;
        EigenPCL PCL3(PCL2);

        std::vector<double> distances(querySet.rows());
        std::vector<int32_t> inds(querySet.rows());
        int rows = querySet.rows();

#pragma omp parallel for
        for (long long int i = 0; i < rows; i++)
        {
            int query_index = i;
            Eigen::Vector3d pt_query = querySet.row(query_index);
            double cur_closest_distance;
            int closestIndex = findClosestIndex(pt_query, my_tree, cur_closest_distance);
            distances[i] = cur_closest_distance;
            inds[i] = closestIndex;
            PCL3.row(i) = PCL1.row(closestIndex);
        }

        avg_distance = 0;
        double cur_closest_distance;
        for (size_t i = 0; i < distances.size(); i++)
        {
            cur_closest_distance = distances[i];
            avg_distance = avg_distance * i / (i + 1) + cur_closest_distance / (i + 1);
        }

        std::cout << "Avg closest distance: " << avg_distance << std::endl;

        Matrix4<double> T;
        T = best_fit_transform(PCL2, PCL3);
        return T;
    }

    void applyTransformation(EigenPCLMat<double> &PCL, const Eigen::Matrix4d T)
    {
        const int rows = int(PCL.rows());
        Eigen::Matrix<double, Eigen::Dynamic, -1> ones(rows, 1);
        ones.setConstant(1);
        Eigen::Matrix<double, Eigen::Dynamic, -1> PCL_hom(rows, 4);
        PCL_hom << PCL, ones;
        PCL_hom = PCL_hom * (T.transpose());
        PCL << PCL_hom.leftCols<3>().eval();
    }

    Matrix4<double> ICP(const EigenPCLMat<double> &PCL1, const EigenPCLMat<double> &PCL2_in, int max_iter)
    {
        MyKdTree<EigenPCL> my_tree(EigenPCL::ColsAtCompileTime, std::cref(PCL1));
        my_tree.index->buildIndex();
        Matrix4<double> T_res = Matrix4<double>::Identity(4, 4);
        Matrix4<double> T = Matrix4<double>::Identity(4, 4);
        EigenPCLMat<double> PCL2(PCL2_in);

        double avg_distance = std::numeric_limits<double>::max();
        double decrease_th = 0.05;
        size_t i;
        for (i = 0; i < max_iter; i++)
        {
            double prev_dist = avg_distance;
            T = ICPiter(PCL1, PCL2, my_tree, avg_distance);
            if (avg_distance > prev_dist)
            {
                std::cout << "Distance increased!" << std::endl;
                return T_res;
            }
            T_res = T * T_res ;
            
            applyTransformation(PCL2, T);

            if (prev_dist / avg_distance < 1 + decrease_th)
            {
                std::cout << "Converged!" << std::endl;
                break;
            }
        }
        if (i == max_iter)
        {
            std::cout << "Not Converged! Reached maximum iterations" << std::endl;
        }


        return T_res;
    }

    std::vector<int> findMinValInds(std::vector<double> &dists, const int &N)
    {
        std::vector<int> indices(dists.size());
        for (int i = 0; i < dists.size(); i++){
            indices[i] = i;
        }

        std::partial_sort(indices.begin(), indices.begin() + N, indices.end(),
                          [&dists](int i, int j)
                          { return dists[i] < dists[j]; });

        return std::vector<int>(indices.begin(), indices.begin() + N);
    }

    void extractRows(const Eigen::Matrix<double, Eigen::Dynamic, 3> &matrix, Eigen::Matrix<double, Eigen::Dynamic, 3> &subMatrix, std::vector<int> &rows)
    {
        subMatrix.resize(rows.size(), Eigen::NoChange);
        for (size_t i = 0; i < rows.size(); i++)
        {
            subMatrix.row(i) = matrix.row(rows[i]).eval();
        }
    }

    Matrix4<double> ICPtrimmedIter(const EigenPCLMat<double> &PCL1, const EigenPCLMat<double> &PCL2, MyKdTree<EigenPCL> &my_tree, double &avg_distance, double overlap)
    {

        int Noverlap = PCL2.rows() * overlap;
        std::cout << "N overlapping points: " << Noverlap << std::endl;

        EigenPCL PCL3(PCL2);


        int rows = PCL2.rows();
        std::vector<double> distances(rows);
        std::vector<int32_t> inds(rows);

#pragma omp parallel for
        for (long long int i = 0; i < rows; i++)
        {
            Eigen::Vector3d pt_query = PCL2.row(i);
            double cur_closest_distance;
            int closestIndex = findClosestIndex(pt_query, my_tree, cur_closest_distance);
            distances[i] = cur_closest_distance;
            inds[i] = closestIndex;
            PCL3.row(i) = PCL1.row(closestIndex);
        }
        std::vector<int> minValInds = findMinValInds(distances, Noverlap);

        EigenPCL PCL2_sub;
        EigenPCL PCL3_sub;

        extractRows(PCL2, PCL2_sub, minValInds);
        extractRows(PCL3, PCL3_sub, minValInds);

        avg_distance = 0;
        double cur_closest_distance;
        for (size_t i = 0; i < minValInds.size(); i++)
        {
            cur_closest_distance = distances[minValInds[i]];
            avg_distance = avg_distance * i / (i + 1) + cur_closest_distance / (i + 1);
        }

        std::cout << "Avg closest distance: " << avg_distance << std::endl;

        Matrix4<double> T;
        T = best_fit_transform(PCL2_sub, PCL3_sub);
        return T;
    }

    Matrix4<double> ICPtrimmed(const EigenPCLMat<double> &PCL1, const EigenPCLMat<double> &PCL2_in, double overlap, int max_iter)
    {

        MyKdTree<EigenPCL> my_tree(EigenPCL::ColsAtCompileTime, std::cref(PCL1));
        my_tree.index->buildIndex();
        Matrix4<double> T_res = Matrix4<double>::Identity(4, 4);
        Matrix4<double> T = Matrix4<double>::Identity(4, 4);
        EigenPCLMat<double> PCL2(PCL2_in);

        double avg_distance = std::numeric_limits<double>::max();
        double decrease_th = 0.05;
        size_t i;
        for (i = 0; i < max_iter; i++)
        {
            double prev_dist = avg_distance;
            T = ICPtrimmedIter(PCL1, PCL2, my_tree, avg_distance, overlap);
            // T = ICPtrimmedIter(PCL1, PCL2, my_tree, avg_distance);
            if (avg_distance > prev_dist)
            {
                std::cout << "Distance increased!" << std::endl;
                return T;
            }
            T_res = T * T_res;
            const int rows = int(PCL2.rows());
            Eigen::Matrix<double, Eigen::Dynamic, -1> ones(rows, 1);
            ones.setConstant(1);
            Eigen::Matrix<double, Eigen::Dynamic, -1> PCL2_hom(rows, 4);
            PCL2_hom << PCL2, ones;
            PCL2_hom = PCL2_hom * (T);
            PCL2 << PCL2_hom.leftCols<3>().eval();
            if (prev_dist / avg_distance < 1 + decrease_th)
            {
                std::cout << "Converged!" << std::endl;
                break;
            }
        }
        if (i == max_iter)
        {
            std::cout << "Not Converged! Reached maximum iterations" << std::endl;
        }

        return T_res;
    }

}
