
#ifndef MY_VIS_HPP
#define MY_VIS_HPP

#include <pcl/point_types.h>
#include <pcl/visualization/cloud_viewer.h>
#include <thread>
#include <cstdlib>
#include <ctime>
#include <iostream>
#include "my_types.h"
#include "my_conversions.h"

using namespace std::chrono_literals;


// using EigenPCL = Eigen::Matrix<double, Eigen::Dynamic, 3>;

namespace my_vis
{
    inline void vis_point_cloud(const EigenPCL &pclToVis)
    {
        // using PointType = pcl::PointXYZ;
        // using PclType = pcl::PointCloud<PointType>;
        PclType pcl_vis;

        (pclToVis, pcl_vis, 0, 255);

        PclType::Ptr ptrCloud(&pcl_vis);

        pcl::visualization::CloudViewer viewer("Simple Cloud Viewer");
        pcl::visualization::PointCloudColorHandlerRGBField<pcl::PointXYZRGB> rgb(ptrCloud);
        viewer.showCloud(ptrCloud, "cloud");

        while (!viewer.wasStopped())
        {
        }
    }

    inline void vis_point_clouds(const EigenPCL &pcl1, const EigenPCL &pcl2)
    {
        PclType pcl1_vis;
        my_conv::EigenToPcl(pcl1, pcl1_vis, 0, 255);
        PclType::Ptr ptrCloud1(&pcl1_vis);

        PclType pcl2_vis;
        my_conv::EigenToPcl(pcl2, pcl2_vis, 255, 0);
        PclType::Ptr ptrCloud2(&pcl2_vis);

        pcl::visualization::PCLVisualizer::Ptr viewer(new pcl::visualization::PCLVisualizer("3D Viewer"));
        viewer->setBackgroundColor(0, 0, 0);
        pcl::visualization::PointCloudColorHandlerRGBField<pcl::PointXYZRGB> rgb(ptrCloud1);
        viewer->addPointCloud<pcl::PointXYZRGB>(ptrCloud1, rgb, "cloud1");
        viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 3, "cloud1");
        pcl::visualization::PointCloudColorHandlerRGBField<pcl::PointXYZRGB> rgb2(ptrCloud2);
        viewer->addPointCloud<pcl::PointXYZRGB>(ptrCloud2, rgb2, "cloud2");
        viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 3, "cloud2");
        
        while (!viewer->wasStopped())
        {
            viewer->spinOnce(100);
            std::this_thread::sleep_for(100ms);
        }
        viewer->close();
        
    }
}

#endif
