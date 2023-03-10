cmake_minimum_required(VERSION 2.8)
project( Hello_Nanoflann )

#find_package(nanoflann)
find_package(PCL REQUIRED COMPONENTS visualization)
include_directories(${PCL_INCLUDE_DIRS})
include_directories(include)
link_directories(${PCL_LIBRARY_DIRS})
add_definitions(${PCL_DEFINITIONS})

find_package(Eigen3)
find_package(OpenMP)
find_package (OpenCV REQUIRED )
FIND_PACKAGE(Boost COMPONENTS program_options filesystem REQUIRED)
INCLUDE_DIRECTORIES(${Boost_INCLUDE_DIRS})

add_executable(TrICP src/main.cpp src/my_icp.cpp)
target_include_directories(TrICP PRIVATE ${EIGEN3_INCLUDE_DIR})
target_link_libraries(TrICP  ${OpenCV_LIBS} ${Boost_LIBRARIES} ${PCL_LIBRARIES}) #nanoflann::nanoflann

if(OpenMP_CXX_FOUND)
    target_link_libraries(TrICP OpenMP::OpenMP_CXX)
endif()
# READ MORE:
#   https://github.com/jlblancoc/nanoflann#11-obtaining-the-code
#   https://github.com/jlblancoc/nanoflann#17-use-in-your-project-via-cmake