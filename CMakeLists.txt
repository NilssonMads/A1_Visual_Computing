cmake_minimum_required(VERSION 3.16)
project(PanoramaStitcher LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

cmake_policy(SET CMP0074 NEW)

find_package(OpenCV REQUIRED)

add_executable(PanoramaStitcher main.cpp)

target_link_libraries(PanoramaStitcher PRIVATE
    opencv_core
    opencv_imgproc
    opencv_imgcodecs
    opencv_highgui
    opencv_features2d
    opencv_calib3d
    opencv_stitching
)
