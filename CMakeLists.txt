cmake_minimum_required(VERSION 3.10)
project(OpenCVTest)

# set(OpenCV_DIR "/home/ubuntu/libs/opencv-4.12")
set(OpenCV_DIR "C:\\lib\\opencv-4.12.0\\build\\x64\\vc16\\lib")

find_package(OpenCV REQUIRED)

set(SRC 
    src/ObjectDetector.hpp 
    src/ObjectDetector.cpp
    src/utils.h 
    src/utils.cpp
    src/main.cpp
)

add_executable(OpenCVTest ${SRC})
target_compile_options(OpenCVTest PRIVATE /O2)
target_link_libraries(OpenCVTest ${OpenCV_LIBS})