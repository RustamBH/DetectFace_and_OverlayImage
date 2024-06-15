cmake_minimum_required(VERSION 3.5)

project(DetectFace)

set(OpenCV_DIR "C:/opencv/build")

# Пытаемся найти предустановленный OpenCV
find_package(OpenCV REQUIRED PATHS "${OpenCV_DIR}" NO_DEFAULT_PATH)
#find_package(OpenCV REQUIRED)

if (OPENCV_FOUND)
    message(STATUS "The OpenCV installed in the system will be used")
else ()
    # Если OpenCV не найден - выкачиваем 
    message(STATUS "The OpenCV not installed")
endif ()

add_executable(FaceRcg FaceRcg.cpp)

# Add src to the include directories for the test target
target_include_directories(
    FaceRcg PRIVATE
    "${OpenCV_DIR}/include"
)

target_link_libraries(FaceRcg PRIVATE ${OpenCV_DIR}/x64/vc16/lib/opencv_world4100d.lib)

set_target_properties(FaceRcg PROPERTIES CXX_STANDARD 17 CXX_STANDARD_REQUIRED ON)