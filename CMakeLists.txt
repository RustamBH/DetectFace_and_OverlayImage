cmake_minimum_required(VERSION 3.5)

project(DetectFace)

# Пытаемся найти предустановленный OPENCV
find_package(OpenCV)

if (OPENCV_FOUND)
    message(STATUS "The OpenCV installed in the system will be used")
else ()
    # Если OpenCV не найден - выкачиваем 
    message(STATUS "Get OpenCV from the https://github.com/opencv/opencv/releases")
    include(FetchContent)
    FetchContent_Declare(OPENCV
            GIT_REPOSITORY https://github.com/opencv/opencv/releases
            GIT_TAG bff38466091c313dac21a0b73eea8278316a89c1d434c6f0b10697e087670168)
    FetchContent_MakeAvailable(OPENCV)
endif ()

add_executable(FaceRcg	FaceRcg.cpp)

target_link_libraries(physics PRIVATE opencv_world4100d.dll)

set_target_properties(FaceRcg PROPERTIES CXX_STANDARD 17 CXX_STANDARD_REQUIRED ON)