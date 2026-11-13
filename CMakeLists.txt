cmake_minimum_required(VERSION 3.18)
project(ML_CUDA_Course_Project LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE Debug CACHE STRING "Build type" FORCE)
endif()

find_package(CUDAToolkit REQUIRED)

# Only include reproduced examples under ml-course.
add_subdirectory(course2)
add_subdirectory(course3)
add_subdirectory(course4)
add_subdirectory(course5)

