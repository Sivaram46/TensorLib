cmake_minimum_required(VERSION 3.1)
project(TensorLib)
set(CMAKE_BUILD_TYPE Release)

# Bring include directories in the project
include_directories(TensorLib)

# Add source files
file(GLOB SOURCES TensorLib/*.cpp)

# Generate a shared library from the source files
add_library(TensorLib SHARED ${SOURCES})