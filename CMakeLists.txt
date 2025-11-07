cmake_minimum_required(VERSION 3.22)

find_package(Python)

if(Python_FOUND)
    set(PYTHON_PATH ${Python_EXECUTABLE})
else()
    message(FATAL_ERROR "Python not found. Please install Python or set the Python_ROOT_DIR to proceed.")
endif()

project(ocarina)
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake/")
include(CopyTargetOutputs)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED True)
include_directories(src)

set(CMAKE_FIND_PACKAGE_SORT_ORDER NATURAL)
set(CMAKE_FIND_PACKAGE_SORT_DIRECTION DEC)

add_subdirectory(src)
