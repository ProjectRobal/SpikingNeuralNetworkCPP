cmake_minimum_required(VERSION 3.22)

project(Kolejka)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=native -g -Wall -std=c++20")

include_directories("${CMAKE_CURRENT_SOURCE_DIR}/include/")
include_directories("${CMAKE_CURRENT_SOURCE_DIR}/sources/")

file(GLOB all_SRCS
        "${PROJECT_SOURCE_DIR}/include/*.h"
        "${PROJECT_SOURCE_DIR}/include/*.hpp"
        "${PROJECT_SOURCE_DIR}/sources/*.cpp"
        "${PROJECT_SOURCE_DIR}/sources/*.c"
        )

add_executable(main main.cpp ${all_SRCS})
