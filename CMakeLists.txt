cmake_minimum_required(VERSION 2.8.9)
set(PROJECT_NAME_STR UnscentedKalmanFilter)
project(${PROJECT_NAME_STR})

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g -Wall -std=c++11")

file(GLOB SOURCES "${PROJECT_SOURCE_DIR}/src/*.cpp")

find_package(GTest REQUIRED)
include_directories("${GTEST_INCLUDE_DIR}")

set(EIGEN3_INCLUDE_DIR /opt/local/include/eigen3/Eigen/) # Change this as necessary
include_directories("${EIGEN3_INCLUDE_DIR}")

set(PROJECT_TEST_NAME ${PROJECT_NAME_STR}_test)
file(GLOB TEST_SOURCES "${PROJECT_SOURCE_DIR}/test/*.cpp")

add_executable(${PROJECT_TEST_NAME} ${SOURCES} ${TEST_SOURCES})
target_link_libraries(${PROJECT_TEST_NAME} ${GTEST_BOTH_LIBRARIES})
target_compile_definitions(${PROJECT_TEST_NAME} PUBLIC TESTING)
