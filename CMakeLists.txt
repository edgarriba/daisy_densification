cmake_minimum_required(VERSION 2.8.7)

project(daisy_densification)

include_directories(include)

find_package(OpenCV REQUIRED)
message(STATUS "OpenCV version: ${OpenCV_VERSION}")
message(STATUS "OpenCV headers: ${OpenCV_INCLUDE_DIRS}")

add_library(depth src/robust_matcher.cpp)

add_executable(main src/main.cpp)
target_link_libraries(main depth ${OpenCV_LIBS})
