cmake_minimum_required(VERSION 3.20)
project(hello_cpp LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)


add_executable(hello_cpp src/main.cpp)

find_package(Armadillo REQUIRED)
find_package(PkgConfig REQUIRED)

pkg_check_modules(FFTW3 REQUIRED fftw3)
pkg_check_modules(MATIO REQUIRED matio)



target_link_libraries(hello_cpp PRIVATE ${FFTW3_LIBRARIES})
target_include_directories(hello_cpp PRIVATE ${FFTW3_INCLUDE_DIRS})
target_include_directories(hello_cpp PRIVATE ${MATIO_INCLUDE_DIRS})
target_link_directories(hello_cpp PRIVATE ${MATIO_LIBRARY_DIRS})
target_link_libraries(hello_cpp PRIVATE ${MATIO_LIBRARIES})

target_link_libraries(hello_cpp PRIVATE armadillo)