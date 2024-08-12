cmake_minimum_required(VERSION 3.25)
project(MiniCollider VERSION 1 LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED TRUE)

find_package(Boost)

add_executable(engine engine/dsp.cc)
target_link_libraries(engine gccjit)
