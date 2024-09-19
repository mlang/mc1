cmake_minimum_required(VERSION 3.25)
project(MiniCollider VERSION 1 LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED TRUE)

find_package(Boost REQUIRED)
find_package(PkgConfig REQUIRED)
pkg_check_modules(PipeWire REQUIRED IMPORTED_TARGET libpipewire-0.3)

add_executable(engine mc1/dag.cpp mc1/engine.cpp)
target_link_libraries(engine gccjit PkgConfig::PipeWire)

enable_testing()
add_test(NAME mypy COMMAND mypy -p mc1
  WORKING_DIRECTORY "${PROJECT_SOURCE_DIR}"
)
