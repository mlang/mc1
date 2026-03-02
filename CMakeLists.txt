cmake_minimum_required(VERSION 3.25)
project(MiniCollider VERSION 1 LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED TRUE)

find_package(Boost CONFIG REQUIRED)
find_package(nlohmann_json REQUIRED)

add_executable(engine mc1/compiler.cpp mc1/dag.cpp mc1/engine.cpp)
target_link_libraries(engine gccjit nlohmann_json::nlohmann_json)

add_executable(dag2json test/dag2json.cpp mc1/dag.cpp)
target_link_libraries(dag2json nlohmann_json::nlohmann_json)

enable_testing()
add_test(NAME mypy COMMAND mypy -p mc1
  WORKING_DIRECTORY "${PROJECT_SOURCE_DIR}"
)
add_test(NAME pytest COMMAND pytest
  WORKING_DIRECTORY "${PROJECT_SOURCE_DIR}"
)
