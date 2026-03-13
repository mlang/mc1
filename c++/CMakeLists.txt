set(PYBIND11_FINDPYTHON ON)
find_package(pybind11 CONFIG REQUIRED)
find_package(Boost REQUIRED CONFIG)

pybind11_add_module(_core MODULE compiler.cpp dag.cpp audio.cpp runtime.cpp python.cpp)
target_include_directories(_core PRIVATE ${Boost_INCLUDE_DIRS})
target_link_libraries(_core PRIVATE gccjit)
install(TARGETS _core DESTINATION ${SKBUILD_PROJECT_NAME})
