set(PYBIND11_FINDPYTHON ON)
find_package(pybind11 CONFIG REQUIRED)

pybind11_add_module(_core MODULE compiler.cpp dag.cpp audio.cpp python.cpp)
target_link_libraries(_core PRIVATE gccjit)
install(TARGETS _core DESTINATION ${SKBUILD_PROJECT_NAME})
