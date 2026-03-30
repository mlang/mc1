set(PYBIND11_FINDPYTHON ON)
find_package(pybind11 CONFIG REQUIRED)
find_package(Boost REQUIRED CONFIG)

pybind11_add_module(_core MODULE compiler.cpp dag.cpp audio_device.cpp runtime.cpp python.cpp)
target_link_libraries(_core PRIVATE Boost::boost gccjit)
pybind11_add_module(_test MODULE compiler.cpp dag.cpp audio_device.cpp runtime.cpp test_python.cpp)
target_link_libraries(_test PRIVATE Boost::boost gccjit)
install(TARGETS _core _test DESTINATION ${SKBUILD_PROJECT_NAME})
