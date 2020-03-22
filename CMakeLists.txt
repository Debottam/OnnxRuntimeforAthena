cmake_minimum_required(VERSION 3.14.3)
project(OnnxRuntimeforAthena)
set(CMAKE_BUILD_TYPE Debug)
set(CMAKE_CXX_FLAGS "-std=c++14")
set(PROJECT_LINK_LIBS libonnxruntime.so)
link_directories(/afs/cern.ch/work/d/dbakshig/public/onnxExternal/build/)
include_directories(/afs/cern.ch/work/d/dbakshig/public/Onnx/onnxruntime/include/onnxruntime/core/session/)
add_executable(C_Api_Sample /afs/cern.ch/work/d/dbakshig/public/OnnxRuntimeforAthena/C_Api_Sample4.cpp)
target_link_libraries(C_Api_Sample ${PROJECT_LINK_LIBS})


