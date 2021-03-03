cmake_minimum_required(VERSION 3.14.3)
project(OnnxRuntimeforAthena)
set(CMAKE_BUILD_TYPE Debug)
set(CMAKE_CXX_FLAGS "-std=c++14")
set(PROJECT_LINK_LIBS /home/mfenton/ONNX/GPUComp/onnxruntime/build/libonnxruntime.so  )
link_directories(/home/mfenton/ONNX/GPUComp/OnnxRuntimeforAthena/build/)
include_directories(/home/mfenton/ONNX/GPUComp/OnnxRuntimeforAthena/../onnxruntime/include/onnxruntime/core/session/ /home/mfenton/ONNX/GPUComp/OnnxRuntimeforAthena/../onnxruntime/include/onnxruntime/core/providers/cuda/ )
add_executable(Cpp_Api /home/mfenton/ONNX/GPUComp/OnnxRuntimeforAthena/Cpp_Api.cpp)
target_link_libraries(Cpp_Api ${PROJECT_LINK_LIBS})


