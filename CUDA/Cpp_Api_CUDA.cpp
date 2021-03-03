// Copyright(c) Microsoft Corporation.All rights reserved.
// Licensed under the MIT License.
//

#include <assert.h>
#include <onnxruntime_cxx_api.h>
#include <stdlib.h>
#include <stdio.h>
#include <vector>
#include <iostream> 
#include <fstream>
#include <arpa/inet.h>
#include "cuda_provider_factory.h"

// for reading MNIST images
std::vector<std::vector<float>> read_mnist_pixel(const std::string &full_path) //function to load test images
{
  std::vector<std::vector<float>> input_tensor_values;
  input_tensor_values.resize(10000, std::vector<float>(28*28*1));  
  std::ifstream file (full_path.c_str(), std::ios::binary);
  int magic_number=0;
  int number_of_images=0;
  int n_rows=0;
  int n_cols=0;
  file.read((char*)&magic_number,sizeof(magic_number));
  magic_number= ntohl(magic_number);
  file.read((char*)&number_of_images,sizeof(number_of_images));
  number_of_images= ntohl(number_of_images);
  file.read((char*)&n_rows,sizeof(n_rows));
  n_rows= ntohl(n_rows);
  file.read((char*)&n_cols,sizeof(n_cols));
  n_cols= ntohl(n_cols);
  for(int i=0;i<number_of_images;++i)
  {
        for(int r=0;r<n_rows;++r)
        {
            for(int c=0;c<n_cols;++c)
            {
                unsigned char temp=0;
                file.read((char*)&temp,sizeof(temp));
                input_tensor_values[i][r*n_cols+c]= float(temp)/255;
            }
        }
  }
  return input_tensor_values;
}

//********************************************************************************
// for reading MNIST labels
std::vector<int> read_mnist_label(const std::string &full_path) //function to load test labels
{
  std::vector<int> output_tensor_values(1*10000);
    std::ifstream file (full_path.c_str(), std::ios::binary);
    int magic_number=0;
    int number_of_labels=0;
    file.read((char*)&magic_number,sizeof(magic_number));
    magic_number= ntohl(magic_number);
    file.read((char*)&number_of_labels,sizeof(number_of_labels));
    number_of_labels= ntohl(number_of_labels);
    for(int i=0;i<number_of_labels;++i)
    {
                unsigned char temp=0;
                file.read((char*)&temp,sizeof(temp));
                output_tensor_values[i]= int(temp);
  }
  return output_tensor_values;
}


int main(int argc, char* argv[]) {
  //*************************************************************************
  // initialize  enviroment...one enviroment per process
  // enviroment maintains thread pools and other state info

  Ort::Env env;
  Ort::SessionOptions sessionOptions;
  sessionOptions.SetIntraOpNumThreads( 1 );
  sessionOptions.SetGraphOptimizationLevel( ORT_ENABLE_BASIC );

  //Ort::CUDAProviderOptions cudaoptions;
  //cudaoptions.device_id=0;
  
  OrtSessionOptionsAppendExecutionProvider_CUDA(sessionOptions,0);
  const char* model_path = "/home/mfenton/ONNX/GPUComp/MNIST_testModel.onnx";
  Ort::AllocatorWithDefaultOptions allocator;  
  Ort::Session session(env, model_path, sessionOptions);
  std::vector<int64_t> input_node_dims;
  size_t num_input_nodes = session.GetInputCount();
  std::vector<const char*> input_node_names(num_input_nodes);
  for( std::size_t i = 0; i < num_input_nodes; i++ ) {
     // print input node names
     char* input_name = session.GetInputName(i, allocator);
     std::cout<<"Input "<<i<<" : "<<" name= "<<input_name<<std::endl;
     input_node_names[i] = input_name;
 // print input node types
    Ort::TypeInfo type_info = session.GetInputTypeInfo(i);
    auto tensor_info = type_info.GetTensorTypeAndShapeInfo();
    ONNXTensorElementDataType type = tensor_info.GetElementType();
    std::cout<<"Input "<<i<<" : "<<" type= "<<type<<std::endl;

    // print input shapes/dims
    input_node_dims = tensor_info.GetShape();
    std::cout<<"Input "<<i<<" : num_dims= "<<input_node_dims.size()<<std::endl;
    for (int j = 0; j < input_node_dims.size(); j++){
      if(input_node_dims[j]<0)
       input_node_dims[j] =1;
      std::cout<<"Input"<<i<<" : dim "<<j<<"= "<<input_node_dims[j]<<std::endl;
 }  
}
//output nodes
  std::vector<int64_t> output_node_dims;
  size_t num_output_nodes = session.GetOutputCount();
  std::vector<const char*> output_node_names(num_output_nodes);

  for( std::size_t i = 0; i < num_output_nodes; i++ ) {
     // print output node names
     char* output_name = session.GetOutputName(i, allocator);
     std::cout<<"Output "<<i<<" : "<<" name= "<<output_name<<std::endl;
     output_node_names[i] = output_name;
 // print input node types
    Ort::TypeInfo type_info = session.GetOutputTypeInfo(i);
    auto tensor_info = type_info.GetTensorTypeAndShapeInfo();
    ONNXTensorElementDataType type = tensor_info.GetElementType();
    std::cout<<"Output "<<i<<" : "<<" type= "<<type<<std::endl;

    // print input shapes/dims
    output_node_dims = tensor_info.GetShape();
    std::cout<<"Output "<<i<<" : num_dims= "<<output_node_dims.size()<<std::endl;
    for (int j = 0; j < output_node_dims.size(); j++){
      if(output_node_dims[j]<0)
       output_node_dims[j] =1;
      std::cout<<"Output"<<i<<" : dim "<<j<<"= "<<output_node_dims[j]<<std::endl;
 }  
}
//*************************************************************************
  // Score the model using sample data, and inspect values
  //loading input data
  
  std::vector<std::vector<float>> input_tensor_values_ = read_mnist_pixel("/afs/cern.ch/work/d/dbakshig/public/Onnx/onnxruntime/t10k-images-idx3-ubyte");
  std::vector<int> output_tensor_values_ = read_mnist_label("/afs/cern.ch/work/d/dbakshig/public/Onnx/onnxruntime/t10k-labels-idx1-ubyte");
  
  //preparing container to hold input data
  
  size_t input_tensor_size = 1*28*28;
  std::vector<float> input_tensor_values(input_tensor_size);
  input_tensor_values = input_tensor_values_[0];
  
 //preparing container to hold output data
 int output_tensor_values = output_tensor_values_[0]; 

 // create input tensor object from data values
  auto memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
  Ort::Value input_tensor = Ort::Value::CreateTensor<float>(memory_info, input_tensor_values.data(), input_tensor_size, input_node_dims.data(), 3);
  assert(input_tensor.IsTensor());

  // score model & input tensor, get back output tensor
  auto output_tensors = session.Run(Ort::RunOptions{nullptr}, input_node_names.data(), &input_tensor, 1, output_node_names.data(), 1);
  assert(output_tensors.size() == 1 && output_tensors.front().IsTensor());
  
// Get pointer to output tensor float values
  float* floatarr = output_tensors.front().GetTensorMutableData<float>();
 // assert(abs(floatarr[0] - 0.000045) < 1e-6);

for (int i = 0; i < 10; i++)
   std::cout<<"Score for class "<<i<<" = "<<floatarr[i]<<std::endl;
  // show  true label for the test input
std::cout<<"Label for the input test data  = "<<output_tensor_values<<std::endl;


return 0;
}
