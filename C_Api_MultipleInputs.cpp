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

# define my_sizeof(type) ((char *)(&type+1)-(char*)(&type)) 

int main(int argc, char* argv[]) {
  //*************************************************************************
  // initialize  enviroment...one enviroment per process
  // enviroment maintains thread pools and other state info

  Ort::Env env;
  Ort::SessionOptions sessionOptions;
  sessionOptions.SetIntraOpNumThreads( 1 );
  sessionOptions.SetGraphOptimizationLevel( ORT_ENABLE_BASIC );
  const char* model_path = "/afs/cern.ch/work/d/dbakshig/public/Onnx/DNNCaloSim.onnx";
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
   // std::cout<<"Input "<<i<<" : "<<" type= "<<type<<std::endl;

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
// initialise
  // take values from file in order to compare to python (where rand inputs are generated)
  std::ifstream inputfile;
  inputfile.open("inputs.txt");

  // Inputs; noise
  float tmp[300];

Ort::TypeInfo Input_noise_info = session.GetInputTypeInfo(0);
auto Input_noise_tensor_info = Input_noise_info.GetTensorTypeAndShapeInfo();

// print input shapes/dims
std::vector<int64_t> Input_noise_dims;
Input_noise_dims = Input_noise_tensor_info.GetShape(); 
//std::cout<<"dims_data: "<<Input_noise_dims.data()<<" dims_size: "<<Input_noise_dims.size()<<std::endl;

for (int i=0; i < Input_noise_dims[1]; ++i)
  {
      //std::cout << i << "," ;
      float t;
      inputfile >> t;
      tmp[i] = t;
  }

//std::cout<<"tmp size: "<< sizeof(tmp)<<std::endl;
constexpr size_t values_length = sizeof(tmp) / sizeof(tmp[0]);
std::vector<int64_t> dims = {1,300};
std::cout<<"values_length: "<<values_length<<std::endl;

// create input tensor object from data values
  Ort::MemoryInfo info("Cpu", OrtDeviceAllocator, 0, OrtMemTypeDefault);
  Ort::Value Input_noise_tensor = Ort::Value::CreateTensor<float>(info, tmp, values_length, dims.data(), dims.size());
  assert(Input_noise_tensor.IsTensor());

 // Input_Energy
    float input_energy[] = {-1.18083284,-1.18083284}; // hardcoded from python for now
    std::vector<int64_t> dims_energy = {1,1};
    constexpr size_t length_energy = sizeof(input_energy) / sizeof(input_energy[0]);
    Ort::Value Input_energy_tensor = Ort::Value::CreateTensor<float>(info, input_energy, length_energy, dims_energy.data(), dims_energy.size());
    assert(Input_energy_tensor.IsTensor());    

 // Input_Phi_Config
    float Input_Phi_Config[] = {1., 0., 0., 0.};
    std::vector<int64_t> dims_phi = {1,4};
    constexpr size_t length_phi = sizeof(Input_Phi_Config) / sizeof(Input_Phi_Config[0]);
    Ort::Value Input_Phi_tensor = Ort::Value::CreateTensor<float>(info, Input_Phi_Config, length_phi, dims_phi.data(), dims_phi.size());
    assert(Input_Phi_tensor.IsTensor());

// Input_Eta_Config
    float Input_Eta_Config[] ={1.,0.};
    std::vector<int64_t> dims_eta = {1,2};
    constexpr size_t length_eta = sizeof(Input_Eta_Config) / sizeof(Input_Eta_Config[0]);
    Ort::Value Input_Eta_tensor = Ort::Value::CreateTensor<float>(info, Input_Eta_Config, length_eta, dims_eta.data(), dims_eta.size());
    assert(Input_Eta_tensor.IsTensor());    

// Input_RIPos
    float input_ripos[] = {0.24399453, -1.34985878};
    std::vector<int64_t> dims_ripos = {1,2};
    constexpr size_t length_ripos = sizeof(input_ripos) / sizeof(input_ripos[0]);
    Ort::Value Input_ripos_tensor = Ort::Value::CreateTensor<float>(info, input_ripos, length_ripos, dims_ripos.data(), dims_ripos.size());   
    assert(Input_ripos_tensor.IsTensor());

std::vector<Ort::Value> ort_inputs;

ort_inputs.push_back(std::move(Input_noise_tensor));
ort_inputs.push_back(std::move(Input_energy_tensor));
ort_inputs.push_back(std::move(Input_Phi_tensor));
ort_inputs.push_back(std::move(Input_Eta_tensor));
ort_inputs.push_back(std::move(Input_ripos_tensor));

std::vector<const char*> input_names = {"Input_noise", "Input_Energy", "Input_Phi_Config", "Input_Eta_Config", "Input_RIPos"};

const char* const output_names[] = {"dense_10"};

std::vector<Ort::Value> ort_outputs = session.Run(Ort::RunOptions{nullptr}, input_names.data(), ort_inputs.data(), ort_inputs.size(),output_node_names.data(), output_node_names.size());
assert(ort_outputs.size() == 1 && ort_outputs.front().IsTensor());
std::cout<<"ort_outputs_size: "<<ort_outputs.size()<<std::endl;


// Get pointer to output tensor float values
  float* floatarr = ort_outputs.front().GetTensorMutableData<float>();

for (int i = 0 ; i < 266; ++i){
    std::cout << floatarr[i] <<std::endl;
 }

return 0;
}
