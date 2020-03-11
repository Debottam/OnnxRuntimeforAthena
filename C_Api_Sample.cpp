// Copyright(c) Microsoft Corporation.All rights reserved.
// Licensed under the MIT License.
//

#include <assert.h>
#include <onnxruntime_c_api.h>
#include <stdlib.h>
#include <stdio.h>
#include <vector>

const OrtApi* g_ort = OrtGetApiBase()->GetApi(ORT_API_VERSION);

//*****************************************************************************
// helper function to check for status
void CheckStatus(OrtStatus* status)
{
    if (status != NULL) {
      const char* msg = g_ort->GetErrorMessage(status);
      fprintf(stderr, "%s\n", msg);
      g_ort->ReleaseStatus(status);
      exit(1);
    }
}

int main(int argc, char* argv[]) {
  //*************************************************************************
  // initialize  enviroment...one enviroment per process
  // enviroment maintains thread pools and other state info
  OrtEnv* env;
  CheckStatus(g_ort->CreateEnv(ORT_LOGGING_LEVEL_WARNING, "test", &env));

  // initialize session options if needed
  OrtSessionOptions* session_options;
  CheckStatus(g_ort->CreateSessionOptions(&session_options));
  g_ort->SetIntraOpNumThreads(session_options, 1);

  // Sets graph optimization level
  g_ort->SetSessionGraphOptimizationLevel(session_options, ORT_ENABLE_BASIC);

  //*************************************************************************
  // create session and load model into memory
  OrtSession* session;
  const char* model_path = "/afs/cern.ch/work/d/dbakshig/public/OnnxRuntimeforAthena/saved_mnist/saved_model.onnx";

  printf("Using Onnxruntime C API\n");
  CheckStatus(g_ort->CreateSession(env, model_path, session_options, &session));

  //*************************************************************************
  // print model input layer (node names, types, shape etc.)
  size_t num_input_nodes;
  size_t num_output_nodes;
  OrtStatus* status;
  OrtStatus* status_Out;
  OrtAllocator* allocator;
  CheckStatus(g_ort->GetAllocatorWithDefaultOptions(&allocator));

  // print number of model input nodes
  status = g_ort->SessionGetInputCount(session, &num_input_nodes);
  std::vector<const char*> input_node_names(num_input_nodes);
  std::vector<int64_t> input_node_dims;  // simplify... this model has only 1 input node {1, 3, 224, 224}.
                                         // Otherwise need vector<vector<>>

  printf("Number of inputs = %zu\n", num_input_nodes);

// print number of model output nodes
  status_Out = g_ort->SessionGetOutputCount(session, &num_output_nodes);
  std::vector<const char*> output_node_names(num_output_nodes);
  std::vector<int64_t> output_node_dims;  // simplify... this model has only 1 input node {1, 3, 224, 224}.
                                         // Otherwise need vector<vector<>>

  printf("Number of outputs = %zu\n", num_output_nodes);


  // iterate over all input nodes
  for (size_t i = 0; i < num_input_nodes; i++) {
    // print input node names
    char* input_name;
    status = g_ort->SessionGetInputName(session, i, allocator, &input_name);
    printf("Input %zu : name=%s\n", i, input_name);
    input_node_names[i] = input_name;

    // print input node types
    OrtTypeInfo* typeinfo;
    status = g_ort->SessionGetInputTypeInfo(session, i, &typeinfo);
    const OrtTensorTypeAndShapeInfo* tensor_info;
	CheckStatus(g_ort->CastTypeInfoToTensorInfo(typeinfo, &tensor_info));
    ONNXTensorElementDataType type;
	CheckStatus(g_ort->GetTensorElementType(tensor_info, &type));
    printf("Input %zu : type=%d\n", i, type);

    // print input shapes/dims
    size_t num_dims;
	CheckStatus(g_ort->GetDimensionsCount(tensor_info, &num_dims));
    printf("Input %zu : num_dims=%zu\n", i, num_dims);
    input_node_dims.resize(num_dims);
	g_ort->GetDimensions(tensor_info, (int64_t*)input_node_dims.data(), num_dims);
    for (size_t j = 0; j < num_dims; j++)
      printf("Input %zu : dim %zu=%jd\n", i, j, input_node_dims[j]);

	g_ort->ReleaseTypeInfo(typeinfo);
  }

// iterate over all output nodes
  for (size_t i = 0; i < num_output_nodes; i++) {
    // print output node names
    char* output_name;
    status_Out = g_ort->SessionGetOutputName(session, i, allocator, &output_name);
    printf("Output %zu : name=%s\n", i, output_name);
    output_node_names[i] = output_name;

    // print output node types
    OrtTypeInfo* typeinfo;
    status_Out = g_ort->SessionGetOutputTypeInfo(session, i, &typeinfo);
    const OrtTensorTypeAndShapeInfo* tensor_info;
      CheckStatus(g_ort->CastTypeInfoToTensorInfo(typeinfo, &tensor_info));
    ONNXTensorElementDataType type;
      CheckStatus(g_ort->GetTensorElementType(tensor_info, &type));
    printf("Output %zu : type=%d\n", i, type);

    // print output shapes/dims
    size_t num_dims;
      CheckStatus(g_ort->GetDimensionsCount(tensor_info, &num_dims));
    printf("Output %zu : num_dims=%zu\n", i, num_dims);
    output_node_dims.resize(num_dims);
      g_ort->GetDimensions(tensor_info, (int64_t*)output_node_dims.data(), num_dims);
    for (size_t j = 0; j < num_dims; j++){
      if(output_node_dims[j]<0)
       output_node_dims[j] =1;
      printf("Output %zu : dim %zu=%jd\n", i, j, output_node_dims[j]);
   }
    g_ort->ReleaseTypeInfo(typeinfo);
  }

  g_ort->ReleaseSession(session);
  g_ort->ReleaseSessionOptions(session_options);
  g_ort->ReleaseEnv(env);
  printf("Done!\n");
  return 0;
}
