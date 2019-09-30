# OnnxRuntimeforAthena

**Why onnx?**
* OnnxRuntime (https://github.com/Microsoft/onnxruntime) can run .onnx inference using C++ api.
* model generated in any platform (e.g. TensorFlow, PyTorch, Café, etc.) in proto-buffer frormat (.pb) can be converted to .onnx models.

**Packages we will need**
* TensorFlow
* tf2onnx
* onnxruntime

**Building onnxruntime from source using cmake**
Dependencies
* protobuf 3.6.1 and higher (tensorflow comes with a built in)
* Python 3.5 and higher
* gcc>=5.0

With above dependencies I am able to use onnxruntime's c/c++ Api to make prediction with .onnx inference. In this page I have demonstrated successful use of onnxruntime's c/c++ api to use .onnx to predict MNIST's handwritten dataset. Now it would be great to know if we can get these dependcies available in Athena.

This section includes
* **tfToOnnx:**
 I have trained a tf.keras model to predict handwritten digits using MNIST dataset for handwritten digits and coverted that model to .onnx format
* **saved_model.onnx:**
 onnx model generated from previous step
* **C_Api_Sample4.cpp:**
 c/c++ API of onnx_runtime to use onnx model for scoring
 
 The **saved_model.onnx:** model has following dimention
```
Input:
input name flatten_1_input:0
input shape [None, 28, 28]
input type tensor(float)

Output:
output_name dense_3/Softmax:0
output shape [None, 10]
output type tensor(float)
```
The same **saved_model.onnx:** has been loaded to **C_Api_Sample4.cpp:** using onnx_runtime's c/c++ API and I get following 
```
Using Onnxruntime C API
Number of inputs = 1
Number of outputs = 1
Input 0 : name=flatten_input:0
Input 0 : type=1
Input 0 : num_dims=3
Input 0 : dim 0=1
Input 0 : dim 1=28
Input 0 : dim 2=28
Output 0 : name=dense_1/Softmax:0
Output 0 : type=1
Output 0 : num_dims=2
Output 0 : dim 0=1
Output 0 : dim 1=10
Score for class [0] =  0.000000
Score for class [1] =  0.000000
Score for class [2] =  0.000000
Score for class [3] =  0.000000
Score for class **[4] =  0.996822**
Score for class [5] =  0.000003
Score for class [6] =  0.000000
Score for class [7] =  0.000277
Score for class [8] =  0.001590
Score for class [9] =  0.001308
Label for the input test data  =  **4**
Done!
```
