# Approaches 

### Transformer

### GNN

### clustering

### Kalman filter

# Extended Experiment

## expriment setup

### VisDrone dataset
**hardware :**

NVIDIA A1O

**requirement:** 

ultralytics
### TVM implement
**hardware :**

NVIDIA Jetson Orin

**requirement:** 

onnxruntime 

onnxslim

openvino

tvm

**TVM setup**

>Install clang+llvm (version >= 15.0)

>Install tvm from Source according to https://tvm.apache.org/docs/install/from_source.html
>
>>echo "set(USE_LLVM \"llvm-config --ignore-libllvm --link-static\")" >> config.cmake
>
>>echo "set(USE_CUDA   ON)" >> config.cmake

### expriment result

>> see expriment folder 
![implement_exp](./ExtendedExperiment/res_vis/implement_exp.jpg "this is implement_exp image")
![tracking_vis](./ExtendedExperiment/res_vis/tracking_vis.png "this is tracking_vis image")
![yolov11_res](./ExtendedExperiment/res_vis/yolov11_res.png "this is yolov11 training res image")
