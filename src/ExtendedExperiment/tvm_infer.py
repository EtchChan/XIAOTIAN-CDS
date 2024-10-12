import tvm
from tvm import relay
import onnx
import time
import cv2
import numpy as np

# onnx路径，需要在同级路径下保存.so .json .params文件，即编译过的onnx文件
model_path = 'weights/best.onnx'
# 任意图片，用于测试模型是否编译成功


lib_file = model_path.replace('.onnx', '.so')
json_file = model_path.replace('.onnx', '.json')
params_file = model_path.replace('.onnx', '.params')
lib = tvm.runtime.load_module(lib_file)
graph = open(json_file).read()
file_data = bytearray(open(params_file, "rb").read())
params = tvm.runtime.load_param_dict(file_data)

target = 'cuda'
#target = tvm.target.cuda()
ctx = tvm.device(target, 0) #0对应第0号显卡

module = tvm.contrib.graph_executor.create(graph, lib, ctx)
module.load_params(relay.save_param_dict(params))


model = onnx.load(model_path)
input_names = []
output_names = []
for output in model.graph.output:
    output_names.append(output.name)

weight_name = []
for weight in model.graph.initializer:
    weight_name.append(weight.name)

for i, input in enumerate(model.graph.input):
    if input.name in weight_name:
        continue
    input_names.append(input.name)

input_tensor = np.random.rand(1, 3, 640, 640).astype(np.float32)
input_tensor = np.ascontiguousarray(input_tensor)


total_time = 0
print('===========')
for i in range(16):
    module.set_input(input_names[0], input_tensor)
    start = time.time()
    module.run()
    end = time.time()
    output = []
    for j in range(module.get_num_outputs()):
        output.append(module.get_output(j).asnumpy())
    total_time += end - start

print('total time: ', total_time / 16)

