import onnxruntime as ort
import numpy as np
import time

# Load ONNX model using ONNX Runtime
model_path = 'weights/best.onnx'
session = ort.InferenceSession(model_path)

# Get input and output names
input_name = session.get_inputs()[0].name
output_name = session.get_outputs()[0].name

# Create a random input tensor
input_tensor = np.random.rand(1, 3, 640, 640).astype(np.float32)

# Measure inference time over multiple runs
total_time = 0
for i in range(16):
    start = time.time()
    outputs = session.run([output_name], {input_name: input_tensor})
    end = time.time()
    total_time += end - start

# Calculate and print the average time per inference
print('ONNX Runtime - Total time: ', total_time / 16)
