from openvino.runtime import Core
import numpy as np
import time

# Initialize OpenVINO runtime
ie = Core()

# Load the model
model_path = 'weights/best_openvino_model/best.xml'
compiled_model = ie.compile_model(model=model_path, device_name="CPU")  # Use 'CPU' if GPU is not available

# Create input tensor
input_tensor = np.random.rand(1, 3, 640, 640).astype(np.float32)

# Get the input and output layers
input_layer = compiled_model.input(0)
output_layer = compiled_model.output(0)

# Measure inference time over multiple runs
total_time = 0
for i in range(16):
    start = time.time()
    result = compiled_model([input_tensor])
    end = time.time()
    total_time += end - start

# Calculate and print the average time per inference
print('OpenVINO - Total time: ', total_time / 16)
