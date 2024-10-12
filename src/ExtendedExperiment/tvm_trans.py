import onnx
import tvm
import tvm.relay as relay

def model_info(model_path):
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

    return input_names, output_names


def build_models(model_path, input_names, input_shapes, target, target_host):

    lib_file = model_path.replace('.onnx', '.so')
    json_file = model_path.replace('.onnx', '.json')
    params_file = model_path.replace('.onnx', '.params')
    print(input_names)

    onnx_model = onnx.load(model_path)
    shapes = {i: d for i, d in zip(input_names, input_shapes)}
    mod, params = relay.frontend.from_onnx(
        onnx_model, shape=shapes)
    print("relay.frontend.from_onnx is  sunccess")

    with relay.build_config(opt_level=3):
        graph, lib, params = relay.build(mod,
                                            target=target,
                                            #target_host=target_host,
                                            params=params)
    print("relay.build is  sunccess")
    lib.export_library(lib_file)
    with open(json_file, "w") as fo:
        fo.write(graph)
    with open(params_file, "wb") as fo:
        fo.write(tvm.runtime.save_param_dict(params))

    return graph, lib, params



if __name__ == '__main__':
     # 模型存放路径，编译完成后会在该onnx同级路径下生成.so .params .json文件
    model_path = 'weights/best.onnx'
     # onnx图片输入大小，可以在这个网站看https://netron.app/
    input_shape = [1, 3, 640, 640]

    print("model path: ", model_path)
    input_names, output_names = model_info(model_path)

    #target = f'xpu -libs=xdnn -split-device-funcs -device-type=xpu{os.environ.get("XPUSIM_DEVICE_MODEL", "KUNLUN2")[-1]}'
    target = tvm.target.cuda(0)
    target_host = 'llvm'
    check_ret = []
    perf_ret = []
  
    print("------------- ", input_shape, " --------------")
    graph, lib, params = build_models(model_path,  input_names, [input_shape], target, target_host)
