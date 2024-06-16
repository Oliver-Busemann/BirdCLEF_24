import onnxruntime.quantization
import onnxruntime.quantization.preprocess
import torch
import timm
import onnx
import onnxruntime
from onnxruntime.quantization import shape_inference, quantize_dynamic, QuantType
import os


model_name = 'efficientnet_b0'

filename = 'Full_Data_TrainFull_EPOCHS_10.pth'
new_filename = f'{filename.split(".")[0]}.onnx'
quant_filename = f'{filename.split(".")[0]}_quantized.onnx'
folder_weights = '/home/olli/Projects/Kaggle/BirdCLEF/Model_Weights'

path_weights = os.path.join(folder_weights, filename)

path_onnx_weights = os.path.join(folder_weights, new_filename)

path_onnx_quant_weights = os.path.join(folder_weights, quant_filename)

model = timm.create_model(model_name=model_name, pretrained=False, num_classes=182, in_chans=1)

model.load_state_dict(torch.load(path_weights))

model.eval()

dummy_input = torch.randn((1, 1, 256, 256))

torch.onnx.export(
    model,
    dummy_input,
    path_onnx_weights,
    export_params=True,
    opset_version=16,
    do_constant_folding=True,
    input_names=['input'],
    output_names=['output'],
    dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}
    )

# load the onnx model to perform three optimization steps before quantiaztion
onnx_model = onnx.load(path_onnx_weights)

# symbolic shape inference & model optimization (?)
onnx_model = shape_inference.quant_pre_process(
    onnx_model,
    path_onnx_weights,
    use_symbolic_shape_inference=True
    )

onnx_model = onnx.load(path_onnx_weights)
onnx.shape_inference.infer_shapes(onnx_model)
onnx.save(onnx_model, path_onnx_weights)

quantize_dynamic(path_onnx_weights, path_onnx_quant_weights, weight_type=QuantType.QUInt8)