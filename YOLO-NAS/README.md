# **YOLONAS Torch2TRT-batchedNMS**

## Environment

I'm running with docker `nvcr.io/nvidia/tritonserver:22.12`

- Python 3.8
- Torch 1.13.1
- ONNX 1.14.0
- Tensorrt 8.5.1.7

## Convert Yolo-nas Pytorch to ONNX
- Open file ```torch2onnx.py``` and update attribute values to suit your model
- Run: 
```Shell
CUDA_VISIBLE_DEVICES=1 python torch2onnx.py --weights weights/<your_model_name>.pt --output weights/<your_output_model_name>.onnx
```
## Add NMS Batched to onnx model
- Open file ```add_nms_plugins.py``` and update attribute values to suit your model
- Run:
```Shell
python3 add_nms_plugins.py --model weights/<your_output_model_name>.onnx
```
## Convert ONNX model to TrT model
- Run:
```Shell
/usr/src/tensorrt/bin/trtexec --onnx=weights/<your_output_model_name>-nms.onnx \
                                --saveEngine=weights/<your_output_trt_model_name>.trt \
                                --explicitBatch \
                                --minShapes=input:1x3x416x416 \
                                --optShapes=input:1x3x896x896 \
                                --maxShapes=input:1x3x896x896 \
                                --verbose \
                                --device=1
```

## Inference
- Open file ```infer_trt.py``` and modify attribute values
- Run: 
```Shell
python3 infer_trt.py
```

# REFERENCE
1. https://github.com/Deci-AI/super-gradients