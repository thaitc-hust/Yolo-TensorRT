# **YOLOV5 Torch2TRT-batchedNMS**

## Environment
- Python3.6
- Torch 1.8.1+cu102
- ONNX 1.9.0
- Tensorrt 7.2.3.4

## Convert Yolov5 Pytorch to ONNX
- Open file ```torch2onnx.py``` and update attribute values to suit your model
- Run: 
```Shell
CUDA_VISIBLE_DEVICES=1 python torch2onnx.py --weights weights/<your_model_name>.pt --output weights/<your_output_model_name>.onnx --max_size 640
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
                                --minShapes=input:1x3x640x640 \
                                --optShapes=input:1x3x640x640 \
                                --maxShapes=input:4x3x640x640 \
                                --verbose \
                                --device=1
```

## Inference
- Open file ```object_detector_trt_nms.py``` and modify attribute values
- Run: 
```Shell
python3 object_detector_trt_nms.py
```

# REFERENCE
1. https://github.com/ultralytics/yolov5
2. https://github.com/NNDam/yolor
