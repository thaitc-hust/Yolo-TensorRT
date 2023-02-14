# YOLOR CONVERT TO TENSORRT ADD BATCHEDNMS

I fixed **"IR version checking error"** in [NNDam's](https://github.com/NNDam/yolor) code

## Environment
- Python3.6
- Torch 1.8.1+cu102
- ONNX 1.9.0
- Tensorrt 7.2.3.4

## Convert Yolor pytorch to onnx add post processor
- Open file ```torch2onnx.py``` and update attribute values to suit your model
- Run:
```Shell
CUDA_VISIBLE_DEVICES=1 python torch2onnx.py --weights yolor_p6.pt --cfg cfg/config.cfg --output yolor_p6.onnx
```

## Add NMS Batched to onnx model
- Open file ```onnx_add_nms.py``` and update attribute values to suit your model
- Run:
```Shell
python3 onnx_add_nms.py --model yolor_p6.onnx
```

## Convert ONNX model to TrT model
- Run:
```Shell
/usr/src/tensorrt/bin/trtexec --onnx=yolor_p6-nms.onnx \
                                --saveEngine=yolor_p6-nms.trt \
                                --explicitBatch \
                                --minShapes=input:1x3x416x416 \
                                --optShapes=input:1x3x896x896 \
                                --maxShapes=input:1x3x896x896 \
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
1. https://github.com/WongKinYiu/yolor
2. https://github.com/NNDam/yolor
