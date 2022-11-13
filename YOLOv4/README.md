# Convert Yolov4 Darknet to TensorRT add BatchedNMSDynamic

## Convert Yolov4 Darknet to ONNX
- If you want to convert model not Dynamic: setting batch_size > 0.
- Run:
``` Shell
python3 darknet2onnx.py --config config/yolov4.cfg --weightfile weights/yolov4_50000.weights --batch_size -1 --onnx_file_path save_convert/yolov4.onnx
```

## Add NMS Plugins 
- Open file ```add-nms.py``` and update attribute values to suit your model
- Then, run: 
```Shell
python3 add-nms.py --model save_convert/yolov4.onnx
```

## Convert ONNX to TensorRT
```Shell
/usr/src/tensorrt/bin/trtexec --onnx=save_convert/yolov4-nms.onnx \
                                --saveEngine=save_convert/yolov4-add-nms.trt \
                                --explicitBatch \
                                --minShapes=input:1x3x416x416 \
                                --optShapes=input:1x3x608x608 \
                                --maxShapes=input:1x3x608x608 \
                                --verbose \
                                --device=1
```

## Inference TrT Model add NMS:
- Open file ```demo_trt_nms.py``` and update attribute values: ```model_weights```, ```max_size```, ```names```
- Run:
```Shell
python3 demo_trt_nms.py
```

# REFERENCE
1. https://github.com/Tianxiaomo/pytorch-YOLOv4
2. https://github.com/NNDam/yolor/blob/main/object_detector_trt_nms.py
3. https://github.com/NVIDIA-AI-IOT/yolov4_deepstream
