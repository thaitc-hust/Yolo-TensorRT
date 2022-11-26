/usr/src/tensorrt/bin/trtexec --onnx=weights/yolov5-nms.onnx \
                              --saveEngine=weights/yolov5-nms-fp16.trt \
                              --explicitBatch \
                              --minShapes=input:1x3x416x416 \
                              --optShapes=input:1x3x448x448 \
                              --maxShapes=input:1x3x448x448 \
                              --verbose \
                              --fp16 \
                              --device=1