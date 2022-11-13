/usr/src/tensorrt/bin/trtexec --onnx=NNDAM/yolor_p6.onnx.nms.onnx \
                                --saveEngine=NNDAM/yolor_p6-nms.trt \
                                --explicitBatch \
                                --minShapes=input:1x3x416x416 \
                                --optShapes=input:1x3x896x896 \
                                --maxShapes=input:1x3x896x896 \
                                --verbose \
                                --device=1
