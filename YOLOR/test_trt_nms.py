import argparse
import os
import platform
import shutil
import time
from pathlib import Path

import cv2
import numpy as np
from numpy import random



from exec_backends.trt_loader import TrtModelNMS
# from models.models import Darknet

def letterbox(img, new_shape=(640, 640), color=(114, 114, 114), auto=True, scaleFill=False, scaleup=True, auto_size=32):
    # Resize image to a 32-pixel-multiple rectangle https://github.com/ultralytics/yolov3/issues/232
    shape = img.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:  # only scale down, do not scale up (for better test mAP)
        r = min(r, 1.0)

    # Compute padding
    ratio = r, r  # width, height ratios
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
    if auto:  # minimum rectangle
        dw, dh = np.mod(dw, auto_size), np.mod(dh, auto_size)  # wh padding
    elif scaleFill:  # stretch
        dw, dh = 0.0, 0.0
        new_unpad = (new_shape[1], new_shape[0])
        ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]  # width, height ratios

    dw /= 2  # divide padding into 2 sides
    dh /= 2

    if shape[::-1] != new_unpad:  # resize
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
    return img, ratio, (dw, dh)

def load_classes(path):
    # Loads *.names file at 'path'
    with open(path, 'r') as f:
        names = f.read().split('\n')
    return list(filter(None, names))  # filter removes empty strings (such as last line)

class YOLOR(object):
    def __init__(self, 
            model_weights = 'yolor_p6-nms.trt', 
            max_size = 448, 
            names = 'data/coco.names'):
        self.names = load_classes(names)
        self.colors = [[random.randint(0, 255) for _ in range(3)] for _ in range(len(self.names))]
        self.imgsz = (max_size, max_size)
        # Load model
        self.model = TrtModelNMS(model_weights, max_size)


    def detect(self, bgr_img):   
        # Prediction
        ## Padded resize
        inp = letterbox(bgr_img, new_shape=self.imgsz, auto_size=64)[0]
        inp = inp[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB
        inp = inp.astype('float32') / 255.0  # 0 - 255 to 0.0 - 1.0
        inp = np.expand_dims(inp, 0)
        print(inp.shape)        
        
        ## Inference
        t1 = time.time()
        num_detection, nmsed_bboxes, nmsed_scores, nmsed_classes = self.model.run(inp)
        print(num_detection)
        print(nmsed_bboxes)
        print(nmsed_scores)
        t2 = time.time()
        print('Time cost: ', t2 - t1)
        ## Apply NMS
        num_detection = num_detection[0][0]
        nmsed_bboxes  = nmsed_bboxes[0]
        nmsed_scores  = nmsed_scores[0]
        nmsed_classes  = nmsed_classes[0]
        print('Detected {} object(s)'.format(num_detection))
        # Rescale boxes from img_size to im0 size
        _, _, height, width = inp.shape
        h, w, _ = bgr_img.shape
        nmsed_bboxes[:, 0] *= w/width
        nmsed_bboxes[:, 1] *= h/height
        nmsed_bboxes[:, 2] *= w/width
        nmsed_bboxes[:, 3] *= h/height
        visualize_img = bgr_img.copy()
        for ix in range(num_detection):       # x1, y1, x2, y2 in pixel format
            cls = int(nmsed_classes[ix])
            label = '%s %.2f' % (self.names[cls], nmsed_scores[ix])
            x1, y1, x2, y2 = nmsed_bboxes[ix]

            cv2.rectangle(visualize_img, (int(x1), int(y1)), (int(x2), int(y2)), self.colors[int(cls)], 2)
            cv2.putText(visualize_img, label, (int(x1), int(y1-10)), cv2.FONT_HERSHEY_SIMPLEX, 1, self.colors[int(cls)], 2, cv2.LINE_AA)

        cv2.imwrite('result.jpg', visualize_img)
        return visualize_img

if __name__ == '__main__':
    model = YOLOR()
    img = cv2.imread('car2.png')
    model.detect(img)