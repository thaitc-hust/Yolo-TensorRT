import argparse
import os
import platform
import shutil
import time
from pathlib import Path

import cv2
import torch
import numpy as np
from numpy import random

from utils.datasets import letterbox
from utils.general import non_max_suppression, scale_coords
from utils.plots import plot_one_box
import onnxruntime as rt

# from models.models import Darknet

def load_classes(path):
    # Loads *.names file at 'path'
    with open(path, 'r') as f:
        names = f.read().split('\n')
    return list(filter(None, names))  # filter removes empty strings (such as last line)

class YOLOR(object):
    def __init__(self, 
            model_weights = 'yolor_p6_x_star.onnx', 
            imgsz = (448, 448), 
            names = '/home/thaitran/hawkice/car-logo/yolor/data/car-logo.names'):
        self.names = load_classes(names)
        self.colors = [[random.randint(0, 255) for _ in range(3)] for _ in range(len(self.names))]
        self.imgsz = imgsz
        # Load model
        self.model = rt.InferenceSession(model_weights)


    def detect(self, bgr_img, threshold = 0.4):   
        # Prediction
        ## Padded resize
        inp = letterbox(bgr_img, new_shape=self.imgsz, auto_size=64)[0]
        # print(inp.shape)
        # inp = cv2.resize(bgr_img, self.imgsz)
        inp = inp[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB
        inp = inp.astype('float32') / 255.0  # 0 - 255 to 0.0 - 1.0
        inp = np.expand_dims(inp, 0)
        # inp = np.transpose(inp, (0, 3, 1, 2))
        print(inp.shape)
        ## Convert to torch
        
        ## Inference
        t1 = time.time()
        ort_inputs = {self.model.get_inputs()[0].name: inp}
        pred = self.model.run(None, ort_inputs)[0]
        t2 = time.time()
        ## Apply NMS
        with torch.no_grad():
            pred = non_max_suppression(torch.tensor(pred), conf_thres=threshold, iou_thres=0.6)
        t3 = time.time()
        print('Inference: {}'.format(t2-t1))
        print('NMS: {}'.format(t3-t2))
    
        # Process detections
        visualize_img = bgr_img.copy()
        det = pred[0]  # detections per image
        if det is not None and len(det):
            # Rescale boxes from img_size to im0 size
            _, _, height, width = inp.shape
            h, w, _ = bgr_img.shape
            det[:, 0] *= w/width
            det[:, 1] *= h/height
            det[:, 2] *= w/width
            det[:, 3] *= h/height
            for x1, y1, x2, y2, conf, cls in det:       # x1, y1, x2, y2 in pixel format
                label = '%s %.2f' % (self.names[int(cls)], conf)
                plot_one_box((x1, y1, x2, y2), visualize_img, label=label, color=self.colors[int(cls)], line_thickness=3)

        cv2.imwrite('result.jpg', visualize_img)
        return visualize_img

if __name__ == '__main__':
    model = YOLOR()
    img = cv2.imread('/home/thaitran/hawkice/car-logo/yolor/car2.png')
    model.detect(img)
