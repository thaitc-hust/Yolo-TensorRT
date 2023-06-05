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


def letterbox(img, new_shape=(416, 416), color=(114, 114, 114), auto=True, scaleFill=False, scaleup=True, auto_size=32):
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
            model_weights = '/models/Yolo-nas/weights/coco_yolonas.trt', 
            max_size = 416, 
            names = '/models/Yolo-nas/coco.names'):
        # self.names = [f"tattoo{i}" for i in range(80)]
        self.names = load_classes(names)
        self.colors = [[random.randint(0, 255) for _ in range(3)] for _ in range(len(self.names))]
        self.imgsz = (max_size, max_size)
        # Load model
        self.model = TrtModelNMS(model_weights, max_size)


    def detect(self, bgr_img):
        # input, (x_ratio, y_ratio) =  preprocess(bgr_img, (416, 416))
        # print(input.shape)   
        # Prediction
        ## Padded resize
        h, w, _ = bgr_img.shape
        # bgr_img, _, _ = letterbox(bgr_img)
        scale = min(self.imgsz[0]/w, self.imgsz[1]/h)
        inp = np.zeros((self.imgsz[1], self.imgsz[0], 3), dtype = np.float32)
        nh = int(scale * h)
        nw = int(scale * w)
        inp[: nh, :nw, :] = cv2.resize(cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB), (nw, nh))
        inp = inp.astype('float32') / 255.0  # 0 - 255 to 0.0 - 1.0
        inp = np.expand_dims(inp.transpose(2, 0, 1), 0)
        # print(inp.shape)  
        # print(x_ratio, y_ratio)
        ## Inference
        t1 = time.time()
        print(inp[0].transpose(0, 1, 2).shape)
        cv2.imwrite("test.jpg", inp[0].transpose(1, 2, 0) * 255)
        num_detection, nmsed_bboxes, nmsed_scores, nmsed_classes = self.model.run(inp)
        print(num_detection)
        print(nmsed_bboxes)
        print(nmsed_scores)
        print(nmsed_classes)
        t2 = time.time()
        print('Time cost: ', t2 - t1)
        ## Apply NMS
        num_detection = num_detection[0][0]
        nmsed_bboxes  = nmsed_bboxes[0]
        nmsed_scores  = nmsed_scores[0]
        nmsed_classes  = nmsed_classes[0]
        print(num_detection)
        # print(nmsed_classes)
        print('Detected {} object(s)'.format(num_detection))
        # print(nmsed_bboxes[:2])
        for bbx in nmsed_bboxes[:2]:
            print(bbx)
        # Rescale boxes from img_size to im0 size
        _, _, height, width = inp.shape
        h, w, _ = bgr_img.shape
        nmsed_bboxes[:, 0] /= scale
        nmsed_bboxes[:, 1] /= scale
        nmsed_bboxes[:, 2] /= scale
        nmsed_bboxes[:, 3] /= scale
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
    model = YOLOR(model_weights="/models/Yolo-nas/weights/coco_yolonas_v2.trt")
    img = cv2.imread('/models/hinhxam_test/614be9e3db868.jpg')
    model.detect(img)

