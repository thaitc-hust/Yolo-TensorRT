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



def load_classes(path):
    # Loads *.names file at 'path'
    with open(path, 'r') as f:
        names = f.read().split('\n')
    return list(filter(None, names))  # filter removes empty strings (such as last line)
# def bbox2points(bbox):
#     """
#     From bounding box yolo format
#     to corner points cv2 rectangle
#     """
#     x, y, w, h = bbox
#     xmin = int(round(x - (w / 2)))
#     xmax = int(round(x + (w / 2)))
#     ymin = int(round(y - (h / 2)))
#     ymax = int(round(y + (h / 2)))
#     return xmin, ymin, xmax, ymax

class YOLOR(object):
    def __init__(self, 
            model_weights = './yolor_p6-nms.trt', 
            max_size = 640, 
            names = '/home/thaitran/hawkice/car-logo/yolor/data/car-logo.names'):
        self.names = load_classes(names)
        self.colors = [[random.randint(0, 255) for _ in range(3)] for _ in range(len(self.names))]
        self.imgsz = (max_size, max_size)
        # Load model
        self.model = TrtModelNMS(model_weights, max_size)

    def detect(self, bgr_img, img_name):   
        # Prediction
        ## Padded resize
        h, w, _ = bgr_img.shape
        scale = min(self.imgsz[0]/w, self.imgsz[1]/h)
        inp = np.zeros((self.imgsz[1], self.imgsz[0], 3), dtype = np.float32)
        nh = int(scale * h)
        nw = int(scale * w)
        inp[: nh, :nw, :] = cv2.resize(cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB), (nw, nh))
        inp = inp.astype('float32') / 255.0  # 0 - 255 to 0.0 - 1.0
        inp = np.expand_dims(inp.transpose(2, 0, 1), 0)
        print(inp.shape)        
        
        ## Inference
        t1 = time.time()
        num_detection, nmsed_bboxes, nmsed_scores, nmsed_classes = self.model.run(inp)
        t2 = time.time()
        # print('Time cost: ', t2 - t1)
        ## Apply NMS
        num_detection = num_detection[0][0]
        nmsed_bboxes  = nmsed_bboxes[0]
        nmsed_scores  = nmsed_scores[0]
        nmsed_classes  = nmsed_classes[0]
        print('Detected {} object(s)'.format(num_detection))
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
            print(nmsed_bboxes[ix])
            cv2.rectangle(visualize_img, (int(x1), int(y1)), (int(x2), int(y2)), self.colors[int(cls)], 2)
            cv2.putText(visualize_img, label, (int(x1), int(y1-10)), cv2.FONT_HERSHEY_SIMPLEX, 1, self.colors[int(cls)], 2, cv2.LINE_AA)
        cv2.imwrite(img_name, visualize_img)
        return visualize_img

if __name__ == '__main__':
    import glob
    model = YOLOR()
    # img = cv2.imread('car1.png')
    # model.detect(img, './result-test.jpg')
    path = '/home/thaitran/hawkice/car-logo/yolor-convert/test_images2/*'
    path_save = '/home/thaitran/hawkice/car-logo/yolor-convert/result2'
    list_img = glob.glob(path)
    for p_img in list_img:
        print(p_img)
        img = cv2.imread(p_img)
        print(img.shape)
        model.detect(img, os.path.join(path_save, p_img.split('/')[-1]))

