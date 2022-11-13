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


from models.models import Darknet


def load_classes(path):
    # Loads *.names file at 'path'
    with open(path, 'r') as f:
        names = f.read().split('\n')
    return list(filter(None, names))  # filter removes empty strings (such as last line)

def detect(bgr_img, model_cfg = 'cfg/yolor_csp_x.cfg', model_weights = 'yolor_csp_x_star.pt', imgsz = (896, 896), fp16 = True, device = 'cuda:1', names = 'data/coco.names'):
    # Initialize
    device = torch.device(device)

    # Get names and colors
    names = load_classes(names)
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in range(len(names))]

    # Load model
    model = Darknet(model_cfg, imgsz).cuda()
    model.load_state_dict(torch.load(model_weights, map_location=device)['model'])
    model.to(device).eval()
    if fp16:
        model.half()  # to FP16

    # Prediction
    ## Padded resize
    inp = letterbox(bgr_img, new_shape=imgsz, auto_size=64)[0]
    inp = inp[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
    inp = np.ascontiguousarray(inp)
    ## Convert to torch

    with torch.no_grad():
        inp = torch.from_numpy(inp).to(device)
        inp = inp.half() if fp16 else inp.float()  # uint8 to fp16/32
        inp /= 255.0  # 0 - 255 to 0.0 - 1.0

        if inp.ndimension() == 3:
            inp = inp.unsqueeze(0)
        ## Inference
        t1 = time.time()
        pred = model(inp, augment=False)[0]
        t2 = time.time()
        ## Apply NMS
        pred = non_max_suppression(pred, conf_thres=0.5, iou_thres=0.6)
        t3 = time.time()
        print('Inference: {}'.format(t2-t1))
        print('NMS: {}'.format(t3-t2))
    

        # Process detections
        visualize_img = bgr_img.copy()
        for i, det in enumerate(pred):  # detections per image
            gn = torch.tensor(bgr_img.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            if det is not None and len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(inp.shape[2:], det[:, :4], bgr_img.shape).round()
                for *xyxy, conf, cls in det:
                    label = '%s %.2f' % (names[int(cls)], conf)
                    plot_one_box(xyxy, visualize_img, label=label, color=colors[int(cls)], line_thickness=3)

    cv2.imwrite('result.jpg', visualize_img)


if __name__ == '__main__':
    img = cv2.imread('img.png')
    detect(img)
