import torch
import torchvision
from torchvision import models, datasets, transforms
import torch.nn as nn
import os 
import cv2 
import argparse 
import onnx 
from onnx import shape_inference 
from models.models import Darknet

class YOLORAddNMS(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.model.eval()

    def forward(self, input):
        """ 
            Split output [n_batch, n_bboxes, 85] to 3 output: bboxes, scores, classes
        """ 
        # x, y, w, h -> x1, y1, x2, y2
        output = self.model(input)
        print('Output: ', len(output))
        for x in output:
            if type(x).__name__ == 'tuple':
                print([y.shape for y in x])
            else:
                print('single ', x.shape)
        output = output[0]
        bboxes_x = output[..., 0:1]
        bboxes_y = output[..., 1:2]
        bboxes_w = output[..., 2:3]
        bboxes_h = output[..., 3:4]
        bboxes_x1 = bboxes_x - bboxes_w/2
        bboxes_y1 = bboxes_y - bboxes_h/2
        bboxes_x2 = bboxes_x + bboxes_w/2
        bboxes_y2 = bboxes_y + bboxes_h/2
        bboxes = torch.cat([bboxes_x1, bboxes_y1, bboxes_x2, bboxes_y2], dim = -1)
        bboxes = bboxes.unsqueeze(2) # [n_batch, n_bboxes, 4] -> [n_batch, n_bboxes, 1, 4]
        obj_conf = output[..., 4:5]
        cls_conf = output[..., 5:]
        scores   = obj_conf * cls_conf # conf = obj_conf * cls_conf
        # print(scores)
        # print(bboxes)
        return bboxes, scores

if __name__ =="__main__":
    parse = argparse.ArgumentParser()
    parse.add_argument('--weights', type=str, default="yolor_p6.pt", help='weights path')
    parse.add_argument('--cfg', type=str, default='cfg/yolor_csp_x.cfg', help='config path')
    parse.add_argument('--output', type=str, default='yolor_add_post_processing.onnx', help='model ONNX path output')
    parse.add_argument('--max_size', type=int, default=448, help='max size of input image')
    opt= parse.parse_args()

    model_cfg = opt.cfg
    model_weights = opt.weights
    output_model_path = opt.output
    max_size = opt.max_size

    device = torch.device('cuda')

    # Load origin model 
    model = Darknet(model_cfg, (max_size, max_size)).cuda()
    model.load_state_dict(torch.load(model_weights, map_location= device)['model'])
    # model.to(device).eval()

    model = YOLORAddNMS(model)
    model.to(device).eval()

    img = torch.zeros((7, 3, max_size, max_size), device=device)  # init img

    torch.onnx.export(model,               # model being run
                      img,                         # model input (or a tuple for multiple inputs)
                      output_model_path,   # where to save the model (can be a file or file-like object)
                      export_params=True,        # store the trained parameter weights inside the model file
                      opset_version=11,          # the ONNX version to export the model to
                      do_constant_folding=True,  # whether to execute constant folding for optimization
                      input_names = ['input'],   # the model's input names
                      output_names = ['bboxes', 'scores'], # the model's output names
                      dynamic_axes={'input' : {0 : 'batch_size', 2: 'height', 3:'width'},    # variable length axes
                                    'bboxes' : [0, 1], 'scores' : [0, 1]})
    # onnx_module = shape_inference.infer_shapes(onnx.load(output_model_path))
    # while len(onnx_module.graph.output) != 1:
    #     for output in onnx_module.graph.output:
    #         if output.name != 'output':
    #             print('--> remove', output.name)
    #             onnx_module.graph.output.remove(output)
    # graph = gs.import_onnx(onnx_module)
    # graph.cleanup()
    # graph.toposort()
    # graph.fold_constants().cleanup()
    # onnx.save_model(gs.export_onnx(graph), output_model_path)
    # print('Convert successfull !')


