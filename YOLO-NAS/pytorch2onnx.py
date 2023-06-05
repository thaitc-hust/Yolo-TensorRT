import torch
import onnx 
from models.experimental import attempt_load
import argparse
import onnx_graphsurgeon as gs 
from onnx import shape_inference
# from models.yolo import ClassificationModel, Detect, DetectionModel, SegmentationModel
import torch.nn as nn
from super_gradients.training import models
from super_gradients.common.object_names import Models

class YOLOnasAddNMS(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.model.eval()

    def forward(self, input):
        """ 
            Split output [n_batch, 84, n_bboxes] to 3 output: bboxes, scores, classes
        """ 
        input = input * 255.0
        # x, y, w, h -> x1, y1, x2, y2
        output = self.model(input)
        # print('Output: ', len(output))
        # for x in output:
        #     if type(x).__name__ == 'tuple':
        #         print([y.shape for y in x])
        #     else:
        #         print('single ', x.shape)
        output = output[0]
        print("[INFO] Output's origin model shape: ",output.shape)
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
        print(scores)
        print(bboxes)
        print(scores.shape)
        print(bboxes.shape)
        # exit(1)
        # print(scores)
        # print(bboxes)
        return bboxes, scores

    


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str, default='/home/data2/hungpham/Triton_Infer/YOLO-to-TensorRT-NMSBatched/Yolo-nas/weights/yolo_nas_tattoo.pth', help='weights path')
    # parser.add_argument('--cfg', type=str, default='cfg/yolo_nas.cfg', help='config path')
    parser.add_argument('--output', type=str, default='/home/data2/hungpham/Triton_Infer/YOLO-to-TensorRT-NMSBatched/Yolo-nas/weights/yolonas_coco_v2.onnx', help='output ONNX model path')
    parser.add_argument('--max_size', type=int, default=416, help='max size of input image')
    opt = parser.parse_args()

    # model_cfg = opt.cfg
    model_weights = opt.weights
    output_model_path = opt.output
    max_size = opt.max_size
    device = torch.device("cpu")

    # load model 
    # model = attempt_load(model_weights, device=device, inplace=True, fuse=True)
    # model.eval()
    img = torch.zeros(1, 3, max_size, max_size).to(device)
    # print(img)

    # for _ in range(2):
    #     y = model(img)  # dry runs
    print('[INFO] Convert from Torch to ONNX')
    model = models.get(
        Models.YOLOX_S,
        pretrained_weights="coco"
    ).to(device)

    # model = models.get(
    #         model_name='yolo_nas_s',
    #         checkpoint_path=model_weights,
    #         num_classes=1
    #     ).to(device)

    model = YOLOnasAddNMS(model)
    model.to(device).eval()

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

    print('[INFO] Finished Convert!')

