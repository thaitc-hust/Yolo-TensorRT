import torch
import argparse
import onnx_graphsurgeon as gs
import onnx
from onnx import shape_inference
from models.models import Darknet


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str, default='yolor_csp_x_star.pt', help='weights path')
    parser.add_argument('--cfg', type=str, default='cfg/yolor_csp_x.cfg', help='config path')
    parser.add_argument('--output', type=str, default='yolor_csp_x.onnx', help='output ONNX model path')
    parser.add_argument('--max_size', type=int, default=448, help='max size of input image')
    opt = parser.parse_args()
    model_cfg = opt.cfg
    model_weights = opt.weights 
    output_model_path = opt.output
    max_size = opt.max_size

    device = torch.device('cuda')
    # Load model
    model = Darknet(model_cfg, (max_size, max_size)).cuda()
    model.load_state_dict(torch.load(model_weights, map_location=device)['model'])
    model.to(device).eval()
    img = torch.zeros((1, 3, max_size, max_size), device=device)  # init img
    print('Convert from Torch to ONNX')
    # Export the model
    torch.onnx.export(model,               # model being run
                      img,                         # model input (or a tuple for multiple inputs)
                      output_model_path,   # where to save the model (can be a file or file-like object)
                      export_params=True,        # store the trained parameter weights inside the model file
                      opset_version=11,          # the ONNX version to export the model to
                      do_constant_folding=True,  # whether to execute constant folding for optimization
                      input_names = ['input'],   # the model's input names
                      output_names = ['output'], # the model's output names
                      dynamic_axes={'input' : {0 : 'batch_size', 2: 'height', 3:'width'},    # variable length axes
                                    'output' : {0 : 'batch_size', 1: 'n_boxes'}})
    print('Remove unused outputs')
    onnx_module = shape_inference.infer_shapes(onnx.load(output_model_path))
    while len(onnx_module.graph.output) != 1:
        for output in onnx_module.graph.output:
            if output.name != 'output':
                print('--> remove', output.name)
                onnx_module.graph.output.remove(output)
    graph = gs.import_onnx(onnx_module)
    graph.cleanup()
    graph.toposort()
    graph.fold_constants().cleanup()
    onnx.save_model(gs.export_onnx(graph), output_model_path)
    print('Convert successfull !')


