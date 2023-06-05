import torch
import torchvision
from super_gradients.training import models

model_weights = "/home/data2/hungpham/Triton_Infer/YOLO-to-TensorRT-NMSBatched/Yolo-nas/weights/yolo_nas_tattoo.pth"

def inference_func(model, image):
    inputs = [{"image": image}]
    return model.inference(inputs, do_postprocess=False)[0]

model = models.get(
        model_name='yolo_nas_s',
        checkpoint_path=model_weights,
        num_classes=1
    )
model.eval()
example = torch.rand(1, 3, 416, 416)

traced_script_module = torch.jit.trace(model, example)
traced_script_module.save("/home/data2/hungpham/Triton_Infer/YOLO-to-TensorRT-NMSBatched/Yolo-nas/weights/model-final-ylnas.pt")