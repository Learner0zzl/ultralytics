from ultralytics import YOLO

dataset = r"38_FJA_coal_cls"
# Load the YOLO11 model
model_path = rf"E:\Git\ultralytics\runs\classify\{dataset}\0422_e200_i320_b16_v3\weights\best.pt"
model = YOLO(model_path)
model.export(format="onnx", batch=16)

# 输入固定为images   输出固定为output  下面方法已通过推理验证
import onnx
onnx_model_path = model_path.replace('.pt', '.onnx')
model = onnx.load(onnx_model_path)
model.graph.input[0].name = "images"
model.graph.output[0].name = "output"
model.graph.node[-1].output[0] = "output"
onnx.checker.check_model(model, full_check=True)
onnx.save(model, onnx_model_path.replace('.onnx', '_renamed.onnx'))
