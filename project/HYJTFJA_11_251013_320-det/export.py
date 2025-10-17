from ultralytics import YOLO

# Load the YOLO11 model
# model_path = r"E:\Git\ultralytics\runs\detect\SXAZYH_coal_det\0820_e150_i160_b8\weights\best.pt"
# model = YOLO(model_path)
# model.export(format="onnx")

# 输入固定为images   输出固定为output  下面方法已通过推理验证
# import onnx
# onnx_model_path = r"E:\Git\ultralytics\runs\detect\SXAZYH_coal_det\0820_e150_i160_b8\weights\C5_160_batchsize_8.onnx"
# model = onnx.load(onnx_model_path)
# model.graph.input[0].name = "images"
# model.graph.output[0].name = "output"
# model.graph.node[-1].output[0] = "output"
# onnx.checker.check_model(model, full_check=True)
# onnx.save(model, onnx_model_path.replace('.onnx', '_renamed.onnx'))
