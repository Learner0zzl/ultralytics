# -*- coding: utf-8 -*-
import os.path as osp
import shutil
from my_utils import split_img_for_yolo_classify
from ultralytics import YOLO

# Step0. 初始化
ONLY_SPLIT = True
# split param
project = "福源硅石矿"
subset = "0529_a1b15"
dataset = "55_FY_Si_cls"
val_ratio = -1
split_batch = 8
# train param
date = "0521"
epochs = 150
imgsz = 320
batch = 16
close_mosaic = epochs // 10
# export param
export_batch = 8
export_name = "C5.1.2"


def yolo_classify_split():
    src_dir = rf"E:\Data\raw\{project}\{subset}"
    dst_dir = rf"E:\Data\TrainSet\{dataset}\{subset}"
    split_img_for_yolo_classify(src_dir, dst_dir, val_ratio=val_ratio, batch_size=split_batch, mode="copy")
    if ONLY_SPLIT:
        project_dir = r"E:\Git\ultralytics\project"
        src = rf"{project_dir}\39_MT_M_cls"  # 分类用 39_MT_M_cls
        dst = rf"{project_dir}\{dataset}"
        if not osp.exists(dst):
            shutil.copytree(
                src=src,
                dst=dst,
                dirs_exist_ok=False,  # 目标目录已存在时报错
                copy_function=shutil.copy2
            )


def yolo_classify_train():
    # Load a model
    # build from YAML and transfer weights
    model = YOLO("yolo11n-cls.yaml").load(r"E:\Git\ultralytics\weights\yolo11n-cls.pt")

    # Train the model
    results = model.train(data=rf"E:\Data\TrainSet\{dataset}\{subset}\images",
                          cfg=rf"project\{dataset}\cfg.yaml",
                          epochs=epochs, imgsz=imgsz, batch=batch, close_mosaic=close_mosaic,
                          name=rf"{dataset}\{date}_e{epochs}_i{imgsz}_b{batch}")


def yolo_classify_export():
    # Load the YOLO11 model
    model_path = rf"E:\Git\ultralytics\runs\classify\{dataset}\{date}_e{epochs}_i{imgsz}_b{batch}\weights\best.pt"
    model = YOLO(model_path)
    model.export(format="onnx", batch=export_batch)

    # 输入固定为images   输出固定为output  下面方法已通过推理验证
    import onnx

    onnx_model_path = model_path.replace('.pt', '.onnx')
    model = onnx.load(onnx_model_path)
    model.graph.input[0].name = "images"
    model.graph.output[0].name = "output"
    model.graph.node[-1].output[0] = "output"
    onnx.checker.check_model(model, full_check=True)

    dst_dir = rf"E:\Data\TrainSet\{dataset}\{subset}"
    onnx.save(model, osp.join(dst_dir, f"{export_name}.onnx"))

if __name__ == '__main__':
    if ONLY_SPLIT:
        yolo_classify_split()
    else:
        # Step1. 划分数据集
        yolo_classify_split()

        # Step2. 训练模型
        yolo_classify_train()

        # Step3. 导出模型
        yolo_classify_export()
