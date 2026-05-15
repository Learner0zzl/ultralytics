# -*- coding: utf-8 -*-
import os.path as osp
import shutil
from my_utils import split_img_for_yolo_classify
from ultralytics import YOLO


if __name__ == '__main__':
    # Step0. 初始化
    project = "土耳其铝土矿"
    subset = "0424_a1b5"
    dataset = "07_TRQ_Lv_cls"
    # train param
    date = "0501"
    epochs = 150
    imgsz = 320
    batch = 16
    close_mosaic = epochs // 10
    # export param
    export_batch = 8
    export_name = "C5.1.2"
    # Step1. 划分数据集
    # YOLO Classify split
    src_dir = rf"E:\Data\raw\{project}\{subset}"
    dst_dir = rf"E:\Data\TrainSet\{dataset}\{subset}"
    split_img_for_yolo_classify(src_dir, dst_dir, val_ratio=-1, batch_size=16, mode="copy")

    # Step2. 训练模型
    # YOLO Classify train
    # Load a model
    model = YOLO("yolo11n-cls.yaml").load(r"E:\Git\ultralytics\weights\yolo11n-cls.pt")  # build from YAML and transfer weights
    # model = YOLO(r"E:\Git\ultralytics\runs\detect\HYJTFJW_coal_det\0829_e150_i320_b8_cfg1\weights\best.pt")  # build from YAML and transfer weights

    # Train the model
    results = model.train(data=rf"E:\Data\TrainSet\{dataset}\{subset}\images",
                          cfg=rf"project\{dataset}\cfg.yaml",
                          epochs=epochs, imgsz=imgsz, batch=batch, close_mosaic=close_mosaic,
                          name=rf"{dataset}\{date}_e{epochs}_i{imgsz}_b{batch}")

    # Step3. 导出模型
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
    onnx.save(model, osp.join(dst_dir, f"{export_name}.onnx"))
