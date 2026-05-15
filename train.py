# -*- coding: utf-8 -*-
import os.path as osp
import shutil
from my_utils import split_img_for_yolo_classify, extract_archive
from ultralytics import YOLO


if __name__ == '__main__':
    file_name = r"图古日格老虎口.zip"
    export_name = "C5.1.2"
    # 移动压缩包并解压缩
    download_dir = r"D:\Tencent\WXWork\1688854342747258\Cache\File\2026-05"
    data_root_dir = r"E:\Data"
    src = osp.join(download_dir, file_name)
    dst = osp.join(data_root_dir, "raw", file_name)
    if not osp.exists(dst):
        shutil.copyfile(src, dst)
    extract_archive(dst)
    # 划分数据集
    sub_name = osp.splitext(file_name)[0]
    src_dir = osp.join(data_root_dir, "raw", sub_name)
    dst_dir = osp.join(data_root_dir, "TrainSet", sub_name)
    split_img_for_yolo_classify(src_dir, dst_dir, val_ratio=-1, batch_size=16, mode="copy")
    print(f"数据集划分完成，数据集路径为：{dst_dir}")
    # Step2. 训练模型
    model = YOLO("yolo11n-cls.yaml").load(r"E:\Git\ultralytics\weights\yolo11n-cls.pt")  # build from YAML and transfer weights
    # Train the model
    results = model.train(data=rf"{dst_dir}\images",
                          cfg=rf"E:\Git\ultralytics\project\39_MT_M_cls\cfg.yaml",
                          epochs=150, imgsz=320, batch=16, close_mosaic=15,
                          name=rf"{sub_name}\0501_e150_i32_b16")
    # Step3. 导出模型
    # Load the YOLO11 model
    model_path = rf"E:\Git\ultralytics\runs\classify\{sub_name}\0501_e150_i32_b16\weights\best.pt"
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
    onnx.save(model, osp.join(dst_dir, f"{export_name}.onnx"))
