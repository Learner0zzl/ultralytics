# -*- coding: utf-8 -*-
from ultralytics import YOLO


def train():
    # Load a model
    model = YOLO("yolo11n-cls.yaml").load(r"E:\Git\ultralytics\weights\yolo11n-cls.pt")  # build from YAML and transfer weights
    # model = YOLO(r"E:\Git\ultralytics\runs\detect\HYJTFJW_coal_det\0829_e150_i320_b8_cfg1\weights\best.pt")  # build from YAML and transfer weights

    # Train the model
    results = model.train(data=r"E:\Data\06_ZYZK_cls\images",
                          cfg=r"project\06_ZYZK_cls\cfg.yaml",
                          epochs=200, imgsz=160, batch=16, close_mosaic=20,
                          name=r"06_ZYZK_cls\1011_e200_i160_b16")

if __name__ == '__main__':
    train()
