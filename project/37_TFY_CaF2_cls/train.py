# -*- coding: utf-8 -*-
from ultralytics import YOLO


def train():
    # Load a model
    model = YOLO("yolo11n-cls.yaml").load(r"E:\Git\ultralytics\weights\yolo11n-cls.pt")  # build from YAML and transfer weights
    # model = YOLO(r"E:\Git\ultralytics\runs\classify\09_GN_coal_cls\1024_e150_i320_b16\weights\best.pt")  # build from YAML and transfer weights

    # Train the model
    subset = "0515_a1.5b12"
    dataset = "37_TFY_CaF2_cls"
    name = "0515_e200_i640_b8"
    results = model.train(data=rf"E:\Data\TrainSet\{dataset}\{subset}\images",
                          cfg=rf"project\{dataset}\cfg.yaml",
                          epochs=200, imgsz=640, batch=8, close_mosaic=20,
                          name=rf"{dataset}\{name}")


if __name__ == '__main__':
    train()
