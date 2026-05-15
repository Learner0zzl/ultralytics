# -*- coding: utf-8 -*-
from ultralytics import YOLO


def train():
    # Load a model
    model = YOLO("yolo11n-cls.yaml").load(r"E:\Git\ultralytics\weights\yolo11n-cls.pt")  # build from YAML and transfer weights
    # model = YOLO(r"E:\Git\ultralytics\runs\classify\09_GN_coal_cls\1024_e150_i320_b16\weights\best.pt")  # build from YAML and transfer weights

    # Train the model
    subset = "0422_a3b15_v2"
    dataset = "35_MMSGLT_cls"
    results = model.train(data=rf"E:\Data\TrainSet\{dataset}\{subset}\images",
                          cfg=rf"project\{dataset}\cfg.yaml",
                          epochs=200, imgsz=160, batch=16, close_mosaic=20, patience=0,
                          name=rf"{dataset}\0422_e200_i160_b16_v2")


if __name__ == '__main__':
    train()
