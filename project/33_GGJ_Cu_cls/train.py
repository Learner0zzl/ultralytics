# -*- coding: utf-8 -*-
from ultralytics import YOLO


def train():
    # Load a model
    model = YOLO("yolo11n-cls.yaml").load(r"E:\Git\ultralytics\weights\yolo11n-cls.pt")  # build from YAML and transfer weights
    # model = YOLO(r"E:\Git\ultralytics\runs\classify\09_GN_coal_cls\1024_e150_i320_b16\weights\best.pt")  # build from YAML and transfer weights

    # Train the model
    subset = "0416_a1.8b15"
    dataset = r"33_GGJ_Cu_cls"
    results = model.train(data=rf"E:\Data\TrainSet\{dataset}\{subset}\images",
                          cfg=rf"project\{dataset}\cfg.yaml",
                          epochs=150, imgsz=640, batch=8, close_mosaic=15,
                          name=rf"{dataset}\0416_e150_i640_b8")


if __name__ == '__main__':
    train()
