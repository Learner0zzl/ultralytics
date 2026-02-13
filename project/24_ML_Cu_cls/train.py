# -*- coding: utf-8 -*-
from ultralytics import YOLO


def train():
    # Load a model
    model = YOLO("yolo11n-cls.yaml").load(r"E:\Git\ultralytics\weights\yolo11n-cls.pt")  # build from YAML and transfer weights
    # model = YOLO(r"E:\Git\ultralytics\runs\classify\09_GN_coal_cls\1024_e150_i320_b16\weights\best.pt")  # build from YAML and transfer weights

    # Train the model
    results = model.train(data=r"E:\Data\TrainSet\24_BL_Cu_cls\0211_a1b15\images",
                          cfg=r"project\24_BL_Cu_cls\cfg.yaml",
                          epochs=150, imgsz=320, batch=16, close_mosaic=15,
                          name=r"24_BL_Cu_cls\0205_e150_i320_b16")


if __name__ == '__main__':
    train()
