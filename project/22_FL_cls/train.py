# -*- coding: utf-8 -*-
from ultralytics import YOLO


def train():
    # Load a model
    model = YOLO("yolo11n-cls.yaml").load(r"E:\Git\ultralytics\weights\yolo11n-cls.pt")  # build from YAML and transfer weights
    # model = YOLO(r"E:\Git\ultralytics\runs\classify\09_GN_coal_cls\1024_e150_i320_b16\weights\best.pt")  # build from YAML and transfer weights

    # Train the model
    results = model.train(data=r"E:\Data\TrainSet\22_FL_cls\0205_a1.3b20\images",
                          cfg=r"project\22_FL_cls\cfg.yaml",
                          epochs=100, imgsz=320, batch=16, close_mosaic=10,
                          name=r"22_FL_cls\0205_e100_i320_b16")


if __name__ == '__main__':
    train()
