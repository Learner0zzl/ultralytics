# -*- coding: utf-8 -*-
from ultralytics import YOLO


def train():
    # Load a model
    # model = YOLO("yolo11n.yaml").load(r"E:\Git\ultralytics\weights\yolo11n.pt")  # build from YAML and transfer weights
    model = YOLO(r"E:\Git\ultralytics\runs\detect\tune\weights\best.pt")  # build from YAML and transfer weights

    # Train the model
    results = model.train(data=r"project\ore_det\dataset.yaml",
                          cfg=r"project\ore_det\cfg0.yaml",
                          epochs=100, imgsz=320, batch=16, close_mosaic=20,
                          name=r"ore_det\1014_e100_i320_b16_from_tune")

if __name__ == '__main__':
    train()
