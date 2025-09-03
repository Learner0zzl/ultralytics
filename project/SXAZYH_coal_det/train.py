# -*- coding: utf-8 -*-
from ultralytics import YOLO


def train():
    # Load a model
    model = YOLO("yolo11n.yaml").load(r"E:\Git\ultralytics\weights\yolo11n.pt")  # build from YAML and transfer weights
    # model = YOLO(r"E:\Git\ultralytics\runs\segment\ore_seg\0812_e100_i2048_b4\weights\best.pt")  # build from YAML and transfer weights

    # Train the model
    results = model.train(data=r"project\SXAZYH_coal_det\dataset.yaml",
                          cfg=r"project\SXAZYH_coal_det\cfg.yaml",
                          epochs=150, imgsz=160, batch=8, close_mosaic=50,
                          name=r"SXAZYH_coal_det\0821_e150_i160_b8")

if __name__ == '__main__':
    train()
