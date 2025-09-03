# -*- coding: utf-8 -*-
from ultralytics import YOLO


def train():
    # Load a model
    # model = YOLO("yolo11n.yaml").load(r"E:\Git\ultralytics\weights\yolo11n.pt")  # build from YAML and transfer weights
    model = YOLO(r"E:\Git\ultralytics\runs\detect\HYJTFJW_coal_det\0829_e150_i320_b8_cfg1\weights\best.pt")  # build from YAML and transfer weights

    # Train the model
    results = model.train(data=r"project\HYJTFJW_coal_det\dataset.yaml",
                          cfg=r"project\HYJTFJW_coal_det\cfg1.yaml",
                          epochs=150, imgsz=320, batch=8, close_mosaic=50, save_period=10,
                          name=r"HYJTFJW_coal_det\0829_e150_i320_b8_cfg1_v2_from_0829_e150_i320_b8_cfg1")

if __name__ == '__main__':
    train()
