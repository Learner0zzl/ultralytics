# -*- coding: utf-8 -*-
from ultralytics import YOLO


def train():
    # Load a model
    model = YOLO("yolo11n.yaml").load(r"E:\Git\ultralytics\weights\yolo11n.pt")  # build from YAML and transfer weights
    # model = YOLO(r"E:\Git\ultralytics\runs\detect\HYJTFJW_coal_det\0829_e150_i320_b8_cfg1\weights\best.pt")  # build from YAML and transfer weights

    # Train the model
    results = model.train(data=r"project\ore_det\dataset.yaml",
                          cfg=r"project\ore_det\cfg0.yaml",
                          epochs=50, imgsz=320, batch=8, close_mosaic=20, save_period=50,
                          name=r"ore_det\0904_e50_i320_b8_cfg0_new_data")

if __name__ == '__main__':
    train()
