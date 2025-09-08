# -*- coding: utf-8 -*-
from ultralytics import YOLO


def train():
    # Load a model
    model = YOLO("yolo11n.yaml").load(r"E:\Git\ultralytics\weights\yolo11n.pt")  # build from YAML and transfer weights
    # model = YOLO(r"E:\Git\ultralytics\runs\detect\HYJTFJW_coal_det\0829_e150_i320_b8_cfg1\weights\best.pt")  # build from YAML and transfer weights

    # Train the model
    results = model.train(data=r"project\HYJTFJA_21_250825_320-det\dataset.yaml",
                          cfg=r"project\HYJTFJA_21_250825_320-det\cfg1.yaml",
                          epochs=200, imgsz=320, batch=16, close_mosaic=50,
                          name=r"HYJTFJA_21_250825_320-det\0908_e200_i320_b16_cfg1_tv")

if __name__ == '__main__':
    train()
