# -*- coding: utf-8 -*-
from ultralytics import YOLO


def train():
    # Load a model
    model = YOLO("yolo11n.yaml").load(r"E:\Git\ultralytics\weights\yolo11n.pt")  # build from YAML and transfer weights
    # model = YOLO(r"E:\Git\ultralytics\runs\detect\HYJTFJW_coal_det\0829_e150_i320_b8_cfg1\weights\best.pt")  # build from YAML and transfer weights
    print(model.info(verbose=True))
    # Train the model
    results = model.train(data=r"project\HYJTFJA_11_251013_320-det\dataset.yaml",
                          cfg=r"project\HYJTFJA_11_251013_320-det\cfg3.yaml",
                          epochs=150, imgsz=320, batch=16, close_mosaic=20,
                          project=r"E:\Git\ultralytics\runs\detect\HYJTFJA_11_251013_320-det",
                          name=r"1014_e150_i320_b16_cfg1")

if __name__ == '__main__':
    train()
