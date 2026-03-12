# -*- coding: utf-8 -*-
from ultralytics import YOLO


def train():
    # Load a model
    model = YOLO("yolo11n.yaml").load(r"E:\Git\ultralytics\weights\yolo11n.pt")  # build from YAML and transfer weights
    # model = YOLO(r"E:\Git\ultralytics\runs\detect\HYJTFJW_coal_det\0829_e150_i320_b8_cfg1\weights\best.pt")  # build from YAML and transfer weights
    print(model.info(verbose=True))
    # Train the model
    results = model.train(data=r"project\26_DBS_Cu_det\dataset.yaml",
                          cfg=r"project\26_DBS_Cu_det\cfg.yaml",
                          epochs=300, imgsz=640, batch=16, close_mosaic=30,
                          project=r"E:\Git\ultralytics\runs\detect\26_DBS_Cu_det",
                          name=r"0310_v2_e300_i640_b16_all")

if __name__ == '__main__':
    train()
