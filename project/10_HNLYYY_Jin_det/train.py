# -*- coding: utf-8 -*-
from ultralytics import YOLO


def train():
    # Load a model
    # model = YOLO("yolo11n.yaml").load(r"E:\Git\ultralytics\weights\yolo11n.pt")  # build from YAML and transfer weights
    model = YOLO(r"E:\Git\ultralytics\runs\detect\10_HNLYYY_Jin_det\1027_e100_i320_b16_cfg1\weights\best.pt")  # build from YAML and transfer weights
    print(model.info(verbose=True))
    # Train the model
    results = model.train(data=r"project\10_HNLYYY_Jin_det\dataset.yaml",
                          cfg=r"project\10_HNLYYY_Jin_det\cfg1.yaml",
                          epochs=100, imgsz=320, batch=16, close_mosaic=20,
                          project=r"E:\Git\ultralytics\runs\detect\10_HNLYYY_Jin_det",
                          name=r"1027_e100_i320_b16_cfg1_v2_from_1027_e100_i320_b16_cfg1")

if __name__ == '__main__':
    train()
