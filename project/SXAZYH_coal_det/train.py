# -*- coding: utf-8 -*-
from ultralytics import YOLO


def train():
    # Load a model
    model = YOLO("yolo11n.yaml").load(r"E:\Git\ultralytics\weights\yolo11n.pt")  # build from YAML and transfer weights
    # model = YOLO(r"E:\Git\ultralytics\runs\segment\ore_seg\0812_e100_i2048_b4\weights\best.pt")  # build from YAML and transfer weights

    # Train the model
    results = model.train(data=r"project\SXAZYH_coal_det\dataset.yaml",
                          # cfg=r"project\SXAZYH_coal_det\cfg.yaml",
                          epochs=10, imgsz=160, batch=4, close_mosaic=10, device='cpu',
                          project=r"E:\Git\ultralytics\runs\detect\SXAZYH_coal_det",
                          name=r"0930_e10_i160_b4-tmp-cpu_test")

if __name__ == '__main__':
    train()
