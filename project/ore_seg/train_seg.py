from ultralytics import YOLO


def train():
    # Load a model
    model = YOLO("yolo11n-seg.yaml").load(r"E:\Git\ultralytics\weights\yolo11n-seg.pt")  # build from YAML and transfer weights
    # model = YOLO(r"E:\Git\ultralytics\runs\segment\ore_seg\0812_e100_i2048_b4\weights\best.pt")  # build from YAML and transfer weights

    # Train the model
    results = model.train(data=r"project\ore_seg\dataset.yaml",
                          cfg=r"project\ore_seg\cfg.yaml",
                          epochs=100, imgsz=2048, batch=4,
                          name=r"ore_seg\0814_e100_i2048_b4")

if __name__ == '__main__':
    train()
