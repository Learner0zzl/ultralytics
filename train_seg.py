from ultralytics import YOLO


if __name__ == '__main__':
    # Load a model
    # model = YOLO("yolo11n-seg.yaml")  # build a new model from YAML
    # model = YOLO("yolo11n-seg.pt")  # load a pretrained model (recommended for training)
    model = YOLO("yolo26n-seg.yaml").load("yolo26n-seg.pt")  # build from YAML and transfer weights

    # Train the model
    results = model.train(data="coco8-seg.yaml", epochs=10, imgsz=640)
