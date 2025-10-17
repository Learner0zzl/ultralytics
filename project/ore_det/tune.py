from ultralytics import YOLO

# Initialize the YOLO model
model = YOLO("yolo11n.yaml").load(r"E:\Git\ultralytics\weights\yolo11n.pt")

# Define search space
search_space = {
    "lr0": (1e-4, 1e-1),
    "momentum": (0.6, 0.98),
    "hsv_h": (0.0, 0.1),
    "hsv_s": (0.0, 0.9),
    "hsv_v": (0.0, 0.9),
    "mosaic": (0.0, 1.0),
}

# Tune hyperparameters
model.tune(
    data=r"project\ore_det\dataset.yaml",
    imgsz=320,
    epochs=50,
    batch=16,
    iterations=100,
    optimizer="auto",
    space=search_space,
    flipud=0.5,
    fliplr=0.5,
    plots=True,
    save=False,
    val=False,
)