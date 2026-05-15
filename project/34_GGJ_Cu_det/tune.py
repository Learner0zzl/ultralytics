from ultralytics import YOLO

# Initialize the YOLO model
# model = YOLO("yolo11n.yaml").load(r"E:\Git\ultralytics\weights\yolo11n.pt")
model = YOLO(r"E:\Git\ultralytics\runs\detect\HYJTFJA_11_251013_320-det\1014_e150_i320_b16_cfg1\weights\best.pt")

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
    data=r"project\HYJTFJA_11_251013_320-det\dataset.yaml",
    imgsz=320,
    epochs=30,
    batch=16,
    iterations=50,
    optimizer="auto",
    space=search_space,
    flipud=0.5,
    fliplr=0.5,
    plots=True,
    save=False,
    val=False,
    project=r"E:\Git\ultralytics\runs\detect\HYJTFJA_11_251013_320-det"
)