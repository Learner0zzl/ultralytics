import warnings
warnings.filterwarnings('ignore')
warnings.simplefilter('ignore')
import torch
import cv2
import numpy as np
import requests
import torchvision.transforms as transforms
from pytorch_grad_cam import EigenCAM, GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image, scale_cam_image
from PIL import Image
from ultralytics import YOLO

COLORS = np.random.uniform(0, 255, size=(80, 3))

def parse_detections(results):
    # detections = results.pandas().xyxy[0]
    # detections = results.to_dict()
    boxes, colors, names = [], [], []

    for det in results[0].summary():
        confidence = det["confidence"]
        if confidence < 0.2:
            continue
        x1 = int(det["box"]['x1'])
        y1 = int(det["box"]['y1'])
        x2 = int(det["box"]['x2'])
        y2 = int(det["box"]['y2'])
        name = det["name"]
        category = det["class"]
        color = COLORS[category]

        boxes.append((x1, y1, x2, y2))
        colors.append(color)
        names.append(name)
    return boxes, colors, names


def draw_detections(boxes, colors, names, img):
    for box, color, name in zip(boxes, colors, names):
        xmin, ymin, xmax, ymax = box
        cv2.rectangle(
            img,
            (xmin, ymin),
            (xmax, ymax),
            color,
            2)

        cv2.putText(img, name, (xmin, ymin - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2,
                    lineType=cv2.LINE_AA)
    return img


img_path = r"E:\Data\HYJTFJA\HYJTFJA_11_250722_320_27-det\images\val\8\8-0.31445_C_1_20250621_092337_80.bmp"
img = np.array(Image.open(img_path))
img = cv2.resize(img, (320, 320))
rgb_img = img.copy()
img = np.float32(img) / 255
transform = transforms.ToTensor()
tensor = transform(img).unsqueeze(0)

# model = YOLO(r"E:\Git\ultralytics\runs\detect\HYJTFJW_coal_det\0829_e150_i320_b8_cfg1\weights\best.pt")
# # model = YOLO(r"E:\Git\ultralytics\weights\yolov5s.pt")
# model.eval()
# model.cpu()
# target_layers = model.model.model[-2]
# print(target_layers)

model = YOLO(r"E:\Git\ultralytics\ultralytics\cfg\models\11\yolo11.yaml").load(rf"E:\Git\ultralytics\runs\detect\HYJTFJW_coal_det\0829_e150_i320_b8_cfg1\weights\best.pt")
model.to(torch.device('cuda:0')).eval()

results = model([rgb_img])
boxes, colors, names = parse_detections(results)
detections = draw_detections(boxes, colors, names, rgb_img.copy())
cv2.imwrite("detections.jpg", detections)

model = model.model.model
target_layers = [model[-2]]

# cam = EigenCAM(model, target_layers)
cam = GradCAM(model=model, target_layers=target_layers)
grayscale_cam = cam(tensor, targets=target_layers)[0, :]
cam_image = show_cam_on_image(img, grayscale_cam, use_rgb=True)
cv2.imwrite("cam.jpg", cam_image)