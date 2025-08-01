import utils

from ultralytics import YOLO
from utils import *


if __name__ == '__main__':
    model = YOLO(r"E:\Git\ultralytics\runs\segment\ore_seg\0801_e50_i2048_b4\weights\best.pt")  # pretrained YOLO11n model
    parameters = {
        "save_txt": False,
    }
    img_paths = find_image_files(r"E:\Data\JLHD\images", 'jpg')
    for idx, img_path in enumerate(img_paths):
        print(f"{idx+1}/{len(img_paths)}: img_path={img_path}")
        src_img = cv2_imread(img_path)
        result = model.predict(src_img, **parameters)[0]
        # result.show()  # display to screen
        # result.save(filename=rf"{img_path}_show.png")  # save to disk
        show_img = result.plot()
        cv2_imwrite(f"{img_path}_show.png", show_img, ".png")
