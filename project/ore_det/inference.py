import numpy as np
import os
import os.path as osp
from pathlib import Path
from ultralytics import YOLO
from my_utils import find_files_by_ext, cv2_imread, cv2_imwrite, Timer, draw_labelme_annotation, yolo_det2labelme, readTxt, yolo_txt2labelme

def postprocess(result):
    cls = "unknown"
    conf = 0
    result = result.summary()
    if len(result):
        result.sort(key=lambda x: x["confidence"], reverse=True)
        cls = result[0]["name"]
        conf = result[0]["confidence"]

    return cls, conf


if __name__ == '__main__':
    model_name = r"0904_e50_i320_b8_cfg0_new_data"
    model = YOLO(rf"E:\Git\ultralytics\runs\detect\ore_det\{model_name}\weights\best.pt")
    parameters = {
        "save_txt": False,
    }
    root_dir = r"E:\Data\HYJTFJA\HYJTFJA_11_250722_320_27-det\images\val"
    img_paths = find_files_by_ext(root_dir, '.bmp', mode="dict", func=lambda f: osp.splitext(f)[0])
    txt_dir = r"E:\Data\HYJTFJA\HYJTFJA_11_250722_320_27-det\labels_raw\val"
    txt_paths = find_files_by_ext(txt_dir, '.txt', mode="dict", func=lambda f: osp.splitext(f)[0])
    dst_dir = rf"{root_dir}_{model_name}"
    os.makedirs(dst_dir, exist_ok=True)
    # 是否在原图上画标注进行对比
    show_label = False
    timer = Timer()
    for idx, (name, img_path) in enumerate(img_paths.items()):
        print(f"{idx+1}/{len(img_paths)}: \nimg_path={img_path}")
        # 读txt标注，判断是否需要处理
        txt_path = txt_paths.get(name)
        print(f"txt_path={txt_path}")
        txt_data = readTxt(txt_path)
        if len(txt_data) == 1 and txt_data[0][2:] == "0.5 0.5 0.99 0.99":
            src_img = cv2_imread(img_path)
            if src_img is None:
                continue
            # if "C_1_20250612_002958_1334" not in name:
            #     continue
            if idx < 5:
                result = model.predict(src_img, **parameters)[0]
            else:
                timer.tic(stage="inference")
                result = model.predict(src_img, **parameters)[0]
                timer.hold()
            # print(result)
            # 转为labelme格式
            img_height, img_width, _ = src_img.shape
            new_json_path = img_path.replace(".bmp", ".json")
            class_name = Path(img_path).parent.name
            yolo_det2labelme(result.summary(), img_path, img_width, img_height, json_path=new_json_path)
            show_img = result.plot()
            img_name = osp.basename(img_path)
            old_cls = osp.basename(osp.dirname(img_path))
            dst_path = osp.join(dst_dir, img_name)
            os.makedirs(osp.dirname(dst_path), exist_ok=True)
            # if show_label:
            #     timer.tic(stage="draw_label")
            #     json_path = json_paths[name]
            #     print(f"json_path={json_path}")
            #     if json_path:
            #         label_img = draw_labelme_annotation(src_img, json_path)
            #     else:
            #         label_img = src_img.copy()
            #     timer.hold()
            #     show_img = np.hstack((label_img, show_img))
            cv2_imwrite(dst_path, show_img, ".png")
            if idx > 30:
                break
        else:
            json_dir = Path(img_path).parent
            class_name = Path(img_path).parent.name
            cls2name = {int(class_name): class_name}
            yolo_txt2labelme(txt_path, json_dir, cls2name, (320, 320), ".bmp")
    print(timer.print_stats())
