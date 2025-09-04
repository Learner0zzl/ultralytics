import sys

sys.path.append(r"E:\PythonProject")

import numpy as np
import os
import os.path as osp
from ultralytics import YOLO
from my_utils import find_files_by_ext, cv2_imread, cv2_imwrite, Timer, draw_labelme_annotation

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
    model_name = r"0829_e150_i320_b8_cfg1"
    model = YOLO(rf"E:\Git\ultralytics\runs\detect\HYJTFJW_coal_det\{model_name}\weights\best.pt")
    parameters = {
        "conf": 0.1,
        "save_txt": False,
    }
    # root_dir = r"E:\Data\HYJTFJA\0825_labeled\testset_320"
    # root_dir = r"E:\Data\HYJTFJA\HYJTFJA_11_250722_320_27-det\images\val"
    root_dir = r"E:\Data\HYJTFJA\HYJTFJA_11_250722_320_27-det\images\val_8_unknown"
    img_paths = find_files_by_ext(root_dir, '.bmp', mode="dict", func=lambda f: osp.splitext(f)[0])
    # json_paths = find_files_by_ext(root_dir, '.json', mode="dict", func=lambda f: osp.splitext(f)[0])
    json_dir = r"E:\Data\HYJTFJA\HYJTFJA_11_250722_320_27-det\raw\jsons"
    json_paths = find_files_by_ext(json_dir, '.json', mode="dict", func=lambda f: osp.splitext(f)[0])
    dst_dir = rf"{root_dir}_C5_{model_name}_conf0.1"
    os.makedirs(dst_dir, exist_ok=True)
    # 是否在原图上画标注进行对比
    show_label = True
    timer = Timer()
    for idx, (name, img_path) in enumerate(img_paths.items()):
        print(f"{idx+1}/{len(img_paths)}: \nimg_path={img_path}")
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
        cls, conf = postprocess(result)
        # result.show()  # display to screen
        # result.save(filename=rf"{img_path}_show.png")  # save to disk
        show_img = result.plot()
        img_name = osp.basename(img_path)
        old_cls = osp.basename(osp.dirname(img_path))
        dst_path = osp.join(dst_dir, cls, str(conf) + "#" + img_name + f"#{old_cls}.png")
        os.makedirs(osp.dirname(dst_path), exist_ok=True)
        if show_label:
            timer.tic(stage="draw_label")
            json_path = json_paths[name]
            print(f"json_path={json_path}")
            if json_path:
                label_img = draw_labelme_annotation(src_img, json_path)
            else:
                label_img = src_img.copy()
            timer.hold()
            show_img = np.hstack((label_img, show_img))
        cv2_imwrite(dst_path, show_img, ".png")
        # if idx == 10:
        #     break
    print(timer.print_stats())
