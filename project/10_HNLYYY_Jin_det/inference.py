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


# def postprocess(result, shape, names_gan, names_mei):
#     cls = "unknown"
#     conf = 0.0
#     dets = result.summary()
#     print(dets)
#     if len(shape) == 3:
#         img_h, img_w, _ = shape
#     else:
#         img_h, img_w = shape
#     box_big = []
#     box_small = []
#     # 遍历检测框，根据box的宽高分为大框和小框
#     for det in dets:
#         x1, x2, y1, y2 = det["box"].values()
#         w, h = x2 - x1, y2 - y1
#         if w > img_w * 0.5 and h > img_h * 0.5:
#             box_big.append(det)
#         else:
#             box_small.append(det)
#     # 进行逻辑判断
#     if len(box_big):
#         box_big.sort(key=lambda x: x["confidence"], reverse=True)
#     if len(box_small):
#         box_small.sort(key=lambda x: x["confidence"], reverse=True)
#     if len(box_big):
#         for det in box_big:
#             if det['name'] in names_mei:
#                 cls = "煤"
#                 conf = det["confidence"]
#                 return cls, conf
#             else:
#                 cls = "矸"
#                 conf = max(conf, det["confidence"])
#         if len(box_small):
#             for det in box_small:
#                 if det['name'] in names_gan:
#                     cls = "矸"
#                     conf = max(conf, det["confidence"])
#                 else:
#                     cls = "煤"
#                     conf = det["confidence"]
#                     return cls, conf
#     else:
#         if len(box_small):
#             for det in box_small:
#                 if det['name'] in names_gan:
#                     cls = "矸"
#                     conf = max(conf, det["confidence"])
#                 else:
#                     cls = "煤"
#                     conf = det["confidence"]
#                     return cls, conf
#         else:
#             cls = "煤"
#
#     return cls, conf


if __name__ == '__main__':
    model_name = r"1027_e100_i320_b16_cfg1_v2_from_1027_e100_i320_b16_cfg1"
    model = YOLO(rf"E:\Git\ultralytics\runs\detect\10_HNLYYY_Jin_det\{model_name}\weights\best.pt")
    parameters = {
        "conf": 0.5,
        "save_txt": False,
    }

    root_dir = r"E:\Data\raw\河南洛阳伊源金矿\org - 副本"
    img_paths = find_files_by_ext(root_dir, '.bmp', mode="dict", func=lambda f: osp.splitext(f)[0])

    dst_dir = rf"{root_dir}_C5_{model_name}_conf0.5"
    os.makedirs(dst_dir, exist_ok=True)
    # 是否在原图上画标注进行对比
    show_label = False
    timer = Timer()
    for idx, (name, img_path) in enumerate(img_paths.items()):
        # if "3-0.99316_C_1_20250616_171704_23" not in name:
        #     continue
        print(f"{idx+1}/{len(img_paths)}: \nimg_path={img_path}")
        src_img = cv2_imread(img_path)
        if src_img is None:
            continue
        if idx < 5:
            result = model.predict(src_img, **parameters)[0]
        else:
            timer.tic(stage="inference")
            result = model.predict(src_img, **parameters)[0]
            timer.hold()
        # print(result)
        show_img = result.plot(labels=False)
        img_name = osp.basename(img_path)
        cls, conf = postprocess(result)
        dst_path = osp.join(dst_dir, cls, f"{conf}#{img_name}.png")
        os.makedirs(osp.dirname(dst_path), exist_ok=True)
        cv2_imwrite(dst_path, show_img, ".png")
        # if idx == 10:
        #     break
    print(timer.print_stats())
