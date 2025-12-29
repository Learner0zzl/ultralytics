import numpy as np
import os
import os.path as osp
from ultralytics import YOLO
from my_utils import find_files_by_ext, cv2_imread, cv2_imwrite, Timer, draw_labelme_annotation

# def postprocess(result):
#     cls = "unknown"
#     conf = 0
#     result = result.summary()
#     if len(result):
#         result.sort(key=lambda x: x["confidence"], reverse=True)
#         cls = result[0]["name"]
#         conf = result[0]["confidence"]
#
#     return cls, conf


def postprocess(result, shape, names_gan, names_mei):
    cls = "unknown"
    conf = 0.0
    dets = result.summary()
    print(dets)
    if len(shape) == 3:
        img_h, img_w, _ = shape
    else:
        img_h, img_w = shape
    box_big = []
    box_small = []
    # 遍历检测框，根据box的宽高分为大框和小框
    for det in dets:
        x1, x2, y1, y2 = det["box"].values()
        w, h = x2 - x1, y2 - y1
        if w > img_w * 0.5 and h > img_h * 0.5:
            box_big.append(det)
        else:
            box_small.append(det)
    # 进行逻辑判断
    if len(box_big):
        box_big.sort(key=lambda x: x["confidence"], reverse=True)
    if len(box_small):
        box_small.sort(key=lambda x: x["confidence"], reverse=True)
    if len(box_big):
        for det in box_big:
            if det['name'] in names_mei:
                cls = "煤"
                conf = det["confidence"]
                return cls, conf
            else:
                cls = "矸"
                conf = max(conf, det["confidence"])
        if len(box_small):
            for det in box_small:
                if det['name'] in names_gan:
                    cls = "矸"
                    conf = max(conf, det["confidence"])
                else:
                    cls = "煤"
                    conf = det["confidence"]
                    return cls, conf
    else:
        if len(box_small):
            for det in box_small:
                if det['name'] in names_gan:
                    cls = "矸"
                    conf = max(conf, det["confidence"])
                else:
                    cls = "煤"
                    conf = det["confidence"]
                    return cls, conf
        else:
            cls = "煤"

    return cls, conf


if __name__ == '__main__':
    model_name = r"1014_e150_i320_b16_cfg1"
    model = YOLO(rf"E:\Git\ultralytics\runs\detect\HYJTFJA_11_251013_320-det\{model_name}\weights\best.pt")
    parameters = {
        "conf": 0.1,
        "save_txt": False,
    }
    # names_gan = ["0-YZ_GS", "1-SL_GS", "2-QH_GS", "3-SH_GS", "4-JB_GS", "5-HQM_GS", "6-BDZ_GS", "7-RG_NS", "8-SLG_NS"]
    # names_mei = ["10-MH_NS", "11-MN_NS", "12-RM_NS", "13-5D_NS", "14-4D_M", "15-3D_JGM", "16_JGM", "17_YGM", "18_LM", "19-CZ_JM", "20_JM"]
    names_gan = ["0", "1", "2"]
    names_mei = ["5", "6", "7", "8", "9", "10"]
    # for cls in ["矸", "煤"]:
    #     root_dir = rf"E:\Data\04_FJA_coal\test\{cls}"
    root_dir = r"E:\Data\raw\HYJTFJA\HYJTFJA_11_250722_320_27-det\images\val"
    img_paths = find_files_by_ext(root_dir, '.bmp', mode="dict", func=lambda f: osp.splitext(f)[0])
    # json_dir = r"E:\Data\04_FJA_coal\HYJTFJA_21_250825_320-det\jsons"
    # # json_dir = r"E:\Data\HYJTFJA\HYJTFJA_11_250722_320_27-det\raw\jsons"
    json_dir = r"E:\Data\raw\HYJTFJA\HYJTFJA_11_250722_320_27-det\images\val"
    json_paths = find_files_by_ext(json_dir, '.json', mode="dict", func=lambda f: osp.splitext(f)[0])
    dst_dir = rf"{root_dir}_C5_{model_name}_conf0.1"
    os.makedirs(dst_dir, exist_ok=True)
    # 是否在原图上画标注进行对比
    show_label = True
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
        cls, conf = postprocess(result, src_img.shape, names_gan, names_mei)
        # result.show()  # display to screen
        # result.save(filename=rf"{img_path}_show.png")  # save to disk
        show_img = result.plot()
        img_name = osp.basename(img_path)
        old_cls = osp.basename(osp.dirname(img_path))
        dst_path = osp.join(dst_dir, cls, f"{conf}#" + img_name + f"#{old_cls}.png")
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
        # cv2_imwrite(dst_path, show_img, ".png")
        # if idx == 10:
        #     break
    print(timer.print_stats())
