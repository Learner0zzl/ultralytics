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


def postprocess2(result, shape, names_gan, names_mei):
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
    model_name = r"0310_v2_e300_i640_b16_all"
    model = YOLO(rf"E:\Git\ultralytics\runs\detect\26_DBS_Cu_det\{model_name}\weights\best.pt")
    parameters = {
        "conf": 0.025,
        "save_txt": False,
    }

    sub = "Train"
    root_dir = rf"E:\Data\TrainSet\26_DBS_Cu_det\Test\{sub}"
    img_paths = find_files_by_ext(root_dir, '.bmp', mode="dict", func=lambda f: osp.splitext(f)[0])
    json_dir = r"E:\Data\TrainSet\26_DBS_Cu_det\jsons"
    json_paths = find_files_by_ext(json_dir, '.json', mode="dict", func=lambda f: osp.splitext(f)[0])
    # dst_dir = rf"{root_dir}/C5_{model_name}_conf0.1"
    dst_dir = rf"E:\Data\TrainSet\26_DBS_Cu_det\Test\C5_{model_name}_conf{parameters['conf']}_{sub}"
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
        cls, conf = postprocess(result)
        # result.show()  # display to screen
        # result.save(filename=rf"{img_path}_show.png")  # save to disk
        show_img = result.plot()
        img_name = osp.basename(img_path)
        old_cls = osp.basename(osp.dirname(img_path))
        dst_path = osp.join(dst_dir, cls, f"{conf}#" + img_name + f"#{old_cls}.png")
        os.makedirs(osp.dirname(dst_path), exist_ok=True)
        if show_label:
            timer.tic(stage="draw_label")
            json_path = json_paths.get(name, None)
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
