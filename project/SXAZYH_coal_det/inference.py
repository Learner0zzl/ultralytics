import sys

sys.path.append(r"E:\PythonProject")


import os
import os.path as osp
from ultralytics import YOLO
from my_utils import find_files_by_ext, cv2_imread, cv2_imwrite, Timer

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
    model_name = r"0820_e150_i160_b8"
    model = YOLO(rf"E:\Git\ultralytics\runs\detect\SXAZYH_coal_det\{model_name}\weights\best.pt")
    # model = YOLO(rf"E:\Git\ultralytics\runs\detect\SXAZYH_coal_det\{model_name}\weights\C5_160_batchsize_8_renamed.onnx")
    parameters = {
        "save_txt": False,
    }
    root_dir = r"E:\Data\SXAZYH\C4.27.10.D-3060_1per"
    img_paths = find_files_by_ext(root_dir, 'bmp')
    dst_dir = rf"{root_dir}_C5_{model_name}_show"
    os.makedirs(dst_dir, exist_ok=True)
    timer = Timer()
    for idx, img_path in enumerate(img_paths):
        print(f"{idx+1}/{len(img_paths)}: img_path={img_path}")
        src_img = cv2_imread(img_path)
        if src_img is None:
            continue
        if idx < 10:
            result = model.predict(src_img, **parameters)[0]
        else:
            timer.tic(stage="inference")
            result = model.predict(src_img, **parameters)[0]
            timer.hold()
        # print(result)
        cls, conf = postprocess(result)
        # result.show()  # display to screen
        # result.save(filename=rf"{img_path}_show.png")  # save to disk
        show_img = result.plot(labels=False)
        img_name = osp.basename(img_path)
        old_cls = osp.basename(osp.dirname(img_path))
        dst_path = osp.join(dst_dir, cls, str(conf) + "#" + img_name + f"#{old_cls}.png")
        os.makedirs(osp.dirname(dst_path), exist_ok=True)
        cv2_imwrite(dst_path, show_img, ".png")
        # if idx == 10:
        #     break
    print(timer.print_stats())
