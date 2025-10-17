import os
import os.path as osp
from pathlib import Path
from ultralytics import YOLO
from my_utils import find_files_by_ext, cv2_imread, cv2_imwrite, Timer, random_padding_image, yolo_det2x_anylabeling


if __name__ == '__main__':
    model_name = r"1014_e100_i320_b16_from_tune"
    model = YOLO(rf"E:\Git\ultralytics\runs\detect\ore_det\{model_name}\weights\best.pt")
    parameters = {
        "save_txt": False,
    }
    for cls in range(11):
        root_dir = rf"E:\Data\04_FJA_coal\HYJTFJA_11_251013_320-det\0923数据审查\{cls}"
        img_paths = find_files_by_ext(root_dir, '.bmp', mode="dict", func=lambda f: osp.splitext(f)[0])
        # dst_dir = Path(f"{root_dir}_{model_name}")
        dst_dir = Path(rf"E:\Data\04_FJA_coal\HYJTFJA_11_251013_320-det\raw\{cls}")
        dst_dir.mkdir(parents=True, exist_ok=True)
        timer = Timer()
        for idx, (name, img_path) in enumerate(img_paths.items()):
            print(f"{idx+1}/{len(img_paths)}: \nimg_path={img_path}")
            src_img = cv2_imread(img_path)
            if src_img is None:
                continue
            # if "C_1_20250612_002958_1334" not in name:
            #     continue
            # res = random_padding_image(src_img, 50, 50)
            # src = res["padded_img"]
            src = src_img
            if idx < 5:
                result = model.predict(src, **parameters)[0]
            else:
                timer.tic(stage="inference")
                result = model.predict(src, **parameters)[0]
                timer.hold()
            # print(result)

            # 保存原图
            dst_path = dst_dir / f"{name}.bmp"
            cv2_imwrite(dst_path, src_img, ".bmp")

            # 渲染并保存结果图
            # show_img = result.plot()
            # show_img = show_img[50: -50, 50: -50, :]
            # dst_path = dst_dir / f"{name}_show.png"
            # cv2_imwrite(dst_path, show_img, ".png")

            # 转为x-anylabeling格式
            img_height, img_width, _ = src_img.shape
            old_json_path = img_path.replace(".bmp", ".json")
            new_json_path = dst_dir / f"{name}.json"
            if cls == 0:
                class_name = "白矸石"
            elif cls in [1, 2, 3]:
                class_name = "黑矸石"
            elif cls == 4:
                class_name = "背景"
            elif cls in [5, 6, 10]:
                class_name = str(cls)
            else:
                class_name = "煤"
            if cls in [0, 1, 4, 5, 6, 7, 10]:
                merge_json = False
            else:
                merge_json = True
            yolo_det2x_anylabeling(result.summary(), img_path, img_width, img_height, class_name=class_name,
                                   old_json_path=old_json_path, new_json_path=new_json_path, merge_json=merge_json,
                                   dx=0, dy=0)
                                   # dx=-50, dy=-50)

            # if idx > 30:
            #     break
        print(timer.print_stats())
