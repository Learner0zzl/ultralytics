import os
import os.path as osp
from pathlib import Path
from ultralytics import YOLO
from my_utils import find_files_by_ext, cv2_imread, cv2_imwrite, Timer, random_padding_image, yolo_det2x_anylabeling


if __name__ == '__main__':
    model_name = r"1021_e100_i320_b16"
    model = YOLO(rf"E:\Git\ultralytics\runs\classify\08_XJBTNFLJ_Gui_cls\{model_name}\weights\best.pt")
    parameters = {
        "save_txt": False,
    }
    root_dir = rf"E:\Data\TrainSet\08_XJBTNFLJ_Gui_cls\images\val"
    img_paths = find_files_by_ext(root_dir, '.bmp', mode="dict", func=lambda f: osp.splitext(f)[0])
    # dst_dir = Path(f"{root_dir}_{model_name}")
    dst_dir = Path(rf"E:\Data\TrainSet\08_XJBTNFLJ_Gui_cls\result\val_C5_{model_name}")
    dst_dir.mkdir(parents=True, exist_ok=True)
    timer = Timer()
    for idx, (name, img_path) in enumerate(img_paths.items()):
        print(f"{idx+1}/{len(img_paths)}: \nimg_path={img_path}")
        src_img = cv2_imread(img_path)
        if src_img is None:
            continue
        src = src_img
        if idx < 5:
            result = model.predict(src, **parameters)[0]
        else:
            timer.tic(stage="inference")
            result = model.predict(src, **parameters)[0]
            timer.hold()
        # print(result)

        raw_cls = Path(img_path).parent.name
        now_cls = result.probs.top1

        cls_dir = f"raw{raw_cls}_now{now_cls}"

        # 保存原图
        # dst_path = dst_dir / f"{name}.bmp"
        # cv2_imwrite(dst_path, src_img, ".bmp")

        # 渲染并保存结果图
        show_img = result.plot()
        # show_img = show_img[50: -50, 50: -50, :]
        dst_path = dst_dir / cls_dir / f"{name}_show.png"
        cv2_imwrite(dst_path, show_img, ".png")

    print(timer.print_stats())
