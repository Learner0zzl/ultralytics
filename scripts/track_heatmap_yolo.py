import cv2
import os.path as osp
from ultralytics import solutions
from my_utils import find_files_by_ext
from pathlib import Path

### 不是常规的热力图 是目标追踪的轨迹 ###
if __name__ == '__main__':
    # Initialize heatmap object
    # heatmap = solutions.Heatmap(
    #     show=False,  # display the output
    #     model=r"E:\Git\ultralytics\runs\detect\HYJTFJW_coal_det\0829_e150_i320_b8_cfg1\weights\best.pt",  # path to the YOLO11 model file
    #     colormap=cv2.COLORMAP_PARULA,  # colormap of heatmap
    #     # region=region_points,  # object counting with heatmaps, you can pass region_points
    #     # classes=[0, 2],  # generate heatmap for specific classes i.e person and car.
    # )

    # 单张测试
    # img_path = r"E:\Data\HYJTFJA\HYJTFJA_11_250722_320_27-det\tmp_label\val\9\9-0.99805_C_1_20250612_002958_1408.bmp"
    # img_path = r"E:\Data\HYJTFJA\HYJTFJA_11_250722_320_27-det\images\val_local_obj_test\4-0.83887_C_1_20250605_162209_702.bmp"
    # img_name = osp.basename(img_path)
    # img = cv2.imread(img_path)
    # results = heatmap(img)
    # print(results)
    # cv2.imwrite(f'{img_name}.png', results.plot_im)

    # 批量测试
    img_dir = r"E:\Data\HYJTFJA\HYJTFJA_11_250722_320_27-det\images\val_8_unknown"
    # img_dir = r"E:\Data\HYJTFJA\HYJTFJA_11_250722_320_27-det\images\val_local_obj_test"
    dst_dir = Path(__file__).parent / r"result/val_8_unknown"
    dst_dir.mkdir(parents=True, exist_ok=True)
    for img_path in find_files_by_ext(img_dir, '.bmp'):
        img_name = osp.basename(img_path)
        print(f"正在处理{img_name}")
        img = cv2.imread(img_path)
        heatmap = solutions.Heatmap(
            show=False,  # display the output
            model=r"E:\Git\ultralytics\runs\detect\HYJTFJW_coal_det\0829_e150_i320_b8_cfg1\weights\best.pt",
            # path to the YOLO11 model file
            colormap=cv2.COLORMAP_PARULA,  # colormap of heatmap
            # region=region_points,  # object counting with heatmaps, you can pass region_points
            # classes=[0, 2],  # generate heatmap for specific classes i.e person and car.
            conf=0.1
        )
        results = heatmap(img)
        print(results)
        cv2.imwrite(str(dst_dir / f"{img_name}.png"), results.plot_im)
