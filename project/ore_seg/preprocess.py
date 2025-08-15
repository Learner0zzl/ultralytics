import sys

sys.path.append(r"E:\PythonProject")

import time
import numpy as np
from my_utils import *


def get_rois(add_random: bool = True):
    rois = []
    x, y, w, h = [100, 0, 4096, 4096]
    overlap_x, overlap_y = 360, 1144
    for idx_x in range(2):
        for idx_y in range(3):
            rois.append([x + (w - overlap_x) * idx_x, y + (h - overlap_y) * idx_y, w, h])
    n = len(rois)
    if add_random:
        # 再随机移动生成一组
        for idx_x in range(2):
            for idx_y in range(3):
                new_x = np.clip(x + (w - overlap_x) * idx_x + np.random.randint(-512, 512), 0, 8192 - w)
                new_y = np.clip(y + (h - overlap_y) * idx_y + np.random.randint(-512, 512), 0, 10000 - h)
                rois.append([new_x, new_y, w, h])

    return rois, list(range(n, len(rois)))




if __name__ == '__main__':
    print(f"Start!")
    start = time.time()
    '''修改标注数据后缀'''
    # root_dir = r"E:\Data\01_Ore_seg\raw"
    # convert_images_ext(root_dir, "bmp", "jpg")
    '''划分训练-验证集'''
    # split(r"E:\Data\ore_seg\raw", '.jpg', val_ratio=0.2, batch_size=4)
    '''裁剪图像和标注'''
    # root_dir = r"E:\Data\ore_seg"
    # print(f"即将开始裁剪图像和标注")
    # for mode in ['train', 'val']:
    #     crop_img_and_label_by_sliding_window(src_dir=rf'{root_dir}\raw_{mode}',
    #                                          dst_dir=rf'{root_dir}\images\{mode}',
    #                                          save_no_label=True,
    #                                          window_size=(4096, 4096),
    #                                          overlap=(512, 512),
    #                                          add_random=True)
    # 单独处理
    # crop_img_and_label_by_sliding_window(src_dir=rf'E:\Data\01_Ore_seg\tmp',
    #                                      dst_dir=rf'E:\Data\01_Ore_seg\tmp_images',
    #                                      save_no_label=True,
    #                                      window_size=(8192, 8192),
    #                                      overlap=(512, 512),
    #                                      add_random=True)
    '''将labelme标注转为yolo-seg格式'''
    # for set in ["ore_seg"]:
    #     print(f"precess {set}")
    #     root_dir = rf"E:\Data\{set}\images"
    #     names = ["waste", "concentrate", "ore"]
    #     labelme2yolo_seg(root_dir, names, ignore=["ok"])

    end = time.time()
    print(f"Done!\nspend {(end - start) * 1000} ms")
