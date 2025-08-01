import sys
sys.path.append(r"E:\PythonProject")

import time
import numpy as np
from utils import *



if __name__ == '__main__':
    print(f"Start!")
    start = time.time()
    '''修改标注数据后缀'''
    # root_dir = r"E:\Data\JLHD\第一次采集"
    # convert_images_ext(root_dir, "bmp", "jpg")
    '''划分训练-验证集'''
    # split(r"E:\Data\JLHD\raw", '.jpg', val_ratio=0.2, batch_size=4)
    '''裁剪图像和标注'''
    # root_dir = r"E:\Data\JLHD"
    # rois = []
    # x, y, w, h = [100, 0, 4096, 4096]
    # overlap_x, overlap_y = 360, 1144
    # for idx_x in range(2):
    #     for idx_y in range(3):
    #         rois.append([x + (w - overlap_x) * idx_x, y + (h - overlap_y) * idx_y, w, h])
    # # 再随机移动生成一组
    # for idx_x in range(2):
    #     for idx_y in range(3):
    #         new_x = np.clip(x + (w - overlap_x) * idx_x + np.random.randint(-512, 512), 0, 8192 - w)
    #         new_y = np.clip(y + (h - overlap_y) * idx_y + np.random.randint(-512, 512), 0, 10000 - h)
    #         rois.append([new_x, new_y, w, h])
    # print(f"即将开始裁剪图像和标注，\n裁剪rois=\n{rois}")
    # for mode in ['train', 'val']:
    #     crop_img_and_label(src_dir=rf'{root_dir}\raw_{mode}',
    #                        dst_dir=rf'{root_dir}\images\{mode}',
    #                        rois=rois, save_no_label=True,
    #                        no_label_dont_save=[6, 7, 8, 9, 10, 11])
    '''将labelme标注转为yolo-seg格式'''
    # for set in ["JLHD"]:
    #     print(f"precess {set}")
    #     root_dir = rf"E:\Data\{set}\images"
    #     # 标准图
    #     names = ["waste", "concentrate"]
    #     labelme2yolo_seg(root_dir, names, ignore=["ok"])

    end = time.time()
    print(f"Done!\nspend {(end - start) * 1000} ms")
