import os
import shutil
from my_utils import *


if __name__ == '__main__':
    # 汇总废石和精矿裁剪图
    # dst_dir = r"E:\Data\JLHD\第一次采集\废石_mask"
    dst_dir = r"E:\Data\JLHD\第一次采集\精矿_mask"
    os.makedirs(dst_dir, exist_ok=True)
    for idx in range(1, 9):
        # src_dir = rf"E:\Data\JLHD\第一次采集\废 ({idx})"
        src_dir = rf"E:\Data\JLHD\第一次采集\精 ({idx})"
        # img_paths = find_image_files(src_dir, '_crop.jpg')
        img_paths = find_image_files(src_dir, '_mask.jpg')
        for img_path in img_paths:
            shutil.copy2(img_path, dst_dir)
