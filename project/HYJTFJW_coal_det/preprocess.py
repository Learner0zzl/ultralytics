import sys

sys.path.append(r"E:\PythonProject")

import time
import numpy as np
from my_utils import *

if __name__ == '__main__':
    print(f"Start!")
    start = time.time()
    '''划分训练-验证集 同时划分图像和标注'''
    split_img_and_label(img_dir = r"E:\Data\02_SXAZYH_coal\raw",
                        label_dir = r"E:\Data\02_SXAZYH_coal\raw",
                        dst_dir = r"E:\Data\02_SXAZYH_coal",
                        val_ratio=0.2, batch_size=8)

    end = time.time()
    print(f"Done!\nspend {(end - start) * 1000} ms")
