import albumentations as A
import cv2
import numpy as np
import os
import os.path as osp
import time
import shutil
from skimage.feature import local_binary_pattern, graycomatrix, graycoprops
from my_utils import *


class OreMeshTextureExtractor:
    def __init__(self):
        """初始化矿石网状纹理提取器，专注于提取不规则网状纹路"""
        # 边缘检测参数
        self.canny_threshold1 = 50
        self.canny_threshold2 = 100
        self.sobel_ksize = 5  # Sobel算子大小

        # 纹理增强参数
        self.gabor_orientations = 8  # Gabor滤波方向数（捕捉多方向网状结构）
        self.gabor_scales = 5  # Gabor尺度数
        self.tophat_ksize = (9, 9)  # 顶帽变换核大小（突出亮纹路）
        self.blackhat_ksize = (9, 9)  # 黑帽变换核大小（突出暗纹路）

        # 形态学操作参数
        self.dilate_kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (2, 2))
        self.erode_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (1, 1))

        # 创建显示窗口
        # cv2.namedWindow("Original Ore Region", cv2.WINDOW_NORMAL)
        # cv2.namedWindow("Extracted Mesh Texture", cv2.WINDOW_NORMAL)
        # cv2.namedWindow("Enhanced Texture", cv2.WINDOW_NORMAL)

    def _preprocess_ore_region(self, image, mask):
        """预处理：应用mask提取矿石区域，消除背景干扰"""
        # 确保mask是单通道二值图像
        if len(mask.shape) == 3:
            mask_gray = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
        else:
            mask_gray = mask.copy()

        # 二值化mask（确保只有0和255）
        _, binary_mask = cv2.threshold(mask_gray, 127, 255, cv2.THRESH_BINARY)

        # 提取矿石区域（仅处理mask内的区域）
        ore_region = cv2.bitwise_and(image, image, mask=binary_mask)

        # 转换为灰度图用于纹理分析
        ore_gray = cv2.cvtColor(ore_region, cv2.COLOR_BGR2GRAY)

        # 对矿石区域进行局部对比度增强（限制在mask内）
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        ore_clahe = clahe.apply(ore_gray)

        # 去除矿石区域内的噪声（保留边缘的双边滤波）
        ore_denoised = cv2.bilateralFilter(ore_clahe, 9, 75, 75)

        return ore_gray, ore_denoised, binary_mask

    def _extract_sobel_edges(self, gray, mask):
        # 2. 多方向Sobel边缘检测
        sobel_filtered = np.zeros_like(gray)
        # 8个方向的边缘检测（每45度一个方向）
        for angle in range(0, 180, 45):
            rad = np.deg2rad(angle)
            # 创建方向滤波核
            kernel = cv2.getGaborKernel((self.sobel_ksize, self.sobel_ksize), 1.0, rad, 5.0, 0.5)
            # 滤波并提取边缘
            filtered = cv2.filter2D(gray, cv2.CV_64F, kernel)
            filtered = np.uint8(np.absolute(filtered))
            sobel_filtered = cv2.bitwise_or(sobel_filtered, filtered)

        # 仅保留mask区域
        sobel_filtered = cv2.bitwise_and(sobel_filtered, mask)

        return sobel_filtered

    def _multi_directional_edge_detection(self, gray, mask):
        """多方向边缘检测，捕捉不同角度的网状纹路"""
        # 1. Canny边缘检测（基础边缘）
        canny_edges = cv2.Canny(gray, self.canny_threshold1, self.canny_threshold2)
        # cv2.imwrite(f"canny_edges_{self.canny_threshold1}_{self.canny_threshold2}.jpg", canny_edges)
        # exit()

        # 2. 多方向Sobel边缘检测
        sobel_edges = np.zeros_like(gray)
        sobel_filtered = np.zeros_like(gray)
        # 8个方向的边缘检测（每45度一个方向）
        for angle in range(0, 180, 45):
            rad = np.deg2rad(angle)
            # 创建方向滤波核
            kernel = cv2.getGaborKernel((self.sobel_ksize, self.sobel_ksize), 1.0, rad, 5.0, 0.5)
            # 滤波并提取边缘
            filtered = cv2.filter2D(gray, cv2.CV_64F, kernel)
            filtered = np.uint8(np.absolute(filtered))
            sobel_filtered = cv2.bitwise_or(sobel_filtered, filtered)
            # cv2.imwrite(f"sobel_filtered_a{angle}_s{self.sobel_ksize}.jpg", filtered)
            # 阈值化
            _, edge = cv2.threshold(filtered, 10, 255, cv2.THRESH_BINARY)
            sobel_edges = cv2.bitwise_or(sobel_edges, edge)
        #     cv2.imwrite(f"sobel_edges_a{angle}_s{self.sobel_ksize}.jpg", edge)
        cv2.imwrite(f"sobel_filtered_all_s{self.sobel_ksize}.jpg", sobel_filtered)
        # cv2.imwrite(f"sobel_edges_all_s{self.sobel_ksize}.jpg", sobel_edges)

        # 3. 拉普拉斯边缘检测（捕捉细节变化）
        laplacian = cv2.Laplacian(gray, cv2.CV_64F, ksize=3)
        laplacian_edges = np.uint8(np.absolute(laplacian))
        cv2.imwrite("laplacian_edges1_k3.jpg", laplacian_edges)
        # _, laplacian_edges = cv2.threshold(laplacian_edges, 40, 255, cv2.THRESH_BINARY)
        # cv2.imwrite("laplacian_edges2.jpg", laplacian_edges)

        # 合并所有边缘（仅在mask区域内）
        # combined_edges = cv2.bitwise_or(canny_edges, sobel_edges)
        combined_edges = cv2.bitwise_or(sobel_edges, laplacian_edges)
        combined_edges = cv2.bitwise_and(combined_edges, mask)  # 确保只保留矿石区域

        return combined_edges

    def _gabor_texture_extraction(self, gray, mask):
        """Gabor滤波提取多尺度多方向纹理，特别适合网状结构"""
        gabor_results = []
        for scale in range(self.gabor_scales):
            sigma = 1.0 + scale * 0.5  # 不同尺度的标准差 1 + s * 0.5
            lambda_ = 3.0 + scale * 0.0  # 波长 3 + s * 1
            ksize = 9  # 9
            cur_scale_results = []
            for theta in np.linspace(0, np.pi, self.gabor_orientations, endpoint=False):
                # 创建Gabor核
                kernel = cv2.getGaborKernel(
                    (ksize, ksize), sigma, theta, lambda_, 0.5, 0, ktype=cv2.CV_32F
                )
                # 应用滤波
                filtered = cv2.filter2D(gray, cv2.CV_8UC1, kernel)
                # 阈值化提取纹理
                # _, texture = cv2.threshold(filtered, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                texture = filtered
                # 仅保留mask区域
                texture = cv2.bitwise_and(texture, mask)
                gabor_results.append(texture)
                cur_scale_results.append(texture)
                # cv2.imwrite(f"gabor_texture_scale{scale}_theta{theta}.jpg", texture)
                # cv2.imwrite(f"gabor_filtered_scale{scale}_theta{theta}.jpg", texture)
            # 融合当前scale所有Gabor结果（取或操作，保留所有方向的纹理）
            gabor_cur_combined = np.zeros_like(gray)
            for res in cur_scale_results:
                gabor_cur_combined = cv2.bitwise_or(gabor_cur_combined, res)
            cv2.imwrite(f"gabor_filtered_combined_k{ksize}_s{sigma}_l{lambda_}.jpg", gabor_cur_combined)

        # 融合所有Gabor结果（取或操作，保留所有方向的纹理）
        gabor_combined = np.zeros_like(gray)
        for res in gabor_results:
            gabor_combined = cv2.bitwise_or(gabor_combined, res)

        return gabor_combined

    def _extract_gabor_edges(self, gray, mask):
        sigma = 1.0
        lambda_ = 3.0
        ksize = 9
        cur_scale_results = []
        for theta in np.linspace(0, np.pi, self.gabor_orientations, endpoint=False):
            # 创建Gabor核
            kernel = cv2.getGaborKernel(
                (ksize, ksize), sigma, theta, lambda_, 0.5, 0, ktype=cv2.CV_32F
            )
            # 应用滤波
            filtered = cv2.filter2D(gray, cv2.CV_8UC1, kernel)
            # 记录结果
            cur_scale_results.append(filtered)
        # 融合当前scale所有Gabor结果（取或操作，保留所有方向的纹理）
        gabor_cur_combined = np.zeros_like(gray)
        for res in cur_scale_results:
            gabor_cur_combined = cv2.bitwise_or(gabor_cur_combined, res)
        # 仅保留mask区域
        gabor_cur_combined = cv2.bitwise_and(gabor_cur_combined, mask)

        return gabor_cur_combined

    def _morphological_texture_enhancement(self, texture, mask):
        """形态学操作增强纹理连通性，突出网状结构"""
        # 先腐蚀去除噪声点
        eroded = cv2.erode(texture, self.erode_kernel, iterations=1)

        # 膨胀增强纹路连通性（突出网状连接）
        dilated = cv2.dilate(eroded, self.dilate_kernel, iterations=1)

        # 顶帽变换：突出比周围亮的细小纹路
        tophat = cv2.morphologyEx(texture, cv2.MORPH_TOPHAT, self.tophat_ksize)

        # 黑帽变换：突出比周围暗的细小纹路
        blackhat = cv2.morphologyEx(texture, cv2.MORPH_BLACKHAT, self.blackhat_ksize)

        # 合并形态学结果
        morph_combined = cv2.bitwise_or(dilated, tophat)
        morph_combined = cv2.bitwise_or(morph_combined, blackhat)

        # 再次应用mask确保不超出矿石区域
        morph_combined = cv2.bitwise_and(morph_combined, mask)

        return morph_combined

    def _adaptive_threshold_texture(self, gray, mask):
        """自适应阈值提取局部纹理，处理光照不均的矿石表面"""
        # 分块自适应阈值（不同块大小捕捉不同粗细的纹路）
        thresholds = []
        for block_size in [25, 49, 81]:
            thresh = cv2.adaptiveThreshold(
                gray, 255,
                cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY,
                block_size,  # 块大小（奇数）
                3  # 常数
            )
            thresh = cv2.bitwise_and(thresh, mask)
            thresholds.append(thresh)
            cv2.imwrite(f"thresh_texture_block_size{block_size}.jpg", thresh)

        # 融合不同块大小的结果
        thresh_combined = np.zeros_like(gray)
        for th in thresholds:
            thresh_combined = cv2.bitwise_or(thresh_combined, th)

        # 仅保留矿石区域
        thresh_combined = cv2.bitwise_and(thresh_combined, mask)

        return thresh_combined

    def extract_mesh_texture(self, original_image, ore_mask):
        """
        提取矿石表面的不规则网状纹路

        参数:
            original_image: 原始彩色图像（BGR格式，OpenCV读取）
            ore_mask: 矿石区域掩码（单通道或三通道，掩码内为矿石，外为背景）

        返回:
            extracted_texture: 提取的网状纹路（二值图像）
            enhanced_visualization: 增强后的可视化图像（便于观察）
            ore_region: 提取的矿石区域（原始图像）
        """
        # 1. 预处理：提取矿石区域，消除背景干扰
        ore_gray, ore_denoised, binary_mask = self._preprocess_ore_region(original_image, ore_mask)
        ore_region = cv2.bitwise_and(original_image, original_image, mask=binary_mask)
        cv2.imwrite("ore_gray.jpg", ore_gray)
        cv2.imwrite("ore_denoised.jpg", ore_denoised)
        # cv2.imwrite("binary_mask.jpg", binary_mask)

        # 2. 多方法纹理提取
        # 2.1 多方向边缘检测
        edge_texture = self._multi_directional_edge_detection(ore_gray, binary_mask)
        cv2.imwrite("edge_texture.jpg", edge_texture)
        # exit()

        # 2.2 Gabor滤波纹理提取
        gabor_texture = self._gabor_texture_extraction(ore_gray, binary_mask)
        cv2.imwrite("gabor_filtered.jpg", gabor_texture)

        # 2.3 自适应阈值纹理提取
        thresh_texture = self._adaptive_threshold_texture(ore_gray, binary_mask)
        cv2.imwrite("thresh_texture.jpg", thresh_texture)

        # 3. 融合多种纹理提取结果
        # combined_texture = cv2.bitwise_or(edge_texture, gabor_texture)
        # cv2.imwrite("combined_edge&gabor.jpg", combined_texture)
        # combined_texture = cv2.bitwise_or(combined_texture, thresh_texture)
        # cv2.imwrite("combined_edge&gabor&thresh.jpg", combined_texture)

        # 4. 形态学增强，突出网状结构
        final_texture = self._morphological_texture_enhancement(gabor_texture, binary_mask)
        cv2.imwrite("final_texture.jpg", final_texture)

        # 5. 创建可视化增强图像（在原始矿石区域上叠加提取的纹理）
        enhanced_visualization = ore_region.copy()
        # 将提取的纹理以红色叠加到原始图像
        enhanced_visualization[final_texture > 0] = [0, 0, 127]

        return final_texture, enhanced_visualization, ore_region

    def display_results(self, original_ore, extracted_texture, enhanced):
        """显示提取结果"""
        # 显示图像
        cv2.imshow("Original Ore Region", original_ore)
        cv2.imshow("Extracted Mesh Texture", extracted_texture)
        cv2.imshow("Enhanced Texture", enhanced)

        # 等待按键操作
        print("按 's' 保存结果，按 'q' 退出")
        while True:
            key = cv2.waitKey(0)
            if key == ord('q'):
                break
            elif key == ord('s'):
                cv2.imwrite("original_ore_region.jpg", original_ore)
                cv2.imwrite("extracted_mesh_texture.jpg", extracted_texture)
                cv2.imwrite("enhanced_texture_visualization.jpg", enhanced)
                print("结果已保存")
                break

        cv2.destroyAllWindows()



if __name__ == '__main__':
    # 汇总废石和精矿裁剪图
    # dst_dir = r"E:\Data\JLHD\第一次采集\废石"
    # # dst_dir = r"E:\Data\JLHD\第一次采集\精矿_mask"
    # os.makedirs(dst_dir, exist_ok=True)
    # for idx in range(1, 9):
    #     src_dir = rf"E:\Data\JLHD\第一次采集\废 ({idx})"
    #     # src_dir = rf"E:\Data\JLHD\第一次采集\精 ({idx})"
    #     img_paths = find_image_files(src_dir, '_crop.jpg')
    #     # img_paths = find_image_files(src_dir, '_mask.jpg')
    #     for img_path in img_paths:
    #         dst_path = os.path.join(dst_dir, f"F{idx}_{os.path.basename(img_path)}")
    #         shutil.copy2(img_path, dst_path)
    # 源图像路径 and 目标输出路径
    src_dir = r"E:\PythonProject\data\ore\JLHD_limestone_all"
    mask_dir = r"E:\PythonProject\data\ore\JLHD_limestone_all_mask"
    dst_dir = r"E:\PythonProject\data\ore\JLHD_limestone_all_aug_sobel&gabor_texture"
    # 创建输出目录
    os.makedirs(dst_dir, exist_ok=True)

    # 纹理提取
    extractor = OreMeshTextureExtractor()

    # 获取所有jpg文件
    image_paths = find_image_files(src_dir, ".jpg")

    # 随机选择样本
    # sample_paths = np.random.choice(image_paths, min(5, len(image_paths)), replace=False)
    # 使用全部样本
    sample_paths = image_paths

    for img_path in sample_paths:
        src_img = cv2.imread(img_path)
        cv2.imwrite("src_img.jpg", src_img)
        img_name = osp.basename(img_path)
        mask_path = osp.join(mask_dir, img_name.replace("_crop", "_mask"))
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        # 仅提取Sobel纹理
        t0 = time.time()
        # ore_gray, _, _ = extractor._preprocess_ore_region(src_img, mask)
        ore_gray = cv2.cvtColor(src_img, cv2.COLOR_BGR2GRAY)
        sobel_texture = extractor._extract_sobel_edges(ore_gray, mask)
        print(f"仅提取Sobel纹理耗时: {(time.time() - t0) * 1000} ms")
        # show_img = np.hstack((src_img, cv2.cvtColor(sobel_texture, cv2.COLOR_GRAY2BGR)))
        # cv2.imwrite(os.path.join(dst_dir, os.path.basename(img_path)), show_img)
        # 仅提取Gabor纹理
        t0 = time.time()
        # ore_gray, _, _ = extractor._preprocess_ore_region(src_img, mask)
        ore_gray = cv2.cvtColor(src_img, cv2.COLOR_BGR2GRAY)
        gabor_texture = extractor._extract_gabor_edges(ore_gray, mask)
        print(f"仅提取Gabor纹理耗时: {(time.time() - t0) * 1000} ms")
        # show_img = np.hstack((src_img, cv2.cvtColor(gabor_texture, cv2.COLOR_GRAY2BGR)))
        # cv2.imwrite(os.path.join(dst_dir, os.path.basename(img_path)), show_img)
        # 融合sobel纹理和gabor纹理
        # 合并不同尺度的细节
        combined_texture = cv2.addWeighted(gabor_texture, 0.6, sobel_texture, 0.4, 0)
        show_img1 = np.hstack((src_img, cv2.cvtColor(combined_texture, cv2.COLOR_GRAY2BGR)))
        show_img2 = np.hstack((cv2.cvtColor(sobel_texture, cv2.COLOR_GRAY2BGR),
                               cv2.cvtColor(gabor_texture, cv2.COLOR_GRAY2BGR)))
        show_img = np.vstack((show_img1, show_img2))
        cv2.imwrite(os.path.join(dst_dir, os.path.basename(img_path)), show_img)
        # break
