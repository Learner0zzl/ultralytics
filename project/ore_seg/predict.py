import cv2
import numpy as np
import os
import os.path as osp
import time
import torch
from numpy import ndarray
from typing import List, Dict, Tuple, Optional
from ultralytics import YOLO
from my_utils import cv2_imread, cv2_imwrite, find_image_files


class SlidingWindowSegmenter:
    """
    滑动窗口分割器，优化了窗口边缘小比例重叠大目标的合并逻辑
    适用于大尺寸图像的目标分割，通过滑动窗口将大图分块处理，再合并跨窗口的目标
    """

    def __init__(
            self,
            model_path: str,
            window_size: Tuple[int, int] = (4096, 4096),
            model_input_size: Tuple[int, int] = (2048, 2048),
            overlap: int = 512,
            conf_threshold: float = 0.5,
            use_merging: bool = True,
            # 主要IOU阈值
            mask_iou_threshold: float = 0.5,
            box_iou_threshold: float = 0.5,
            # 窗口边缘特殊情况的阈值（更低，便于合并）
            edge_mask_iou_threshold: float = 0.2,  # 边缘情况的mask阈值
            edge_box_iou_threshold: float = 0.2,  # 边缘情况的box阈值
            window_edge_sensitivity: float = 0.15,
            # 小部分包含的判断阈值
            containment_threshold: float = 0.3,  # 一个目标被另一个包含的比例阈值
            # 过滤掉过小的目标，大概率是水滴或者碎渣
            height_thr: int = 30,
            width_thr: int = 30,
            area_thr: int = 2000,
            # 渲染参数
            class_colors: Optional[Dict[str, Tuple]] = None,
    ):
        """
        初始化滑动窗口分割器

        参数:
            model_path: YOLOv11分割模型路径
            window_size: 滑动窗口大小，默认(4096, 4096)
            model_input_size: 模型实际输入尺寸，默认(2048, 2048)
            overlap: 窗口重叠像素数，默认512
            conf_threshold: 置信度阈值，默认0.5
            use_merging: 是否使用跨窗口目标合并
            mask_iou_threshold: 正常情况下mask合并的IOU阈值，默认0.5
            box_iou_threshold: 正常情况下box合并的IOU阈值，默认0.5
            edge_mask_iou_threshold: 窗口边缘目标mask合并的IOU阈值，默认0.2
            edge_box_iou_threshold: 窗口边缘目标box合并的IOU阈值，默认0.2
            window_edge_sensitivity: 判定为窗口边缘的比例阈值，默认0.15（窗口大小的15%）
            containment_threshold: 目标包含关系的判定阈值，默认0.3
            height_thr: 高度阈值，目标是否被过滤的判定阈值，默认30
            width_thr: 宽度阈值，目标是否被过滤的判定阈值，默认30
            area_thr: 面积阈值，目标是否被过滤的判定阈值，默认2000
            class_colors: 类别颜色字典，用于渲染，默认None 随机生成
        """
        self.model_path = model_path
        self.window_size = window_size
        self.model_input_size = model_input_size
        self.overlap = overlap
        self.conf_threshold = conf_threshold
        self.use_merging = use_merging

        # IOU阈值设置
        self.mask_iou_threshold = mask_iou_threshold
        self.box_iou_threshold = box_iou_threshold
        self.edge_mask_iou_threshold = edge_mask_iou_threshold
        self.edge_box_iou_threshold = edge_box_iou_threshold

        self.window_edge_sensitivity = window_edge_sensitivity
        self.containment_threshold = containment_threshold

        # 过滤阈值设置
        self.height_thr = height_thr
        self.width_thr = width_thr
        self.area_thr = area_thr

        # 设备自动选择与模型加载
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model = self._load_model()
        self.names = self.model.names
        print(f"模型加载完成! 设备: {self.device} 检测项: {self.names}")

        # 结果存储
        self.original_image = None
        self.merged_results = None
        self.all_raw_results = []
        self.window_positions = {}  # 窗口ID到位置及邻居的映射

        # 渲染参数
        self.class_colors = class_colors if class_colors is not None else self._generate_class_colors(self.names)

    def _load_model(self) -> YOLO:
        """加载YOLO分割模型并移动到指定设备"""
        try:
            model = YOLO(self.model_path)
            model.to(self.device)
            print(f"模型已加载至 {self.device}")
            return model
        except Exception as e:
            raise RuntimeError(f"模型加载失败: {str(e)}")

    def process_image(self, src_img: np.ndarray) -> Tuple[List[Dict], np.ndarray]:
        """
        处理输入图像，执行滑动窗口分割并合并结果

        参数:
            src_img: 输入原始图像（np.ndarray格式）

        返回:
            合并后的目标列表及原始图像
        """
        self.original_image = src_img
        img_height, img_width = self.original_image.shape[:2]
        window_h, window_w = self.window_size

        # 调整窗口大小以适应图像（如果图像小于窗口尺寸）
        adjusted_window_h = min(window_h, img_height)
        adjusted_window_w = min(window_w, img_width)

        # 计算窗口步长（确保重叠区域）
        step_h = max(1, adjusted_window_h - self.overlap)
        step_w = max(1, adjusted_window_w - self.overlap)

        # 计算所需窗口数量
        num_windows_h = self._calculate_needed_windows(img_height, adjusted_window_h, step_h)
        num_windows_w = self._calculate_needed_windows(img_width, adjusted_window_w, step_w)

        print(f"图像尺寸: {img_width}x{img_height}")
        print(f"窗口尺寸: {adjusted_window_w}x{adjusted_window_h}, 重叠: {self.overlap}px")
        print(f"总窗口数: {num_windows_h * num_windows_w}")
        print(f"正常合并阈值 - Mask IOU: {self.mask_iou_threshold}, Box IOU: {self.box_iou_threshold}")
        print(f"边缘合并阈值 - Mask IOU: {self.edge_mask_iou_threshold}, Box IOU: {self.edge_box_iou_threshold}")

        # 重置结果存储
        self.all_raw_results = []
        self.window_positions = {}
        window_count = 0

        # 滑动窗口处理
        for i in range(num_windows_h):
            # 计算窗口Y坐标（最后一个窗口确保覆盖底部）
            y = i * step_h if i < num_windows_h - 1 else max(0, img_height - adjusted_window_h)
            y_end = min(y + adjusted_window_h, img_height)

            for j in range(num_windows_w):
                # 计算窗口X坐标（最后一个窗口确保覆盖右侧）
                x = j * step_w if j < num_windows_w - 1 else max(0, img_width - adjusted_window_w)
                x_end = min(x + adjusted_window_w, img_width)

                # 窗口ID与邻居关系
                window_id = f"window_{i}_{j}"
                neighbors = self._get_neighbor_windows(i, j, num_windows_h, num_windows_w)
                self.window_positions[window_id] = {
                    'x': x, 'y': y, 'x_end': x_end, 'y_end': y_end,
                    'neighbors': neighbors
                }

                window_count += 1
                print(f"处理窗口 {window_count}/{num_windows_h * num_windows_w} "
                      f"({x}:{x_end}, {y}:{y_end})")

                # 提取窗口图像并调整大小以适应模型输入
                window_img = self.original_image[y:y_end, x:x_end]
                resized_for_model = cv2.resize(
                    window_img, (self.model_input_size[1], self.model_input_size[0]),
                    interpolation=cv2.INTER_LINEAR  # 图像缩放用线性插值更高效
                )

                # 模型推理
                results = self.model(
                    resized_for_model,
                    conf=self.conf_threshold,
                    device=self.device,
                    retina_masks=True
                )

                # 处理窗口结果并转换到原始图像坐标
                window_results = self._process_window_results(
                    results, x, y, x_end, y_end,
                    img_width, img_height,
                    (x_end - x, y_end - y),
                    window_id
                )
                self.all_raw_results.extend(window_results)

        # 合并跨窗口结果
        if self.use_merging:
            self.merged_results = self._merge_cross_window_results()
            print(f"合并后保留 {len(self.merged_results)} 个目标")
        else:
            self.merged_results = self.all_raw_results
            print(f"未合并，共 {len(self.merged_results)} 个目标")

        return self.merged_results, self.original_image

    def _generate_class_colors(self, names):
        class_colors = dict()
        # 为未指定颜色的类别生成随机颜色
        for cls in names.values():
            class_colors[cls] = self._generate_random_colors()

        return class_colors

    def _generate_random_colors(self):
        return tuple(np.random.randint(0, 255, size=3))

    def _get_neighbor_windows(self, i: int, j: int, num_h: int, num_w: int) -> List[str]:
        """获取相邻窗口ID列表"""
        neighbors = [f"window_{i}_{j}"]
        if i > 0:
            neighbors.append(f"window_{i - 1}_{j}")
        if i < num_h - 1:
            neighbors.append(f"window_{i + 1}_{j}")
        if j > 0:
            neighbors.append(f"window_{i}_{j - 1}")
        if j < num_w - 1:
            neighbors.append(f"window_{i}_{j + 1}")
        return neighbors

    def _calculate_needed_windows(self, dimension_size: int, window_size: int, step: int) -> int:
        """
        计算完全覆盖一个维度所需的窗口数量

        参数:
            dimension_size: 图像维度大小（高度或宽度）
            window_size: 窗口在该维度的大小
            step: 窗口步长

        返回:
            所需窗口数量
        """
        if dimension_size <= window_size:
            return 1
        remaining = dimension_size - window_size
        return 1 + (remaining + step - 1) // step  # 向上取整

    def _process_window_results(
            self,
            results,
            x: int,
            y: int,
            x_end: int,
            y_end: int,
            img_width: int,
            img_height: int,
            window_size: Tuple[int, int],
            window_id: str
    ) -> List[Dict]:
        """
        处理单个窗口的推理结果，将坐标和掩码转换到原始图像坐标系

        参数:
            results: 模型推理结果
            x, y: 窗口在原始图像中的左上角坐标
            x_end, y_end: 窗口在原始图像中的右下角坐标
            img_width, img_height: 原始图像尺寸
            window_size: 窗口实际大小 (宽, 高)
            window_id: 窗口唯一标识

        返回:
            转换后的窗口内目标列表
        """
        window_results = []
        window_w, window_h = window_size

        # 计算缩放比例（模型输入尺寸 -> 窗口实际尺寸）
        scale_w = window_w / self.model_input_size[1]
        scale_h = window_h / self.model_input_size[0]

        for result in results:
            if result.masks is None or result.boxes is None:
                continue  # 跳过无掩码或边界框的结果

            # 提取掩码和边界框（转移到CPU并转为numpy）
            masks = result.masks.data.cpu().numpy()
            boxes = result.boxes.data.cpu().numpy()

            for i in range(len(boxes)):
                box = boxes[i]
                mask = masks[i]

                # 将边界框从模型输入尺寸缩放至窗口尺寸，再转换到原始图像坐标
                x1_original = x + box[0] * scale_w
                y1_original = y + box[1] * scale_h
                x2_original = x + box[2] * scale_w
                y2_original = y + box[3] * scale_h

                # 确保坐标在图像范围内
                x1_original = max(0, min(x1_original, img_width))
                y1_original = max(0, min(y1_original, img_height))
                x2_original = max(0, min(x2_original, img_width))
                y2_original = max(0, min(y2_original, img_height))

                # 缩放掩码至窗口尺寸并放置到原始图像坐标系
                mask_resized = cv2.resize(
                    mask, (window_w, window_h), interpolation=cv2.INTER_NEAREST  # 掩码用最近邻插值保持准确性
                ).astype(bool)  # 转为bool类型节省内存

                # 仅创建与掩码有效区域对应的数组（优化内存）
                mask_original = np.zeros((img_height, img_width), dtype=bool)
                mask_original[y:y_end, x:x_end] = mask_resized

                # 判断目标是否靠近窗口边缘
                is_near_window_edge = self._is_near_window_edge(
                    (x1_original, y1_original, x2_original, y2_original),
                    (x, y, x_end, y_end)
                )

                # 计算目标面积
                width = x2_original - x1_original
                height = y2_original - y1_original
                area = width * height

                # 如果目标过小则判为ok
                if width < self.width_thr or height < self.height_thr or area < self.area_thr:
                    cls = "ok"
                else:
                    cls = self.names[int(box[5])]

                window_results.append({
                    'box': [x1_original, y1_original, x2_original, y2_original],
                    'confidence': float(box[4]),
                    'class': cls,
                    'mask': mask_original,
                    'window_id': window_id,
                    'is_near_edge': is_near_window_edge,
                    'area': area
                })

        return window_results

    def _is_near_window_edge(self, box: Tuple[float, float, float, float], window: Tuple[int, int, int, int]) -> bool:
        """
        判断目标是否靠近窗口边缘

        参数:
            box: 目标边界框 (x1, y1, x2, y2)
            window: 窗口边界 (x, y, x_end, y_end)

        返回:
            是否靠近窗口边缘（布尔值）
        """
        x1, y1, x2, y2 = box
        win_x, win_y, win_x_end, win_y_end = window

        # 计算目标到窗口四边的距离
        dist_to_left = x1 - win_x
        dist_to_right = win_x_end - x2
        dist_to_top = y1 - win_y
        dist_to_bottom = win_y_end - y2

        # 边缘阈值为窗口最小边长的一定比例
        edge_threshold = min(win_x_end - win_x, win_y_end - win_y) * self.window_edge_sensitivity

        return (dist_to_left < edge_threshold or
                dist_to_right < edge_threshold or
                dist_to_top < edge_threshold or
                dist_to_bottom < edge_threshold)

    def _merge_cross_window_results(self) -> List[Dict]:
        """
        合并跨窗口的目标结果，优化处理边缘重叠情况

        合并逻辑:
        1. 按类别分组处理（不同类别不合并）
        2. 对每个目标尝试与已合并目标合并
        3. 特殊处理相邻窗口的边缘目标（降低IOU阈值）
        4. 处理包含关系（一个目标大部分被另一个包含时合并）
        """
        if not self.all_raw_results:
            return []

        # 按类别分组（不同类别不合并）
        class_groups = {}
        for result in self.all_raw_results:
            cls = result['class']
            if cls not in class_groups:
                class_groups[cls] = []
            class_groups[cls].append(result)

        merged_results = []

        # 对每个类别进行合并
        for cls, cls_results in class_groups.items():
            # 按面积降序排序，优先处理大目标（减少包含关系判断次数）
            cls_results_sorted = sorted(cls_results, key=lambda x: x['area'], reverse=True)
            merged = []
            for result in cls_results_sorted:
                self._try_merge_result(result, merged, cls)
            merged_results.extend(merged)

        return merged_results

    def _try_merge_result(self, result: Dict, merged: List[Dict], cls: int) -> None:
        """
        尝试将当前目标与已合并目标列表中的目标合并

        参数:
            result: 当前待合并的目标
            merged: 已合并的目标列表
            cls: 目标类别
        """
        for i in range(len(merged)):
            # 1. 检查窗口关系（非相邻窗口的目标不合并）
            if not self._are_windows_neighbors(merged[i], result):
                continue

            # 2. 快速过滤：边界框几乎无重叠的直接跳过
            box_iou = self._calculate_box_iou(result['box'], merged[i]['box'])
            if box_iou < 0.05:
                continue

            # 3. 计算掩码IOU（仅在边界框有重叠时计算）
            mask_iou = self._calculate_mask_iou(result['mask'], merged[i]['mask'])
            if mask_iou < 0.05:
                continue

            # 4. 检查包含关系（一个目标大部分被另一个包含）
            containment = self._calculate_containment(result['mask'], merged[i]['mask'])

            # 5. 确定适用的IOU阈值
            if merged[i]['is_near_edge'] or result['is_near_edge']:
                mask_threshold = self.edge_mask_iou_threshold
                box_threshold = self.edge_box_iou_threshold
            else:
                mask_threshold = self.mask_iou_threshold
                box_threshold = self.box_iou_threshold

            # 6. 判断是否需要合并
            should_merge = (mask_iou > mask_threshold and box_iou > box_threshold) or \
                           (containment > self.containment_threshold)

            if should_merge:
                # 执行合并操作
                merged_mask = np.logical_or(merged[i]['mask'], result['mask'])  # bool类型操作更高效
                max_confidence = max(merged[i]['confidence'], result['confidence'])
                merged_box = self._calculate_merged_box(merged[i]['box'], result['box'])

                # 更新合并窗口ID记录
                merged_window_ids = merged[i].get('merged_window_ids', [merged[i]['window_id']])
                merged_window_ids.append(result['window_id'])

                # 更新合并结果
                merged[i] = {
                    'box': merged_box,
                    'confidence': max_confidence,
                    'class': cls,
                    'mask': merged_mask,
                    'window_id': f"{result['window_id']}",
                    'is_near_edge': merged[i]['is_near_edge'] or result['is_near_edge'],
                    'merged_window_ids': list(set(merged_window_ids)),
                    'area': (merged_box[2] - merged_box[0]) * (merged_box[3] - merged_box[1])
                }
                return

        # 未找到可合并目标，直接添加
        merged.append(result.copy())

    def _are_windows_neighbors(self, window_id1, window_id2) -> bool:
        """判断两个窗口是否相邻"""
        flag = False
        windows1 = window_id1["merged_window_ids"] if window_id1.get("merged_window_ids") else [window_id1["window_id"]]
        windows2 = window_id2["window_id"]
        for window1 in windows1:
            if window1 in self.window_positions[windows2]["neighbors"] or windows2 in self.window_positions[window1]["neighbors"]:
                flag = True
        return flag

    def _calculate_containment(self, mask1: np.ndarray, mask2: np.ndarray) -> float:
        """
        计算两个掩码的包含度（一个掩码被另一个掩码覆盖的比例）

        参数:
            mask1: 第一个掩码（bool类型）
            mask2: 第二个掩码（bool类型）

        返回:
            最大包含比例（0-1之间）
        """
        # 计算交集和各自面积
        intersection = np.logical_and(mask1, mask2).sum()
        area1 = mask1.sum()
        area2 = mask2.sum()

        if area1 == 0 or area2 == 0:
            return 0.0

        # 返回两个方向的最大包含比例
        return max(intersection / area1, intersection / area2)

    def _calculate_mask_iou(self, mask1: np.ndarray, mask2: np.ndarray) -> float:
        """
        计算两个掩码的交并比（IOU），优化版本：仅在掩码有效区域计算

        参数:
            mask1: 第一个掩码（bool类型）
            mask2: 第二个掩码（bool类型）

        返回:
            IOU值（0-1之间）
        """
        # 找到两个掩码的有效区域边界（减少计算范围）
        rows = np.logical_or(mask1.any(axis=1), mask2.any(axis=1))
        cols = np.logical_or(mask1.any(axis=0), mask2.any(axis=0))
        if not np.any(rows) or not np.any(cols):
            return 0.0

        # 裁剪到有效区域
        rmin, rmax = np.where(rows)[0][[0, -1]]
        cmin, cmax = np.where(cols)[0][[0, -1]]
        mask1_cropped = mask1[rmin:rmax + 1, cmin:cmax + 1]
        mask2_cropped = mask2[rmin:rmax + 1, cmin:cmax + 1]

        # 计算交并集
        intersection = np.logical_and(mask1_cropped, mask2_cropped).sum()
        union = np.logical_or(mask1_cropped, mask2_cropped).sum()

        return float(intersection) / float(union) if union != 0 else 0.0

    def _calculate_box_iou(self, box1: List[float], box2: List[float]) -> float:
        """
        计算两个边界框的交并比（IOU）

        参数:
            box1: 第一个边界框 [x1, y1, x2, y2]
            box2: 第二个边界框 [x1, y1, x2, y2]

        返回:
            IOU值（0-1之间）
        """
        x1_1, y1_1, x2_1, y2_1 = box1
        x1_2, y1_2, x2_2, y2_2 = box2

        # 计算交集区域
        xx1 = max(x1_1, x1_2)
        yy1 = max(y1_1, y1_2)
        xx2 = min(x2_1, x2_2)
        yy2 = min(y2_1, y2_2)

        intersection = max(0.0, xx2 - xx1) * max(0.0, yy2 - yy1)
        area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
        area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
        union = area1 + area2 - intersection

        return float(intersection) / float(union) if union != 0 else 0.0

    def _calculate_merged_box(self, box1: List[float], box2: List[float]) -> List[float]:
        """计算两个边界框合并后的新边界框"""
        return [
            min(box1[0], box2[0]),
            min(box1[1], box2[1]),
            max(box1[2], box2[2]),
            max(box1[3], box2[3])
        ]

    def visualize_results(
            self,
            output_path: str = None,
            use_raw_results: bool = False
    ) -> None:
        """
        可视化分割结果，绘制边界框、掩码和标签

        参数:
            output_path: 结果保存路径（None则显示图像）
            use_raw_results: 是否使用原始结果（未合并）
        """
        if self.original_image is None or self.merged_results is None:
            raise RuntimeError("请先调用process_image方法处理图像")

        results_to_use = self.all_raw_results if (
                use_raw_results and self.all_raw_results) else self.merged_results
        print(f"使用{'原始' if use_raw_results else '合并后'}结果可视化，共 {len(results_to_use)} 个目标")

        result_image = self.original_image.copy()

        for idx, result in enumerate(results_to_use):
            x1, y1, x2, y2 = map(int, result['box'])
            cls: str = result['class']
            confidence = result['confidence']
            mask = result['mask']

            color = self.class_colors.get(cls)
            if color is None:
                color = self._generate_random_colors()
                self.class_colors[cls] = color

            # 绘制边界框
            cv2.rectangle(result_image, (x1, y1), (x2, y2), color, 2)

            # 绘制标签
            label = f"{cls}_{idx}: {confidence:.2f}"
            cv2.putText(
                result_image, label, (x1, max(y1 - 10, 10)),  # 避免标签超出图像
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2
            )

            # 绘制掩码（半透明叠加）
            if mask is not None:
                # 仅在掩码有效区域进行绘制（优化速度）
                mask_roi = mask[y1:y2, x1:x2]
                if np.any(mask_roi):
                    # 创建掩码区域的彩色叠加层
                    colored_mask = np.zeros((y2 - y1, x2 - x1, 3), dtype=np.uint8)
                    colored_mask[mask_roi] = color
                    # 叠加到原始图像
                    result_image[y1:y2, x1:x2] = cv2.addWeighted(
                        result_image[y1:y2, x1:x2], 1, colored_mask, 0.5, 0
                    )

        # 保存或显示结果
        if output_path:
            cv2_imwrite(output_path, result_image)
            print(f"结果已保存至: {output_path}")
        else:
            cv2.namedWindow("Result", cv2.WINDOW_NORMAL)
            cv2.imshow("Result", result_image)
            cv2.waitKey(0)
            cv2.destroyWindow("Result")

    def save_object_details(
            self,
            img_path: str,
            save_mask: bool = True,
            save_contour: bool = True,
            use_raw_results: bool = False,
            expand: int = 0
    ) -> None:
        """
        保存每个目标的详细信息（裁剪图像、掩码、轮廓）

        参数:
            img_path: 原始图像路径（用于生成保存路径）
            save_mask: 是否保存掩码
            save_contour: 是否保存轮廓
            use_raw_results: 是否使用原始结果（未合并）
            expand: 裁剪时的外扩像素数
        """
        if self.original_image is None or self.merged_results is None:
            raise RuntimeError("请先调用process_image方法处理图像")

        expand = max(0, int(expand))
        results_to_use = self.all_raw_results if (
                use_raw_results and self.all_raw_results) else self.merged_results
        print(f"保存{'原始' if use_raw_results else '合并后'}目标详情，共 {len(results_to_use)} 个目标")

        base_path = os.path.splitext(img_path)[0]
        img_height, img_width = self.original_image.shape[:2]

        for idx, result in enumerate(results_to_use):
            x1, y1, x2, y2 = map(int, result['box'])
            confidence = result['confidence']
            mask = result['mask']

            # 计算外扩后的坐标（确保在图像范围内）
            x1_expand = max(0, x1 - expand)
            y1_expand = max(0, y1 - expand)
            x2_expand = min(img_width, x2 + expand)
            y2_expand = min(img_height, y2 + expand)

            # 生成目标ID（包含置信度）
            obj_id = f"obj_{idx}_conf_{confidence:.2f}".replace('.', '_')

            # 裁剪目标区域
            object_image = self.original_image[y1_expand:y2_expand, x1_expand:x2_expand].copy()
            crop_path = f"{base_path}_{obj_id}_crop.jpg"
            cv2_imwrite(crop_path, object_image)
            print(f"已保存裁剪图像: {crop_path}")

            # 保存掩码（转换为uint8格式）
            if save_mask and mask is not None:
                object_mask = (mask[y1_expand:y2_expand, x1_expand:x2_expand] * 255).astype(np.uint8)
                mask_path = f"{base_path}_{obj_id}_mask.jpg"
                cv2.imwrite(mask_path, object_mask)
                print(f"已保存mask: {mask_path}")

            # 保存轮廓
            if save_contour and mask is not None:
                object_mask = (mask[y1_expand:y2_expand, x1_expand:x2_expand] * 255).astype(np.uint8)
                contours, _ = cv2.findContours(object_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                if contours:
                    # 取面积最大的轮廓
                    largest_contour = max(contours, key=cv2.contourArea)
                    contour_path = f"{base_path}_{obj_id}_contour.txt"
                    np.savetxt(contour_path, largest_contour.reshape((-1, 2)), fmt="%d")
                    print(f"已保存轮廓点集: {contour_path}")
                else:
                    print(f"目标 {obj_id} 未找到轮廓，跳过...")


# 使用示例
if __name__ == "__main__":
    # 初始化滑动窗口分割器
    class_colors = {
        "ore": (0, 0, 128),
        "ok": (0, 128, 0),
    }
    segmenter = SlidingWindowSegmenter(
        model_path=r"E:\Git\ultralytics\runs\segment\ore_seg\0812_e50_i2048_b4_continue_from_0812_e100_i2048_b4\weights\best.pt",  # 替换为你的模型路径
        window_size=(4096, 4096),
        model_input_size=(2048, 2048),
        overlap=512,
        conf_threshold=0.25,
        class_colors=class_colors,
    )

    # root_dir = r"E:\Data\JLHD\第一次采集"
    # root_dir = r"E:\Data\MVS采集图像\02-钼矿"
    root_dir = r"E:\Data\MVS采集图像\03-工业硅渣"
    img_paths = find_image_files(root_dir, "bmp")
    for idx, img_path in enumerate(img_paths):
        # if "Image_20250723101349379" not in img_path:
        #     continue
        print(f"Processing image {idx + 1}/{len(img_paths)}: {img_path}")
        # 读取图像
        # img_path = r"E:\Data\JLHD\第一次采集\废 (1)\Image_20250723095956704.bmp"  # 替换为你的图像路径
        src_img = cv2_imread(img_path)
        # src_img = src_img[:, :-512]
        # 处理图像
        t0 = time.time()
        merged_results, original_image = segmenter.process_image(src_img)
        print(f"处理完成，耗时: {(time.time() - t0) * 1000:.2f} ms")

        # 可视化并保存结果
        t0 = time.time()
        output_path = osp.splitext(img_path)[0] + "_show_new_conf0.25.png"  # 结果输出路径
        segmenter.visualize_results(
            output_path,
            use_raw_results=False
        )
        print(f"可视化完成，耗时: {(time.time() - t0) * 1000:.2f} ms")

        # 保存目标详情
        t0 = time.time()
        segmenter.save_object_details(
            img_path=img_path,
            save_mask=True,
            save_contour=True,
            use_raw_results=False,
            expand=20
        )
        print(f"保存目标详情完成，耗时: {(time.time() - t0) * 1000:.2f} ms")