import cv2
import numpy as np
from ultralytics import YOLO
import torch
from typing import List, Dict, Tuple, Optional


class SlidingWindowSegmenter:
    """
    使用滑动窗口对大图像进行YOLOv11分割模型推理的类实现
    """

    def __init__(
            self,
            model_path: str,
            window_size: Tuple[int, int] = (4096, 4096),
            overlap: int = 512,
            conf_threshold: float = 0.5,
            iou_threshold: float = 0.5,
            device: Optional[str] = None
    ):
        """
        初始化滑动窗口分割器

        参数:
            model_path: YOLOv11分割模型路径
            window_size: 滑动窗口大小，默认(4096, 4096)
            overlap: 窗口重叠像素数，默认512
            conf_threshold: 置信度阈值，默认0.5
            iou_threshold: NMS的IOU阈值，默认0.5
            device: 推理设备，默认自动选择GPU或CPU
        """
        self.model_path = model_path
        self.window_size = window_size
        self.overlap = overlap
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold

        # 自动选择设备
        self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')

        # 加载模型
        self.model = self._load_model()

        # 存储结果
        self.original_image = None
        self.merged_results = None

    def _load_model(self) -> YOLO:
        """加载YOLO模型"""
        try:
            model = YOLO(self.model_path)
            model.to(self.device)
            print(f"模型已加载至 {self.device}")
            return model
        except Exception as e:
            raise RuntimeError(f"模型加载失败: {str(e)}")

    def _read_image(self, image_path: str) -> np.ndarray:
        """读取图像"""
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"无法读取图像: {image_path}")
        return image

    def process_image(self, image_path: str) -> Tuple[List[Dict], np.ndarray]:
        """
        处理图像并返回合并后的结果

        参数:
            image_path: 原始图像路径

        返回:
            merged_results: 合并后的推理结果
            original_image: 原始图像
        """
        # 读取原始图像
        self.original_image = self._read_image(image_path)
        img_height, img_width = self.original_image.shape[:2]
        window_h, window_w = self.window_size

        # 计算滑动窗口的步长
        step_h = window_h - self.overlap
        step_w = window_w - self.overlap

        # 如果图像尺寸小于窗口尺寸，调整窗口尺寸
        if img_height < window_h:
            window_h = img_height
            step_h = window_h  # 步长等于窗口高度，避免重复
        if img_width < window_w:
            window_w = img_width
            step_w = window_w  # 步长等于窗口宽度，避免重复

        # 存储所有窗口的推理结果
        all_results = []

        # 计算窗口数量
        num_windows_h = max(1, (img_height - window_h + step_h) // step_h)
        num_windows_w = max(1, (img_width - window_w + step_w) // step_w)
        print(f"图像尺寸: {img_width}x{img_height}")
        print(f"窗口尺寸: {window_w}x{window_h}, 重叠: {self.overlap}px")
        print(f"总窗口数: {num_windows_h * num_windows_w} ({num_windows_h}x{num_windows_w})")

        # 滑动窗口
        window_count = 0
        for y in range(0, img_height, step_h):
            for x in range(0, img_width, step_w):
                window_count += 1
                print(f"处理窗口 {window_count}/{num_windows_h * num_windows_w}...")

                # 计算窗口的结束坐标
                y_end = min(y + window_h, img_height)
                x_end = min(x + window_w, img_width)

                # 提取窗口图像
                window_img = self.original_image[y:y_end, x:x_end]

                # 如果窗口小于指定尺寸，进行填充
                pad_bottom = window_h - window_img.shape[0] if window_img.shape[0] < window_h else 0
                pad_right = window_w - window_img.shape[1] if window_img.shape[1] < window_w else 0

                if pad_bottom > 0 or pad_right > 0:
                    window_img_padded = cv2.copyMakeBorder(
                        window_img,
                        0, pad_bottom, 0, pad_right,
                        cv2.BORDER_CONSTANT,
                        value=[0, 0, 0]
                    )
                else:
                    window_img_padded = window_img

                # 对窗口图像进行推理
                results = self.model(
                    window_img_padded,
                    conf=self.conf_threshold,
                    device=self.device,
                    verbose=False
                )

                # 处理推理结果
                window_results = self._process_window_results(
                    results, x, y, x_end, y_end,
                    img_width, img_height,
                    pad_bottom, pad_right,
                    window_img.shape  # 传递原始窗口图像的形状
                )
                all_results.extend(window_results)

        # 应用NMS并合并结果
        self.merged_results = self._apply_nms_and_merge(all_results)
        print(f"合并后保留 {len(self.merged_results)} 个目标")

        return self.merged_results, self.original_image

    def _process_window_results(
            self,
            results,
            x: int,
            y: int,
            x_end: int,
            y_end: int,
            img_width: int,
            img_height: int,
            pad_bottom: int,
            pad_right: int,
            window_img_shape: Tuple[int, int, int]  # 添加窗口图像形状参数
    ) -> List[Dict]:
        """处理单个窗口的推理结果"""
        window_results = []

        for result in results:
            # 提取分割掩码和边界框
            if result.masks is not None and result.boxes is not None:
                masks = result.masks.data.cpu().numpy()
                boxes = result.boxes.data.cpu().numpy()  # 格式: [x1, y1, x2, y2, conf, cls]

                for i in range(len(boxes)):
                    box = boxes[i]
                    mask = masks[i]

                    # 调整坐标到原始图像
                    x1, y1, x2, y2 = box[:4]
                    x1_original = x + x1
                    y1_original = y + y1
                    x2_original = x + x2
                    y2_original = y + y2

                    # 确保坐标不超出原始图像范围
                    x1_original = max(0, min(x1_original, img_width))
                    y1_original = max(0, min(y1_original, img_height))
                    x2_original = max(0, min(x2_original, img_width))
                    y2_original = max(0, min(y2_original, img_height))

                    # 调整掩码到原始图像坐标
                    mask_original = np.zeros((img_height, img_width), dtype=np.uint8)
                    # 去除填充部分 - 使用传入的窗口图像形状
                    mask_roi = mask[:window_img_shape[0] - pad_bottom, :window_img_shape[1] - pad_right]
                    # 将掩码放到原始图像的对应位置
                    mask_original[y:y_end, x:x_end] = (mask_roi * 255).astype(np.uint8)

                    # 存储结果
                    window_results.append({
                        'box': [x1_original, y1_original, x2_original, y2_original],
                        'confidence': float(box[4]),
                        'class': int(box[5]),
                        'mask': mask_original
                    })

        return window_results

    def _apply_nms_and_merge(self, results: List[Dict]) -> List[Dict]:
        """对结果应用NMS并合并分割掩码"""
        if not results:
            return []

        # 将结果按类别分组
        class_groups = {}
        for result in results:
            cls = result['class']
            if cls not in class_groups:
                class_groups[cls] = []
            class_groups[cls].append(result)

        merged_results = []

        # 对每个类别应用NMS
        for cls, cls_results in class_groups.items():
            # 提取边界框和置信度
            boxes = np.array([r['box'] for r in cls_results], dtype=np.float32)
            confidences = np.array([r['confidence'] for r in cls_results], dtype=np.float32)

            # 应用NMS
            indices = self._nms(boxes, confidences, self.iou_threshold)

            # 保留NMS后的结果并合并掩码
            for idx in indices:
                # 找到与保留框重叠的所有框
                overlapping_indices = []
                for i, box in enumerate(boxes):
                    if self._calculate_iou(boxes[idx], box) > self.iou_threshold:
                        overlapping_indices.append(i)

                # 合并重叠的掩码（取最大值，相当于合并所有重叠区域）
                merged_mask = np.zeros_like(cls_results[0]['mask'])
                for i in overlapping_indices:
                    merged_mask = np.maximum(merged_mask, cls_results[i]['mask'])

                # 保留具有最高置信度的框作为代表
                max_conf_idx = np.argmax(confidences[overlapping_indices])
                representative_idx = overlapping_indices[max_conf_idx]

                merged_results.append({
                    'box': cls_results[representative_idx]['box'],
                    'confidence': cls_results[representative_idx]['confidence'],
                    'class': cls,
                    'mask': merged_mask
                })

        return merged_results

    def _nms(self, boxes: np.ndarray, scores: np.ndarray, iou_threshold: float) -> List[int]:
        """非极大值抑制"""
        # 如果没有边界框，返回空列表
        if len(boxes) == 0:
            return []

        # 提取坐标
        x1 = boxes[:, 0]
        y1 = boxes[:, 1]
        x2 = boxes[:, 2]
        y2 = boxes[:, 3]

        # 计算面积
        areas = (x2 - x1 + 1) * (y2 - y1 + 1)

        # 按置信度排序
        order = scores.argsort()[::-1]

        keep = []

        while order.size > 0:
            # 保留置信度最高的框
            i = order[0]
            keep.append(i)

            # 计算与其他框的IOU
            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])

            w = np.maximum(0.0, xx2 - xx1 + 1)
            h = np.maximum(0.0, yy2 - yy1 + 1)

            inter = w * h
            iou = inter / (areas[i] + areas[order[1:]] - inter)

            # 保留IOU小于阈值的框
            inds = np.where(iou <= iou_threshold)[0]
            order = order[inds + 1]

        return keep

    def _calculate_iou(self, box1: List[float], box2: List[float]) -> float:
        """计算两个边界框的IOU"""
        x1_1, y1_1, x2_1, y2_1 = box1
        x1_2, y1_2, x2_2, y2_2 = box2

        # 计算交集区域
        xx1 = max(x1_1, x1_2)
        yy1 = max(y1_1, y1_2)
        xx2 = min(x2_1, x2_2)
        yy2 = min(y2_1, y2_2)

        # 计算交集面积
        w = max(0, xx2 - xx1 + 1)
        h = max(0, yy2 - yy1 + 1)
        inter_area = w * h

        # 计算每个框的面积
        area1 = (x2_1 - x1_1 + 1) * (y2_1 - y1_1 + 1)
        area2 = (x2_2 - x1_2 + 1) * (y2_2 - y1_2 + 1)

        # 计算IOU
        iou = inter_area / float(area1 + area2 - inter_area)

        return iou

    def visualize_results(
            self,
            output_path: str,
            class_names: Optional[Dict[int, str]] = None
    ) -> None:
        """
        可视化推理结果并保存

        参数:
            output_path: 结果图像保存路径
            class_names: 类别名称映射字典
        """
        if self.original_image is None or self.merged_results is None:
            raise RuntimeError("请先调用process_image方法处理图像")

        # 创建结果图像的副本
        result_image = self.original_image.copy()

        # 为每个类别生成随机颜色
        class_colors = {}

        for result in self.merged_results:
            x1, y1, x2, y2 = map(int, result['box'])
            cls = result['class']
            confidence = result['confidence']
            mask = result['mask']

            # 为类别分配颜色
            if cls not in class_colors:
                class_colors[cls] = np.random.randint(0, 255, size=3).tolist()
            color = class_colors[cls]

            # 绘制边界框
            cv2.rectangle(result_image, (x1, y1), (x2, y2), color, 2)

            # 绘制类别和置信度
            class_name = class_names[cls] if class_names and cls in class_names else f"Class {cls}"
            label = f"{class_name}: {confidence:.2f}"
            cv2.putText(
                result_image,
                label,
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                2
            )

            # 绘制掩码（半透明）
            mask_3d = np.stack([mask] * 3, axis=-1) / 255.0
            colored_mask = (mask_3d * color).astype(np.uint8)
            result_image = cv2.addWeighted(result_image, 1, colored_mask, 0.5, 0)

        # 保存结果图像
        cv2.imwrite(output_path, result_image)
        print(f"结果已保存至: {output_path}")


# 使用示例
if __name__ == "__main__":
    # 初始化滑动窗口分割器
    segmenter = SlidingWindowSegmenter(
        model_path="yolov11seg.pt",  # 替换为你的模型路径
        window_size=(4096, 4096),
        overlap=512,
        conf_threshold=0.5,
        iou_threshold=0.5
    )

    # 处理图像
    image_path = "input_image.jpg"  # 替换为你的图像路径
    merged_results, original_image = segmenter.process_image(image_path)

    # 可视化并保存结果
    output_path = "result_image.jpg"  # 结果输出路径
    # 如果有类别名称，可以在这里定义
    # class_names = {0: "person", 1: "car", ...}
    class_names = None
    segmenter.visualize_results(output_path, class_names)
