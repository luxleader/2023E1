import cv2
import numpy as np
from config import Config

cfg = Config()

def detect_laser(mask):
    """检测激光点，返回(x, y)坐标或None"""
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        max_contour = max(contours, key=cv2.contourArea)
        if cv2.contourArea(max_contour) > cfg.CONTOUR_MIN_AREA:
            M = cv2.moments(max_contour)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                return (cx, cy)
    return None

def detect_laser_with_roi(mask, prev_pos=None, roi_size=None):
    """使用ROI优化的激光点检测"""
    if roi_size is None:
        roi_size = cfg.ROI_SIZE
        
    h, w = mask.shape
    
    # 如果有上一个位置，使用ROI加速检测
    if prev_pos:
        x, y = prev_pos
        x1 = max(0, x - roi_size//2)
        y1 = max(0, y - roi_size//2)
        x2 = min(w, x + roi_size//2)
        y2 = min(h, y + roi_size//2)
        
        roi = mask[y1:y2, x1:x2]
        contours, _ = cv2.findContours(roi, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if contours:
            max_contour = max(contours, key=cv2.contourArea)
            if cv2.contourArea(max_contour) > cfg.CONTOUR_MIN_AREA:
                M = cv2.moments(max_contour)
                if M["m00"] != 0:
                    cx = int(M["m10"] / M["m00"]) + x1
                    cy = int(M["m01"] / M["m00"]) + y1
                    return (cx, cy)
    
    # 如果没有上一个位置或ROI中未检测到，则在全图搜索
    return detect_laser(mask)

def detect_dual_laser(mask_red, mask_green, prev_red=None, prev_green=None):
    """同时检测红光和绿光激光点"""
    red_pos = detect_laser_with_roi(mask_red, prev_red)
    green_pos = detect_laser_with_roi(mask_green, prev_green)
    return red_pos, green_pos

def find_boundary_rects(frame):
    """检测图像中的矩形边界 - 限制在中心区域
    
    Args:
        frame: 输入图像
        
    Returns:
        outer_rect, inner_rect: 外部和内部矩形的四个角点
    """
    h, w = frame.shape[:2]
    
    # 定义检测区域 - 只在图像中心70%的区域内检测
    crop_ratio = 0.7
    margin_x = int(w * (1 - crop_ratio) / 2)
    margin_y = int(h * (1 - crop_ratio) / 2)
    
    # 裁剪到中心区域
    roi = frame[margin_y:h-margin_y, margin_x:w-margin_x]
    
    # 转换为灰度图
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    
    # 使用多种阈值方法来检测黑色区域
    # 方法1: 简单阈值 - 针对黑色边框调整
    _, binary1 = cv2.threshold(gray, 60, 255, cv2.THRESH_BINARY_INV)
    
    # 方法2: 自适应阈值
    binary2 = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                   cv2.THRESH_BINARY_INV, 11, 2)
    
    # 合并结果
    binary = cv2.bitwise_or(binary1, binary2)
    
    # 形态学操作 - 去除噪声并连接断开的线条
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
    
    # 寻找轮廓
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    rectangles = []
    roi_area = roi.shape[0] * roi.shape[1]
    
    for contour in contours:
        # 计算轮廓面积
        area = cv2.contourArea(contour)
        
        # 面积过滤 - 相对于ROI大小
        min_area = roi_area * 0.05  # 至少5%的ROI面积
        max_area = roi_area * 0.8   # 最多80%的ROI面积
        
        if area < min_area or area > max_area:
            continue
            
        # 计算轮廓的凸包
        hull = cv2.convexHull(contour)
        
        # 多边形近似
        epsilon = 0.02 * cv2.arcLength(hull, True)
        approx = cv2.approxPolyDP(hull, epsilon, True)
        
        # 检查是否为四边形
        if len(approx) == 4:
            # 计算边长比例，确保是矩形而不是其他四边形
            points = approx.reshape(4, 2)
            
            # 计算各边长度
            sides = []
            for i in range(4):
                p1 = points[i]
                p2 = points[(i+1)%4]
                side_length = np.linalg.norm(p2 - p1)
                sides.append(side_length)
            
            # 检查对边是否近似相等（矩形特征）
            sides.sort()
            if sides[0] > 0 and sides[2] > 0:  # 避免除零
                ratio1 = sides[1] / sides[0]  # 短边比
                ratio2 = sides[3] / sides[2]  # 长边比
                
                # 对边长度应该接近（容差15%）
                if 0.85 <= ratio1 <= 1.15 and 0.85 <= ratio2 <= 1.15:
                    # 将坐标转换回原图坐标系
                    adjusted_points = []
                    for point in points:
                        adjusted_x = point[0] + margin_x
                        adjusted_y = point[1] + margin_y
                        adjusted_points.append([adjusted_x, adjusted_y])
                    
                    rectangles.append((area, np.array(adjusted_points, dtype=np.int32)))
    
    # 按面积排序，选择最大的两个
    rectangles.sort(key=lambda x: x[0], reverse=True)
    
    if len(rectangles) >= 2:
        outer_rect = rectangles[0][1]
        inner_rect = rectangles[1][1]
        return outer_rect, inner_rect
    elif len(rectangles) == 1:
        # 如果只找到一个矩形，将其作为外矩形，生成内矩形
        outer_rect = rectangles[0][1]
        
        # 生成内矩形（在外矩形内缩小25%）
        center = np.mean(outer_rect, axis=0)
        inner_rect = []
        for point in outer_rect:
            direction = point - center
            inner_point = center + direction * 0.75  # 缩小25%
            inner_rect.append(inner_point)
        inner_rect = np.array(inner_rect, dtype=np.int32)
        
        return outer_rect, inner_rect
    
    return None, None