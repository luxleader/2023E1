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
    """从单帧图像中查找最大的两个矩形边框"""
    if frame is None:
        return None, None
        
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    canny = cv2.Canny(blur, 50, 150)
    
    contours, _ = cv2.findContours(canny, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    found_rects = []
    for cnt in contours:
        if cv2.contourArea(cnt) > 1000:  # 足够大的轮廓
            peri = cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
            if len(approx) == 4:  # 四边形
                found_rects.append(approx)
    
    if len(found_rects) >= 2:
        # 按面积排序
        found_rects.sort(key=cv2.contourArea, reverse=True)
        outer_rect, inner_rect = found_rects[0], found_rects[1]
        return outer_rect, inner_rect
    
    return None, None