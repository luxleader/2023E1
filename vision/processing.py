import cv2
import numpy as np
from config import Config

cfg = Config()

def preprocess_frame(frame):
    """图像预处理：高斯模糊和HSV转换"""
    blur = cv2.GaussianBlur(frame, (5, 5), 0)
    return cv2.cvtColor(blur, cv2.COLOR_BGR2HSV)

def create_mask(hsv, lower_bound, upper_bound):
    """创建简单颜色掩码"""
    mask = cv2.inRange(hsv, lower_bound, upper_bound)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    return cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=2)

def create_red_mask(hsv):
    """创建红色激光掩码（需处理HSV环形边界）"""
    lower_red1, upper_red1 = np.array(cfg.RED_LOWER_1), np.array(cfg.RED_UPPER_1)
    lower_red2, upper_red2 = np.array(cfg.RED_LOWER_2), np.array(cfg.RED_UPPER_2)
    
    mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
    mask = cv2.bitwise_or(mask1, mask2)
    
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    return cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=2)

def create_green_mask(hsv):
    """创建绿色激光掩码"""
    lower_green, upper_green = np.array(cfg.GREEN_LOWER), np.array(cfg.GREEN_UPPER)
    mask = cv2.inRange(hsv, lower_green, upper_green)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    return cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=2)

def create_adaptive_mask(hsv, lower_bound, upper_bound, min_area=50):
    """创建自适应阈值的掩码，处理不同光照条件"""
    # 先用正常阈值
    base_mask = cv2.inRange(hsv, np.array(lower_bound), np.array(upper_bound))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    mask = cv2.morphologyEx(base_mask, cv2.MORPH_OPEN, kernel, iterations=2)
    
    # 检查是否能检测到足够大的区域
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours or cv2.contourArea(max(contours, key=cv2.contourArea)) < min_area:
        # 调整阈值，降低饱和度和亮度要求
        adjusted_lower = np.array([lower_bound[0], max(0, lower_bound[1]-30), max(0, lower_bound[2]-30)])
        adjusted_mask = cv2.inRange(hsv, adjusted_lower, np.array(upper_bound))
        return cv2.morphologyEx(adjusted_mask, cv2.MORPH_OPEN, kernel, iterations=2)
    
    return mask