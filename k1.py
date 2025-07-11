#!/usr/bin/env python3
"""
k1.py - 线性回归巡线模块

实现基于线性回归的巡线算法，替代最小二乘法，用于测试红色色块检测问题是否与角度计算有关。

主要功能：
1. 使用线性回归计算直线角度
2. ROI区域红色色块检测
3. 多线程巡线控制
4. 详细调试信息输出
5. 渐进式参数调试
"""

import cv2
import numpy as np
import time
import threading
from config import Config
from vision.processing import preprocess_frame, create_red_mask, create_adaptive_mask
from vision.detection import detect_laser_with_roi
from utils.logger import get_logger

cfg = Config()
logger = get_logger()

class LinearRegressionLineTracker:
    """线性回归巡线跟踪器"""
    
    def __init__(self, tracker):
        """
        初始化巡线跟踪器
        
        Args:
            tracker: LaserTracker对象，用于硬件控制
        """
        self.tracker = tracker
        self.running = False
        self.thread = None
        self.capture = None
        
        # 调试参数 - 可调节以提高检测灵敏度
        self.pixel_threshold = 30  # 降低像素阈值以提高检测灵敏度
        self.min_points = 3        # 最少需要的点数进行回归
        self.max_points = 10       # 最多保留的历史点数
        self.roi_width = 320       # ROI宽度
        self.roi_height = 240      # ROI高度
        
        # 状态变量
        self.detected_points = []  # 检测到的红色色块中心点
        self.current_angle = 0.0   # 当前计算的角度
        self.line_params = None    # 当前直线参数 (k, b)
        
        # 调试信息
        self.debug_info = {
            'points_detected': 0,
            'regression_valid': False,
            'angle_degrees': 0.0,
            'slope': 0.0,
            'intercept': 0.0
        }
        
    def calculate_line_angle_lr(self, points):
        """
        使用线性回归计算直线角度，替代最小二乘法
        
        Args:
            points: 检测到的点列表 [(x1, y1), (x2, y2), ...]
            
        Returns:
            angle: 计算得到的角度（弧度），如果计算失败返回None
            params: 直线参数 (slope, intercept)，如果计算失败返回None
        """
        if len(points) < self.min_points:
            logger.debug(f"点数不足进行线性回归: {len(points)} < {self.min_points}")
            return None, None
            
        try:
            # 提取x和y坐标
            points_array = np.array(points)
            x_coords = points_array[:, 0].astype(np.float64)
            y_coords = points_array[:, 1].astype(np.float64)
            
            # 打印检测到的点坐标
            logger.debug(f"线性回归输入点: {points}")
            
            # 检查是否为垂直线（所有x坐标相同或几乎相同）
            x_range = np.max(x_coords) - np.min(x_coords)
            if x_range < 1e-6:  # 垂直线情况
                # 对于垂直线，角度为90度（π/2弧度）
                angle_rad = np.pi / 2
                angle_deg = 90.0
                slope = float('inf')  # 无穷大斜率
                intercept = np.mean(x_coords)  # 使用平均x坐标作为"截距"
                
                logger.info(f"检测到垂直线: x={intercept:.2f}")
                logger.info(f"角度计算: {angle_deg:.2f}度 ({angle_rad:.4f}弧度)")
                
                # 更新调试信息
                self.debug_info.update({
                    'regression_valid': True,
                    'angle_degrees': angle_deg,
                    'slope': slope,
                    'intercept': intercept
                })
                
                return angle_rad, (slope, intercept)
            
            # 检查是否为水平线（所有y坐标相同或几乎相同）
            y_range = np.max(y_coords) - np.min(y_coords)
            if y_range < 1e-6:  # 水平线情况
                angle_rad = 0.0
                angle_deg = 0.0
                slope = 0.0
                intercept = np.mean(y_coords)
                
                logger.info(f"检测到水平线: y={intercept:.2f}")
                logger.info(f"角度计算: {angle_deg:.2f}度 ({angle_rad:.4f}弧度)")
                
                # 更新调试信息
                self.debug_info.update({
                    'regression_valid': True,
                    'angle_degrees': angle_deg,
                    'slope': slope,
                    'intercept': intercept
                })
                
                return angle_rad, (slope, intercept)
            
            # 执行线性回归: y = kx + b
            # 使用numpy的最小二乘法拟合
            coefficients = np.polyfit(x_coords, y_coords, 1)
            slope = coefficients[0]      # 斜率 k
            intercept = coefficients[1]  # 截距 b
            
            # 将斜率转换为角度（弧度）
            angle_rad = np.arctan(slope)
            angle_deg = np.degrees(angle_rad)
            
            # 显示回归直线参数
            logger.info(f"线性回归参数: 斜率k={slope:.4f}, 截距b={intercept:.4f}")
            logger.info(f"角度计算: {angle_deg:.2f}度 ({angle_rad:.4f}弧度)")
            
            # 更新调试信息
            self.debug_info.update({
                'regression_valid': True,
                'angle_degrees': angle_deg,
                'slope': slope,
                'intercept': intercept
            })
            
            return angle_rad, (slope, intercept)
            
        except Exception as e:
            logger.error(f"线性回归计算失败: {e}")
            self.debug_info['regression_valid'] = False
            return None, None
    
    def detect_red_blocks_in_roi(self, frame):
        """
        在ROI区域内检测红色色块，优先使用检测到的红色色块中心点
        
        Args:
            frame: 输入图像帧
            
        Returns:
            centers: 检测到的红色色块中心点列表
        """
        h, w = frame.shape[:2]
        
        # 定义ROI区域（图像中心区域）
        roi_x = max(0, (w - self.roi_width) // 2)
        roi_y = max(0, (h - self.roi_height) // 2)
        roi_x2 = min(w, roi_x + self.roi_width)
        roi_y2 = min(h, roi_y + self.roi_height)
        
        # 裁剪ROI
        roi_frame = frame[roi_y:roi_y2, roi_x:roi_x2]
        
        # 预处理
        hsv = preprocess_frame(roi_frame)
        
        # 创建自适应红色掩码 - 降低阈值以提高检测灵敏度
        mask_red1 = create_adaptive_mask(hsv, cfg.RED_LOWER_1, cfg.RED_UPPER_1, min_area=self.pixel_threshold)
        mask_red2 = create_adaptive_mask(hsv, cfg.RED_LOWER_2, cfg.RED_UPPER_2, min_area=self.pixel_threshold)
        mask_red = cv2.bitwise_or(mask_red1, mask_red2)
        
        # 检测轮廓
        contours, _ = cv2.findContours(mask_red, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        centers = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > self.pixel_threshold:  # 使用降低的像素阈值
                # 计算轮廓中心
                M = cv2.moments(contour)
                if M["m00"] != 0:
                    cx = int(M["m10"] / M["m00"]) + roi_x  # 转换回原图坐标
                    cy = int(M["m01"] / M["m00"]) + roi_y
                    centers.append((cx, cy))
                    logger.debug(f"检测到红色色块: 中心({cx}, {cy}), 面积{area:.1f}")
        
        # 添加色块检测状态的详细日志
        logger.debug(f"ROI区域 ({roi_x}, {roi_y}) 到 ({roi_x2}, {roi_y2})")
        logger.debug(f"检测到 {len(centers)} 个红色色块")
        
        self.debug_info['points_detected'] = len(centers)
        
        return centers
    
    def calculate_and_send_control_commands(self, angle, line_params):
        """
        根据线性回归结果计算控制命令并发送给硬件
        
        Args:
            angle: 直线角度（弧度）
            line_params: 直线参数 (slope, intercept)
        """
        if not self.tracker or self.tracker.paused:
            return
            
        try:
            slope, intercept = line_params
            
            # 计算角度偏差 - 目标是保持直线水平（角度为0）
            angle_error = angle  # 直接使用角度作为误差
            
            # 角度转换为像素误差
            # 这里使用简单的比例控制
            # 可以根据实际情况调整比例系数
            angle_to_pixel_ratio = 100.0  # 每弧度对应的像素数
            
            # 计算X和Y方向的误差
            # X误差：角度偏差导致的横向偏移
            error_x = int(angle_error * angle_to_pixel_ratio)
            
            # Y误差：可以基于直线与图像中心的偏移计算
            # 假设图像中心为基准点
            image_center_x = self.roi_width // 2
            line_center_y = slope * image_center_x + intercept if slope != float('inf') else intercept
            image_center_y = self.roi_height // 2
            error_y = int(line_center_y - image_center_y)
            
            # 限制误差范围，避免过大的控制信号
            max_error = 200
            error_x = max(-max_error, min(max_error, error_x))
            error_y = max(-max_error, min(max_error, error_y))
            
            # 发送控制命令
            self.tracker.update_target_error(error_x, error_y)
            
            logger.debug(f"控制命令: error_x={error_x}, error_y={error_y}, angle_rad={angle:.4f}")
            
        except Exception as e:
            logger.error(f"计算控制命令失败: {e}")
    
    def update_detected_points(self, new_points):
        """
        更新检测到的点列表，保持最新的点
        
        Args:
            new_points: 新检测到的点列表
        """
        # 添加新点到历史列表
        self.detected_points.extend(new_points)
        
        # 保持最大点数限制
        if len(self.detected_points) > self.max_points:
            self.detected_points = self.detected_points[-self.max_points:]
        
        logger.debug(f"当前历史点数: {len(self.detected_points)}")
    
    def draw_debug_overlay(self, frame):
        """
        在图像上绘制调试信息
        
        Args:
            frame: 要绘制的图像帧
            
        Returns:
            frame: 绘制了调试信息的图像帧
        """
        overlay_frame = frame.copy()
        
        # 绘制ROI区域
        h, w = frame.shape[:2]
        roi_x = max(0, (w - self.roi_width) // 2)
        roi_y = max(0, (h - self.roi_height) // 2)
        roi_x2 = min(w, roi_x + self.roi_width)
        roi_y2 = min(h, roi_y + self.roi_height)
        cv2.rectangle(overlay_frame, (roi_x, roi_y), (roi_x2, roi_y2), (255, 255, 0), 2)
        cv2.putText(overlay_frame, "ROI", (roi_x, roi_y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        
        # 绘制检测到的点
        for i, point in enumerate(self.detected_points):
            color = (0, 255, 0) if i == len(self.detected_points) - 1 else (0, 0, 255)
            cv2.circle(overlay_frame, point, 5, color, -1)
            cv2.putText(overlay_frame, str(i), (point[0] + 10, point[1]), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
        
        # 绘制回归直线
        if self.line_params is not None and len(self.detected_points) >= 2:
            slope, intercept = self.line_params
            
            # 处理垂直线情况
            if slope == float('inf') or abs(slope) > 1000:
                # 垂直线：在x=intercept处绘制垂直线
                x_line = int(intercept)
                if roi_x <= x_line <= roi_x2:
                    cv2.line(overlay_frame, (x_line, roi_y), (x_line, roi_y2), (255, 0, 255), 2)
            else:
                # 普通直线：在ROI范围内绘制直线
                x1, x2 = roi_x, roi_x2
                y1 = int(slope * x1 + intercept)
                y2 = int(slope * x2 + intercept)
                
                # 确保y坐标在图像范围内
                if y1 < 0:
                    x1 = int((0 - intercept) / slope) if slope != 0 else x1
                    y1 = 0
                elif y1 >= h:
                    x1 = int((h - 1 - intercept) / slope) if slope != 0 else x1
                    y1 = h - 1
                    
                if y2 < 0:
                    x2 = int((0 - intercept) / slope) if slope != 0 else x2
                    y2 = 0
                elif y2 >= h:
                    x2 = int((h - 1 - intercept) / slope) if slope != 0 else x2
                    y2 = h - 1
                
                # 绘制直线
                if 0 <= x1 < w and 0 <= x2 < w and 0 <= y1 < h and 0 <= y2 < h:
                    cv2.line(overlay_frame, (x1, y1), (x2, y2), (255, 0, 255), 2)
        
        # 显示调试信息文本
        info_y = 30
        cv2.putText(overlay_frame, f"Points: {self.debug_info['points_detected']}", 
                   (10, info_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        info_y += 25
        cv2.putText(overlay_frame, f"Regression: {'OK' if self.debug_info['regression_valid'] else 'FAIL'}", 
                   (10, info_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        info_y += 25
        cv2.putText(overlay_frame, f"Angle: {self.debug_info['angle_degrees']:.1f}deg", 
                   (10, info_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        info_y += 25
        slope_text = f"∞" if self.debug_info['slope'] == float('inf') else f"{self.debug_info['slope']:.3f}"
        cv2.putText(overlay_frame, f"Slope: {slope_text}", 
                   (10, info_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        info_y += 25
        
        # 显示参数调试信息
        cv2.putText(overlay_frame, f"Threshold: {self.pixel_threshold}", 
                   (10, info_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        info_y += 20
        cv2.putText(overlay_frame, f"ROI: {self.roi_width}x{self.roi_height}", 
                   (10, info_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        info_y += 20
        cv2.putText(overlay_frame, f"Min Points: {self.min_points}", 
                   (10, info_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        return overlay_frame
    
    def line_tracking_thread(self):
        """
        巡线线程函数 - 持续检测红色色块并计算角度
        """
        logger.info("=== 线性回归巡线线程启动 ===")
        
        while self.running:
            try:
                if self.capture is None:
                    time.sleep(0.1)
                    continue
                    
                # 读取图像帧
                ret, frame = self.capture.read()
                if not ret:
                    logger.warning("无法读取摄像头帧")
                    time.sleep(0.1)
                    continue
                
                # 检测红色色块
                red_centers = self.detect_red_blocks_in_roi(frame)
                
                # 更新点列表
                if red_centers:
                    self.update_detected_points(red_centers)
                
                # 执行线性回归计算角度
                if len(self.detected_points) >= self.min_points:
                    angle, params = self.calculate_line_angle_lr(self.detected_points)
                    
                    if angle is not None and params is not None:
                        self.current_angle = angle
                        self.line_params = params
                        
                        # UART通信控制逻辑 - 根据角度计算控制命令
                        self.calculate_and_send_control_commands(angle, params)
                        
                        logger.debug(f"巡线角度更新: {np.degrees(angle):.2f}度")
                    else:
                        logger.debug("线性回归计算失败")
                else:
                    logger.debug(f"等待更多点进行回归: {len(self.detected_points)}/{self.min_points}")
                
                # 控制循环频率
                time.sleep(0.05)  # 20Hz
                
            except Exception as e:
                logger.error(f"巡线线程发生错误: {e}")
                time.sleep(0.1)
        
        logger.info("=== 线性回归巡线线程结束 ===")
    
    def start_line_tracking(self, capture):
        """
        启动巡线跟踪
        
        Args:
            capture: OpenCV视频捕获对象
        """
        if self.running:
            logger.warning("巡线跟踪已在运行")
            return
            
        self.capture = capture
        self.running = True
        
        # 启动硬件控制线程
        if self.tracker:
            self.tracker.start_tracking_thread(mode=self.tracker.cfg.MODE_PID)
        
        self.thread = threading.Thread(target=self.line_tracking_thread)
        self.thread.daemon = True
        self.thread.start()
        logger.info("巡线跟踪已启动")
    
    def stop_line_tracking(self):
        """停止巡线跟踪"""
        if not self.running:
            return
            
        self.running = False
        if self.thread:
            self.thread.join(timeout=2.0)
        self.capture = None
        
        # 停止硬件控制线程
        if self.tracker:
            self.tracker.stop_tracking_thread()
            # 发送零误差信号停止移动
            self.tracker.update_target_error(0, 0)
            
        logger.info("巡线跟踪已停止")
    
    def reset_tracking_state(self):
        """重置跟踪状态"""
        self.detected_points.clear()
        self.current_angle = 0.0
        self.line_params = None
        self.debug_info = {
            'points_detected': 0,
            'regression_valid': False,
            'angle_degrees': 0.0,
            'slope': 0.0,
            'intercept': 0.0
        }
        logger.info("巡线跟踪状态已重置")
    
    def adjust_parameters(self, **kwargs):
        """
        调整检测参数 - 用于渐进式参数调试
        
        支持的参数:
            pixel_threshold: 像素阈值
            min_points: 最少回归点数
            max_points: 最多历史点数
            roi_width: ROI宽度
            roi_height: ROI高度
        """
        for param, value in kwargs.items():
            if hasattr(self, param):
                old_value = getattr(self, param)
                setattr(self, param, value)
                logger.info(f"参数调整: {param} {old_value} -> {value}")
            else:
                logger.warning(f"未知参数: {param}")


def linear_regression_line_mode(capture, tracker):
    """
    线性回归巡线模式主函数
    
    Args:
        capture: OpenCV视频捕获对象
        tracker: LaserTracker对象
    """
    logger.info("\n=== 进入线性回归巡线模式 ===")
    logger.info("按键说明:")
    logger.info("  'q' - 退出巡线模式")
    logger.info("  'p' - 暂停/继续")
    logger.info("  'r' - 重置跟踪状态")
    logger.info("  '1' - 降低像素阈值(提高检测灵敏度)")
    logger.info("  '2' - 提高像素阈值(减少噪声)")
    logger.info("  '3' - 增大ROI区域")
    logger.info("  '4' - 减小ROI区域")
    logger.info("  '5' - 增加最少回归点数")
    logger.info("  '6' - 减少最少回归点数")
    logger.info("  '7' - 增加最大历史点数")
    logger.info("  '8' - 减少最大历史点数")
    logger.info("==================================\n")
    
    # 创建线性回归巡线跟踪器
    line_tracker = LinearRegressionLineTracker(tracker)
    
    # 启动巡线跟踪线程
    line_tracker.start_line_tracking(capture)
    
    try:
        while True:
            ret, frame = capture.read()
            if not ret:
                logger.error("无法读取摄像头帧")
                break
            
            # 绘制调试覆盖层
            display_frame = line_tracker.draw_debug_overlay(frame)
            
            # 显示状态信息
            if line_tracker.running:
                cv2.putText(display_frame, "Linear Regression Line Tracking", (10, frame.shape[0] - 60), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(display_frame, f"Current Angle: {line_tracker.current_angle:.3f} rad", 
                           (10, frame.shape[0] - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            
            # 如果跟踪暂停，显示提示
            if tracker.paused:
                cv2.putText(display_frame, "PAUSED", 
                           (display_frame.shape[1]//2-100, display_frame.shape[0]//2), 
                           cv2.FONT_HERSHEY_TRIPLEX, 2, (0, 255, 255), 3)
            
            # 显示图像
            cv2.imshow('Linear Regression Line Tracking', display_frame)
            
            # 处理键盘输入
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                logger.info("用户请求退出巡线模式")
                break
            elif key == ord('p'):
                tracker.toggle_pause()
                logger.info(f"巡线模式 {'暂停' if tracker.paused else '继续'}")
            elif key == ord('r'):
                line_tracker.reset_tracking_state()
                logger.info("重置巡线跟踪状态")
            elif key == ord('1'):
                # 调整像素阈值 - 降低以提高检测灵敏度
                line_tracker.adjust_parameters(pixel_threshold=20)
            elif key == ord('2'):
                # 调整像素阈值 - 提高以减少噪声
                line_tracker.adjust_parameters(pixel_threshold=40)
            elif key == ord('3'):
                # 调整ROI大小 - 增大
                line_tracker.adjust_parameters(roi_width=400, roi_height=300)
            elif key == ord('4'):
                # 调整ROI大小 - 减小
                line_tracker.adjust_parameters(roi_width=240, roi_height=180)
            elif key == ord('5'):
                # 调整最少回归点数
                line_tracker.adjust_parameters(min_points=5)
            elif key == ord('6'):
                # 调整最少回归点数
                line_tracker.adjust_parameters(min_points=3)
            elif key == ord('7'):
                # 调整最大历史点数
                line_tracker.adjust_parameters(max_points=15)
            elif key == ord('8'):
                # 调整最大历史点数
                line_tracker.adjust_parameters(max_points=8)
                
    except KeyboardInterrupt:
        logger.info("巡线模式被用户中断")
    except Exception as e:
        logger.exception(f"巡线模式发生错误: {e}")
    finally:
        # 清理资源
        line_tracker.stop_line_tracking()
        cv2.destroyAllWindows()
        logger.info("=== 退出线性回归巡线模式 ===")


if __name__ == "__main__":
    """测试模块"""
    print("k1.py - 线性回归巡线模块")
    print("这是一个模块文件，请通过main.py或其他主程序调用")