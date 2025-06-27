import cv2
import numpy as np
from config import Config

cfg = Config()

class TrackingUI:
    """追踪系统UI组件类"""
    def __init__(self, window_name="Laser Tracking"):
        self.window_name = window_name
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
    
    def draw_path(self, frame, path, color=(255, 255, 0), thickness=2):
        """绘制路径线"""
        if path and len(path) > 0:
            cv2.polylines(frame, [np.array(path)], True, color, thickness)
        return frame
    
    def draw_laser_points(self, frame, red_pos=None, green_pos=None, 
                         red_radius=8, green_radius=8, 
                         red_color=(0, 0, 255), green_color=(0, 255, 0)):
        """绘制激光点"""
        if red_pos:
            cv2.circle(frame, red_pos, red_radius, red_color, -1)
        if green_pos:
            cv2.circle(frame, green_pos, green_radius, green_color, -1)
        return frame
    
    def draw_status(self, frame, status, position=(50, 50), 
                   font=cv2.FONT_HERSHEY_SIMPLEX, scale=1, 
                   color=(255, 255, 255), thickness=2):
        """绘制状态信息"""
        cv2.putText(frame, status, position, font, scale, color, thickness)
        return frame
    
    def draw_saved_points(self, frame, saved_positions, color=(0, 0, 255), 
                         radius=8, numbered=True):
        """绘制已保存的点"""
        for i, pt in enumerate(saved_positions):
            cv2.circle(frame, pt, radius, color, -1)
            if numbered:
                cv2.putText(frame, str(i+1), (pt[0]+10, pt[1]), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        if len(saved_positions) == cfg.MAX_SAVED_POINTS:
            cv2.polylines(frame, 
                         [np.array(saved_positions, dtype=np.int32).reshape((-1, 1, 2))], 
                         True, (0, 255, 255), 2)
        return frame
    
    def draw_overlay(self, frame, text, position=(10, 30), 
                    font=cv2.FONT_HERSHEY_SIMPLEX, scale=0.7, 
                    bg_color=(0,0,0), text_color=(255,255,255),
                    padding=5):
        """绘制带背景的文字覆盖层"""
        (text_width, text_height) = cv2.getTextSize(
            text, font, fontScale=scale, thickness=2)[0]
        
        # 绘制半透明背景
        overlay = frame.copy()
        cv2.rectangle(overlay, 
                     (position[0]-padding, position[1]-text_height-padding),
                     (position[0]+text_width+padding, position[1]+padding),
                     bg_color, -1)
        
        # 混合图层
        alpha = 0.7
        cv2.addWeighted(overlay, alpha, frame, 1-alpha, 0, frame)
        
        # 绘制文字
        cv2.putText(frame, text, position, font, scale, text_color, 2)
        return frame
    
    def show_frame(self, frame):
        """显示帧"""
        cv2.imshow(self.window_name, frame)
    
    def close(self):
        """关闭窗口"""
        cv2.destroyWindow(self.window_name)