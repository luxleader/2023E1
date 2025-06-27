import cv2
import numpy as np

class TrackingUI:
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
    
    def show_frame(self, frame):
        """显示帧"""
        cv2.imshow(self.window_name, frame)
    
    def close(self):
        """关闭窗口"""
        cv2.destroyWindow(self.window_name)