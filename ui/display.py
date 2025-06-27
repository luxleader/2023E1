import cv2
import numpy as np
import time
from config import Config

cfg = Config()

def show_help_screen(frame):
    """显示帮助屏幕"""
    help_frame = frame.copy()
    
    # 创建半透明背景
    overlay = help_frame.copy()
    cv2.rectangle(overlay, (0, 0), (help_frame.shape[1], help_frame.shape[0]), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.7, help_frame, 0.3, 0, help_frame)
    
    # 标题
    cv2.putText(help_frame, "激光追踪控制系统 - 帮助", (50, 50), 
               cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    
    # 命令列表
    commands = [
        "'s' - 保存红光点位置 (最多4个)",
        "'c' - 清空所有保存的点",
        "'f' - 【复位】到第一个标定点",
        "'g' - 【静態追踪】绿光追红光",
        "'b' - 【自动模式】自动识别边框并启动动态追踪",
        "'t' - 【手动模式】手动标定路径后启动动态追踪",
        "'p' - (在追踪模式中) 暂停/恢复",
        "'h' - 显示此帮助屏幕",
        "'ESC' - 退出程序"
    ]
    
    y_pos = 100
    for cmd in commands:
        cv2.putText(help_frame, cmd, (50, y_pos), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        y_pos += 40
    
    # 底部信息
    cv2.putText(help_frame, "按任意键返回", 
               (help_frame.shape[1]//2 - 100, help_frame.shape[0] - 50), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
    
    return help_frame

def show_main_control(frame, saved_positions, fps=0, debug_info=None):
    """增强版的主控制界面
    
    Args:
        frame: 原始帧
        saved_positions: 已保存的点列表
        fps: 当前FPS，可选
        debug_info: 调试信息，可选
        
    Returns:
        处理后的帧
    """
    result_frame = frame.copy()
    
    # 绘制边框和半透明背景
    overlay = result_frame.copy()
    cv2.rectangle(overlay, (5, 5), (300, 100), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.7, result_frame, 0.3, 0, result_frame)
    cv2.rectangle(result_frame, (5, 5), (300, 100), (255, 255, 255), 1)
    
    # 显示保存的点信息
    cv2.putText(result_frame, f"已保存点: {len(saved_positions)}/{cfg.MAX_SAVED_POINTS}", 
                (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    # 显示FPS
    if fps > 0:
        cv2.putText(result_frame, f"FPS: {fps:.1f}", 
                    (15, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    # 显示调试信息
    if debug_info:
        cv2.putText(result_frame, debug_info, 
                    (15, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    # 绘制保存的点和连线
    for i, pt in enumerate(saved_positions):
        cv2.circle(result_frame, pt, 8, (0, 0, 255), -1)
        cv2.putText(result_frame, str(i+1), (pt[0]+10, pt[1]), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    if len(saved_positions) == cfg.MAX_SAVED_POINTS:
        cv2.polylines(result_frame, 
                     [np.array(saved_positions, dtype=np.int32).reshape((-1, 1, 2))], 
                     True, (0, 255, 255), 2)
    
    # 绘制帮助提示
    help_text = "按 'h' 显示帮助"
    cv2.putText(result_frame, help_text, 
               (frame.shape[1] - 200, frame.shape[0] - 20), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    
    # 绘制时间戳
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    cv2.putText(result_frame, timestamp, 
               (10, frame.shape[0] - 20), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    
    return result_frame