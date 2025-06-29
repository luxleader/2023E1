import cv2
import numpy as np
import time
import argparse
import os
import winsound # 为实现声音提示而添加
from config import Config
from hardware import LaserTracker
from vision import preprocess_frame, create_red_mask, create_green_mask, detect_laser, detect_dual_laser
from vision.detection import find_boundary_rects
from tracking import generate_path, generate_centerline_path
from tracking.modes import green_track_red_mode, dynamic_dual_track_mode
from ui.display import show_main_control, show_help_screen
from utils.logger import setup_logger, get_logger

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='激光追踪控制系统')
    parser.add_argument('--port', help='串口设备路径')
    parser.add_argument('--baudrate', type=int, help='波特率')
    parser.add_argument('--camera', type=int, default=0, help='摄像头索引')
    parser.add_argument('--width', type=int, help='摄像头宽度')
    parser.add_argument('--height', type=int, help='摄像头高度')
    parser.add_argument('--debug', action='store_true', help='启用调试模式')
    parser.add_argument('--logdir', default='logs', help='日志目录')
    return parser.parse_args()

def auto_boundary_detection(capture):
    """自动检测边界矩形
    
    Args:
        capture: 视频捕获对象
        
    Returns:
        outer_rect, inner_rect: 检测到的外部和内部矩形
    """
    logger = get_logger()
    logger.info("正在自动寻找矩形边框... 按 'q' 确认或退出。")
    
    cv2.namedWindow("Border Detection")
    
    while True:
        ret, frame = capture.read()
        if not ret: 
            logger.error("无法读取摄像头帧")
            return None, None

        outer_rect, inner_rect = find_boundary_rects(frame)
        
        display_frame = frame.copy()
        
        if outer_rect is not None and inner_rect is not None:
            cv2.drawContours(display_frame, [outer_rect], -1, (0, 255, 0), 2)
            cv2.drawContours(display_frame, [inner_rect], -1, (255, 0, 0), 2)
            cv2.putText(display_frame, "Found! Press 'q' to confirm.", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        else:
            cv2.putText(display_frame, "Searching for 2 rectangles...", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        cv2.imshow("Border Detection", display_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            cv2.destroyWindow("Border Detection")
            return outer_rect, inner_rect

def find_borders_interactive(capture):
    """交互式边框检测，带调整功能"""
    logger = get_logger()
    logger.info("开始交互式边框检测")
    
    # 这里可以实现更复杂的交互式边框调整
    # 为简化示例，我们直接调用自动检测
    return auto_boundary_detection(capture)

def calculate_fps(prev_time):
    """计算FPS"""
    current_time = time.time()
    fps = 1 / (current_time - prev_time)
    return fps, current_time

def main():
    """主函数"""
    # 解析命令行参数
    args = parse_args()
    
    # 初始化配置
    cfg = Config()
    if args.debug:
        cfg.DEBUG_MODE = True
    
    # 设置日志
    logger = setup_logger(debug_mode=cfg.DEBUG_MODE, log_dir=args.logdir)
    logger.info("=== 激光追踪控制系统启动 ===")
    
    # 初始化硬件
    serial_port = args.port or cfg.SERIAL_PORT
    baudrate = args.baudrate or cfg.BAUDRATE
    tracker = LaserTracker(serial_port=serial_port, baudrate=baudrate)

    # 初始化摄像头
    camera_index = args.camera if args.camera is not None else cfg.CAMERA_INDEX
    logger.info(f"正在打开摄像头 {camera_index}")
    cap = cv2.VideoCapture(camera_index)
    
    if not cap.isOpened():
        logger.error("无法打开摄像头")
        tracker.close_serial()
        return
    
    # 设置摄像头参数
    width = args.width or cfg.CAMERA_WIDTH
    height = args.height or cfg.CAMERA_HEIGHT
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    
    # 验证摄像头设置
    actual_width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    actual_height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    logger.info(f"摄像头分辨率: {actual_width}x{actual_height}")

    # 初始化状态变量
    saved_positions = []
    prev_frame_time = time.time()
    fps = 0

    # 打印帮助信息
    logger.info("\n--- 激光追踪控制系统 ---")
    logger.info("'s' - 保存红光点位置")
    logger.info("'c' - 清除保存的位置")
    logger.info("'p' - 沿边界移动红色光斑 (基本要求 2,3,4)")
    logger.info("'g' - 绿色跟踪红色模式 (发挥部分 1)")
    logger.info("'t' - 组合模式: 沿路径移动并跟踪 (发挥部分 2)")
    logger.info("'r' - 红色光斑复位到中心 (基本要求 1)")
    logger.info("'b' - 重新检测边界")
    logger.info("'h' - 显示帮助")
    logger.info("'space' - 暂停/继续所有运动")
    logger.info("'q' - 退出程序")
    logger.info("------------------")
    
    # 初始化边界变量，初始时无边界
    outer_rect, inner_rect = None, None
    
    # 初始化跟踪模式
    current_mode = "manual"  # 默认手动模式
    path_points = []
    path_index = 0
    show_help = False
    is_paused = False # 用于暂停功能的状态

    # 主循环
    try:
        display_frame = np.zeros((int(actual_height), int(actual_width), 3), dtype=np.uint8)
        while True:
            # 读取一帧
            ret, frame = cap.read()
            if not ret:
                logger.error("无法读取摄像头帧")
                break
            
            # 优先处理暂停键
            key = cv2.waitKey(1) & 0xFF
            if key == ord(' '):
                is_paused = not is_paused
                if is_paused:
                    logger.info("程序已暂停")
                    tracker.stop() # 立即停止硬件运动
                else:
                    logger.info("程序已恢复")

            if is_paused:
                # 如果暂停，只显示画面，不执行任何逻辑
                cv2.putText(display_frame, "PAUSED", (int(actual_width/2)-50, int(actual_height/2)), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
                cv2.imshow("激光追踪控制系统", display_frame)
                continue

            # 如果在需要边界的模式下未检测边界，则先进行检测
            if current_mode in ["green_track_red", "dynamic_dual_track", "path_following", "combo_track"]:
                if outer_rect is None:
                    logger.info(f"进入 {current_mode} 模式, 需要先检测边界。")
                    outer_rect, inner_rect = auto_boundary_detection(cap)
                    if outer_rect is None:
                        logger.warning("未能检测到边界，无法启动该模式。切换回手动模式。")
                        current_mode = "manual"
                        continue 
                
            # 预处理帧
            processed_frame = preprocess_frame(frame)
            
            # 创建掩码
            red_mask = create_red_mask(processed_frame)
            green_mask = create_green_mask(processed_frame)
            
            # 检测激光点
            red_center = detect_laser(red_mask)
            green_center = detect_laser(green_mask)
            
            # 计算FPS
            fps, prev_frame_time = calculate_fps(prev_frame_time)
            
            # 复制原始帧用于显示，避免在原图上画图
            display_frame = frame.copy()

            # 绘制通用元素
            if outer_rect is not None:
                cv2.drawContours(display_frame, [outer_rect], -1, (0, 255, 0), 2)
            if inner_rect is not None:
                cv2.drawContours(display_frame, [inner_rect], -1, (0, 0, 255), 2)
            if red_center is not None:
                cv2.circle(display_frame, red_center, 5, (0, 0, 255), -1)
            if green_center is not None:
                cv2.circle(display_frame, green_center, 5, (0, 255, 0), -1)

            # 根据当前模式处理
            if current_mode == "manual":
                for i, pos in enumerate(saved_positions):
                    cv2.circle(display_frame, pos, 3, (255, 0, 0), -1)
                    cv2.putText(display_frame, str(i+1), (pos[0]+5, pos[1]-5), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
            
            elif current_mode == "path_following":
                if path_points and path_index < len(path_points):
                    target_point = path_points[path_index]
                    cv2.circle(display_frame, target_point, 8, (255, 255, 0), 2)
                    if red_center:
                        dx, dy = target_point[0] - red_center[0], target_point[1] - red_center[1]
                        tracker.move_red_towards(dx, dy)
                        if np.sqrt(dx**2 + dy**2) < 15: # 容忍度
                            path_index += 1
                    if path_index >= len(path_points):
                        logger.info("路径执行完成 (基本要求)")
                        current_mode = "manual"
                else:
                    current_mode = "manual"

            elif current_mode == "resetting":
                origin = (int(actual_width / 2), int(actual_height / 2))
                cv2.circle(display_frame, origin, 10, (0, 255, 255), 2)
                if red_center:
                    dx, dy = origin[0] - red_center[0], origin[1] - red_center[1]
                    if np.sqrt(dx**2 + dy**2) > 10:
                        tracker.move_red_towards(dx, dy)
                    else:
                        logger.info("红色光斑已复位 (基本要求)")
                        tracker.stop()
                        current_mode = "manual"
                else:
                    logger.warning("未检测到红色光斑，无法复位")
                    current_mode = "manual"

            elif current_mode == "green_track_red":
                _, target_reached = green_track_red_mode(display_frame, red_center, green_center, tracker, outer_rect, inner_rect)
                if target_reached:
                    winsound.Beep(1000, 100)
                    if green_center: cv2.circle(display_frame, green_center, 15, (255, 255, 255), 2)

            elif current_mode == "combo_track":
                # 1. 移动红色光斑
                if path_points and path_index < len(path_points):
                    target_point = path_points[path_index]
                    cv2.circle(display_frame, target_point, 8, (255, 255, 0), 2)
                    if red_center:
                        dx, dy = target_point[0] - red_center[0], target_point[1] - red_center[1]
                        tracker.move_red_towards(dx, dy)
                        if np.sqrt(dx**2 + dy**2) < 15:
                            path_index += 1
                else:
                    logger.info("组合模式路径执行完成")
                    current_mode = "manual"
                    tracker.stop()
                # 2. 绿色光斑跟踪
                _, target_reached = green_track_red_mode(display_frame, red_center, green_center, tracker, outer_rect, inner_rect, draw_on_frame=False)
                if target_reached:
                    winsound.Beep(1200, 100)
                    if green_center: cv2.circle(display_frame, green_center, 15, (255, 255, 255), 2)

            elif current_mode == "dynamic_dual_track":
                display_frame = dynamic_dual_track_mode(display_frame, red_center, green_center, tracker, outer_rect, inner_rect)

            # 显示FPS和模式信息
            cv2.putText(display_frame, f"FPS: {int(fps)}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            cv2.putText(display_frame, f"模式: {current_mode}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            
            if show_help: show_help_screen(display_frame)
            cv2.imshow("激光追踪控制系统", display_frame)
            
            # 处理键盘输入 (除了暂停键)
            if key != -1 and key != ord(' '):
                if key == ord('q'):
                    logger.info("用户请求退出程序")
                    break
                elif key == ord('s') and red_center:
                    saved_positions.append(red_center)
                    logger.info(f"保存位置 #{len(saved_positions)}: {red_center}")
                elif key == ord('c'):
                    saved_positions = []
                    logger.info("清除所有保存的位置")
                elif key == ord('p'):
                    if outer_rect is not None:
                        logger.info("启动路径跟踪 (基本要求)")
                        path_points = outer_rect.reshape(-1, 2).tolist()
                        path_points.append(path_points[0])
                        path_index = 0
                        current_mode = "path_following"
                    else:
                        logger.warning("请先按 'b' 检测边界")
                elif key == ord('r'):
                    logger.info("启动红色光斑复位 (基本要求)")
                    current_mode = "resetting"
                elif key == ord('t'):
                    if outer_rect is not None:
                        logger.info("启动组合跟踪模式 (发挥部分)")
                        path_points = outer_rect.reshape(-1, 2).tolist()
                        path_points.append(path_points[0])
                        path_index = 0
                        current_mode = "combo_track"
                    else:
                        logger.warning("请先按 'b' 检测边界")
                elif key == ord('g'):
                    current_mode = "green_track_red" if current_mode != "green_track_red" else "manual"
                    logger.info(f"切换到 {current_mode} 模式")
                elif key == ord('d'):
                    current_mode = "dynamic_dual_track" if current_mode != "dynamic_dual_track" else "manual"
                    logger.info(f"切换到 {current_mode} 模式")
                elif key == ord('b'):
                    logger.info("重新检测边界")
                    outer_rect, inner_rect = find_borders_interactive(cap)
                elif key == ord('h'):
                    show_help = not show_help
    
    except KeyboardInterrupt:
        logger.info("程序被用户中断")
    except Exception as e:
        logger.exception(f"程序发生错误: {e}")
    finally:
        # 释放资源
        cap.release()
        cv2.destroyAllWindows()
        tracker.close_serial()
        logger.info("=== 激光追踪控制系统已关闭 ===")

if __name__ == "__main__":
    main()