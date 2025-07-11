import cv2
import numpy as np
import time
import argparse
import os
from config import Config
from hardware import LaserTracker
from vision import preprocess_frame, create_red_mask, create_green_mask, detect_laser, detect_dual_laser
from vision.detection import find_boundary_rects
from tracking import generate_path, generate_centerline_path
from tracking.modes import green_track_red_mode, dynamic_dual_track_mode
from k1 import linear_regression_line_mode
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
    """自动检测边界矩形 - 带调试显示"""
    logger = get_logger()
    logger.info("正在自动寻找矩形边框... 按 'q' 确认或退出，按 'd' 查看调试信息。")
    
    cv2.namedWindow("Border Detection")
    show_debug = False
    
    while True:
        ret, frame = capture.read()
        if not ret: 
            logger.error("无法读取摄像头帧")
            return None, None

        # 显示检测区域
        h, w = frame.shape[:2]
        crop_ratio = 0.7
        margin_x = int(w * (1 - crop_ratio) / 2)
        margin_y = int(h * (1 - crop_ratio) / 2)
        
        outer_rect, inner_rect = find_boundary_rects(frame)
        
        display_frame = frame.copy()
        
        # 绘制检测区域边界
        cv2.rectangle(display_frame, (margin_x, margin_y), 
                     (w-margin_x, h-margin_y), (255, 255, 0), 2)
        cv2.putText(display_frame, "Detection Area", (margin_x, margin_y-10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        
        # 调试显示
        if show_debug:
            roi = frame[margin_y:h-margin_y, margin_x:w-margin_x]
            gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            _, binary = cv2.threshold(gray, 60, 255, cv2.THRESH_BINARY_INV)
            # 将二值图像放大到与原图同样大小便于查看
            debug_img = np.zeros_like(frame)
            debug_img[margin_y:h-margin_y, margin_x:w-margin_x] = cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)
            cv2.imshow("Debug - ROI Binary", debug_img)
        
        if outer_rect is not None and inner_rect is not None:
            cv2.drawContours(display_frame, [outer_rect], -1, (0, 255, 0), 3)
            cv2.drawContours(display_frame, [inner_rect], -1, (255, 0, 0), 3)
            cv2.putText(display_frame, "Found! Press 'q' to confirm.", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        else:
            cv2.putText(display_frame, "Searching in center area... Press 'd' for debug", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        cv2.imshow("Border Detection", display_frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            if show_debug:
                cv2.destroyWindow("Debug - ROI Binary")
            cv2.destroyWindow("Border Detection")
            return outer_rect, inner_rect
        elif key == ord('d'):
            show_debug = not show_debug
            if not show_debug and cv2.getWindowProperty("Debug - ROI Binary", cv2.WND_PROP_VISIBLE) >= 0:
                cv2.destroyWindow("Debug - ROI Binary")

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
    fps = 1 / (current_time - prev_time) if current_time > prev_time else 0
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
    cap = cv2.VideoCapture(1)
    
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
    reset_point = None

    # 初始化边界变量
    outer_rect, inner_rect = None, None

    # 打印帮助信息
    logger.info("\n--- 激光追踪控制系统 ---")
    logger.info("'s' - 保存红光点位置")
    logger.info("'c' - 清除保存的位置")
    logger.info("'p' - 生成并执行路径 / (执行中)暂停或继续")
    logger.info("'r' - (执行中)复位到第5个点")
    logger.info("'g' - 绿色跟踪红色模式")
    logger.info("'d' - 动态双激光跟踪模式")
    logger.info("'k' - 线性回归巡线模式")
    logger.info("'b' - 手动启动边界检测")
    logger.info("'h' - 显示帮助")
    logger.info("'q' - 退出程序")
    logger.info("------------------")
    
    # 初始化跟踪模式
    current_mode = "manual"  # 默认手动模式
    path_points = []
    path_index = 0
    executing_path = False
    is_resetting = False
    show_help = False
    
    # 主循环
    try:
        while True:
            # 读取一帧
            ret, frame = cap.read()
            if not ret:
                logger.error("无法读取摄像头帧")
                break
                
            display_frame = frame.copy()
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
            
            # 根据当前模式处理
            if current_mode == "manual":
                # 在手动模式下，如果边界未定义，则使用默认值进行绘制
                if outer_rect is not None:
                    cv2.drawContours(display_frame, [outer_rect], -1, (0, 255, 0), 2)
                if inner_rect is not None:
                    cv2.drawContours(display_frame, [inner_rect], -1, (0, 0, 255), 2)
                
                # 显示激光点
                if red_center is not None:
                    cv2.circle(display_frame, red_center, 5, (0, 0, 255), -1)
                if green_center is not None:
                    cv2.circle(display_frame, green_center, 5, (0, 255, 0), -1)
                
                # 执行路径
                if executing_path and not tracker.paused:
                    target_point = None
                    # 如果正在复位
                    if is_resetting:
                        target_point = reset_point
                        cv2.putText(display_frame, "RESETTING", (200, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
                    # 否则，执行正常路径
                    elif path_points and path_index < len(path_points):
                        target_point = path_points[path_index]

                    if target_point:
                        cv2.circle(display_frame, target_point, 8, (255, 255, 0), 2)
                        
                        # 向硬件发送移动命令
                        if red_center is not None:
                            dx = target_point[0] - red_center[0]
                            dy = target_point[1] - red_center[1]
                            tracker.update_target_error(dx, dy) # 使用新的函数更新误差
                            
                            # 如果到达目标点附近
                            if np.sqrt(dx**2 + dy**2) < 10:
                                if is_resetting:
                                    is_resetting = False
                                    tracker.toggle_pause() # 到达复位点后自动暂停
                                    logger.info("已到达复位点，系统暂停。按 'p' 继续。")
                                else:
                                    path_index += 1
                                    if path_index >= len(path_points):
                                        executing_path = False
                                        tracker.stop_tracking_thread()
                                        logger.info("路径执行完成")
                
            elif current_mode == "green_track_red":
                # 绿色跟踪红色模式
                display_frame, target_reached = green_track_red_mode(
                    frame, red_center, green_center, tracker)
                
            elif current_mode == "dynamic_dual_track":
                # 动态双激光跟踪模式
                display_frame = dynamic_dual_track_mode(
                    frame, red_center, green_center, tracker)
            
            # 显示FPS和模式信息
            cv2.putText(display_frame, f"FPS: {int(fps)}", (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            cv2.putText(display_frame, f"模式: {current_mode}", (10, 60), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

            if tracker.paused and (executing_path or is_resetting):
                 cv2.putText(display_frame, "PAUSED", (display_frame.shape[1]//2-100, display_frame.shape[0]//2), 
                           cv2.FONT_HERSHEY_TRIPLEX, 2, (0, 255, 255), 3)

            # 显示主控制界面
            show_main_control(display_frame, saved_positions, fps=fps)
            
            # 如果需要，显示帮助屏幕
            if show_help:
                display_frame = show_help_screen(display_frame)
            
            # 显示图像
            cv2.imshow("激光追踪控制系统", display_frame)
            
            # 处理键盘输入
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q'):
                logger.info("用户请求退出程序")
                break
                
            elif key == ord('s') and red_center is not None:
                if len(saved_positions) < cfg.MAX_SAVED_POINTS:
                    saved_positions.append(red_center)
                    logger.info(f"保存位置 #{len(saved_positions)}: {red_center}")
                    if len(saved_positions) == cfg.MAX_SAVED_POINTS:
                        reset_point = saved_positions[cfg.MAX_SAVED_POINTS - 1]
                        logger.info(f"第{cfg.MAX_SAVED_POINTS}个点被设为复位点: {reset_point}")
                else:
                    logger.warning(f"已达到最大保存点数 ({cfg.MAX_SAVED_POINTS})")
                
            elif key == ord('c'):
                saved_positions = []
                reset_point = None
                executing_path = False
                tracker.stop_tracking_thread()
                logger.info("清除所有保存的位置")
                
            elif key == ord('p'):
                # 如果路径正在执行或正在复位，'p'键用于暂停/继续
                if executing_path or is_resetting:
                    tracker.toggle_pause()
                # 否则，'p'键用于开始执行路径
                elif len(saved_positions) >= 4:
                    # 使用前4个点生成路径
                    path_points = generate_path(saved_positions[:4])
                    path_index = 0
                    executing_path = True
                    is_resetting = False
                    if tracker.paused: tracker.toggle_pause() # 如果是暂停状态，则恢复
                    tracker.start_tracking_thread(mode=cfg.MODE_PID)
                    logger.info(f"生成路径并开始执行，共 {len(path_points)} 个点")
                else:
                    logger.warning("至少需要4个点来生成路径")
                    
            elif key == ord('r'):
                if executing_path and reset_point:
                    is_resetting = True
                    if tracker.paused: tracker.toggle_pause() # 如果是暂停状态，则恢复以移动到复位点
                    logger.info(f"正在复位到点: {reset_point}")
                elif not executing_path:
                    logger.warning("路径未在执行，无法复位。")
                elif not reset_point:
                    logger.warning(f"未设置复位点（需要存满{cfg.MAX_SAVED_POINTS}个点）。")
                    
            elif key == ord('g'):
                current_mode = "green_track_red" if current_mode != "green_track_red" else "manual"
                logger.info(f"切换到{current_mode}模式")
                
            elif key == ord('d'):
                current_mode = "dynamic_dual_track" if current_mode != "dynamic_dual_track" else "manual"
                logger.info(f"切换到{current_mode}模式")
                
            elif key == ord('k'):
                if current_mode != "linear_regression_line":
                    current_mode = "linear_regression_line"
                    logger.info("进入线性回归巡线模式")
                    linear_regression_line_mode(cap, tracker)
                    current_mode = "manual"  # 退出后回到手动模式
                else:
                    current_mode = "manual"
                    logger.info("退出线性回归巡线模式")
                
            elif key == ord('b'):
                logger.info("手动启动边界检测")
                outer_rect, inner_rect = find_borders_interactive(cap)
                if outer_rect is None or inner_rect is None:
                    logger.warning("未能检测到边界矩形，将使用默认值（如果需要）")
                    # 设置默认边界以供需要的功能使用
                    h, w = int(actual_height), int(actual_width)
                    outer_rect = np.array([[0, 0], [w, 0], [w, h], [0, h]], dtype=np.int32)
                    inner_offset = min(w, h) // 10
                    inner_rect = np.array([
                        [inner_offset, inner_offset], 
                        [w-inner_offset, inner_offset], 
                        [w-inner_offset, h-inner_offset], 
                        [inner_offset, h-inner_offset]
                    ], dtype=np.int32)
                
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