import cv2
import numpy as np
import time
from config import Config
from vision.processing import preprocess_frame, create_red_mask, create_green_mask, create_adaptive_mask
from vision.detection import detect_dual_laser
from vision.filters import LaserKalmanFilter
from utils.logger import get_logger

cfg = Config()
logger = get_logger()

def green_track_red_mode(capture, tracker):
    """绿光追踪静止的红光模式
    
    Args:
        capture: OpenCV视频捕获对象
        tracker: LaserTracker对象
    """
    success_distance_px = cfg.SUCCESS_DISTANCE_CM * cfg.PIXELS_PER_CM
    time_limit_s = cfg.INITIAL_SUCCESS_TIME_LIMIT_S
    
    # 初始化卡尔曼滤波器
    red_filter = LaserKalmanFilter()
    green_filter = LaserKalmanFilter()
    
    logger.info(f"\n--- 进入绿光追踪红光模式 (目标: {time_limit_s}s内, 距离 <= {success_distance_px:.1f}px) ---")
    
    # 启动双激光控制线程
    tracker.start_tracking_thread(mode=cfg.MODE_DUAL_CONTROL)
    
    start_time = time.time()
    tracking_successful, success_beeped = False, False
    prev_red_pos, prev_green_pos = None, None

    try:
        while True:
            ret, frame = capture.read()
            if not ret: break

            hsv = preprocess_frame(frame)
            
            # 创建自适应掩码
            mask_red1 = create_adaptive_mask(hsv, cfg.RED_LOWER_1, cfg.RED_UPPER_1)
            mask_red2 = create_adaptive_mask(hsv, cfg.RED_LOWER_2, cfg.RED_UPPER_2)
            mask_red = cv2.bitwise_or(mask_red1, mask_red2)
            mask_green = create_adaptive_mask(hsv, cfg.GREEN_LOWER, cfg.GREEN_UPPER)
            
            # 检测激光点
            raw_red_pos, raw_green_pos = detect_dual_laser(mask_red, mask_green, prev_red_pos, prev_green_pos)
            
            # 应用卡尔曼滤波
            red_pos = red_filter.update(raw_red_pos)
            green_pos = green_filter.update(raw_green_pos)
            
            # 更新上一个位置
            if red_pos: prev_red_pos = red_pos
            if green_pos: prev_green_pos = green_pos
            
            elapsed_time = time.time() - start_time

            if not tracking_successful and elapsed_time > time_limit_s:
                logger.info(f"追踪失败：超时({time_limit_s:.1f}秒)！")
                break

            # 控制与判断逻辑
            if not tracker.paused and red_pos and green_pos:
                green_error_x = red_pos[0] - green_pos[0]
                green_error_y = red_pos[1] - green_pos[1]
                tracker.update_dual_error(0, 0, green_error_x, green_error_y)
                
                # 计算激光点间距离
                distance = np.linalg.norm(np.array(red_pos) - np.array(green_pos))
                
                # 判断是否达到目标
                if distance <= success_distance_px:
                    tracking_successful = True
                    if not success_beeped:
                        logger.info(f"追踪成功！用时: {elapsed_time:.2f}s, 距离: {distance:.1f}px")
                        print('\a')
                        success_beeped = True
            
            # 视觉反馈
            result_frame = frame.copy()
            if red_pos: cv2.circle(result_frame, red_pos, 8, (0, 0, 255), -1)
            if green_pos: cv2.circle(result_frame, green_pos, 8, (0, 255, 0), -1 if tracking_successful else 2)
            
            if tracking_successful: 
                cv2.putText(result_frame, "SUCCESS!", (50, 100), cv2.FONT_HERSHEY_TRIPLEX, 1.5, (0, 255, 0), 3)
            
            if tracker.paused: 
                cv2.putText(result_frame, "PAUSED", (result_frame.shape[1]//2-100, result_frame.shape[0]//2), 
                           cv2.FONT_HERSHEY_TRIPLEX, 2, (0, 255, 255), 3)

            cv2.imshow('Green Track Red', result_frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'): break
            elif key == ord('p'): tracker.toggle_pause()
    finally:
        tracker.stop_tracking_thread()
        tracker.update_dual_error(0, 0, 0, 0)
        cv2.destroyAllWindows()
        logger.info("--- 退出绿光追踪红光模式 ---")

def dynamic_dual_track_mode(capture, path_to_track, tracker):
    """动态双激光追踪模式，红激光按路径移动，绿激光追踪
    
    Args:
        capture: OpenCV视频捕获对象
        path_to_track: 路径点列表
        tracker: LaserTracker对象
    """
    if not path_to_track or len(path_to_track) < 2:
        logger.error("路径点数量不足，无法启动追踪")
        return
    
    success_distance_px = cfg.SUCCESS_DISTANCE_CM * cfg.PIXELS_PER_CM
    initial_success_time_limit_s = cfg.INITIAL_SUCCESS_TIME_LIMIT_S
    continuous_failure_time_limit_s = cfg.CONTINUOUS_FAILURE_TIME_LIMIT_S

    # 初始化卡尔曼滤波器
    red_filter = LaserKalmanFilter()
    green_filter = LaserKalmanFilter()

    logger.info(f"\n--- 进入动态双激光追踪模式 ---")
    logger.info(f"路径点数量: {len(path_to_track)}")
    logger.info(f"成功距离阈值: {success_distance_px:.1f}px")
    logger.info(f"初始成功时限: {initial_success_time_limit_s:.1f}秒")
    logger.info(f"连续失败时限: {continuous_failure_time_limit_s:.1f}秒")
    
    # 启动双激光控制线程
    tracker.start_tracking_thread(mode=cfg.MODE_DUAL_CONTROL)
    
    start_time, target_idx = time.time(), 0
    initial_success, is_failing, failure_start, task_failed = False, False, None, False
    prev_red_pos, prev_green_pos = None, None
    
    # 记录历史位置用于稳定性检测
    red_history, green_history = [], []
    history_max_len = 10

    try:
        while not task_failed:
            ret, frame = capture.read()
            if not ret: break

            hsv = preprocess_frame(frame)
            
            # 创建自适应掩码
            mask_red1 = create_adaptive_mask(hsv, cfg.RED_LOWER_1, cfg.RED_UPPER_1)
            mask_red2 = create_adaptive_mask(hsv, cfg.RED_LOWER_2, cfg.RED_UPPER_2)
            mask_red = cv2.bitwise_or(mask_red1, mask_red2)
            mask_green = create_adaptive_mask(hsv, cfg.GREEN_LOWER, cfg.GREEN_UPPER)
            
            # 使用ROI加速检测
            raw_red_pos, raw_green_pos = detect_dual_laser(mask_red, mask_green, prev_red_pos, prev_green_pos)
            
            # 应用卡尔曼滤波
            red_pos = red_filter.update(raw_red_pos)
            green_pos = green_filter.update(raw_green_pos)

            elapsed_time = time.time() - start_time
            
            # 更新历史位置
            if red_pos:
                red_history.append(red_pos)
                if len(red_history) > history_max_len:
                    red_history.pop(0)
                prev_red_pos = red_pos
                
            if green_pos:
                green_history.append(green_pos)
                if len(green_history) > history_max_len:
                    green_history.pop(0)
                prev_green_pos = green_pos

            # 追踪逻辑
            if not tracker.paused and red_pos and green_pos:
                # 获取当前目标点
                target = path_to_track[target_idx]
                
                # 计算红光到目标点的误差
                red_error_x, red_error_y = target[0] - red_pos[0], target[1] - red_pos[1]
                
                # 计算绿光到红光的误差
                green_error_x, green_error_y = red_pos[0] - green_pos[0], red_pos[1] - green_pos[1]
                
                # 更新误差数据
                tracker.update_dual_error(red_error_x, red_error_y, green_error_x, green_error_y)
                
                # 判断红光是否接近目标点
                if abs(red_error_x) < 15 and abs(red_error_y) < 15:
                    target_idx = (target_idx + 1) % len(path_to_track)
                    if cfg.DEBUG_MODE:
                        logger.debug(f"红光已接近目标点，切换到下一点: {target_idx}")
            
            # 判断追踪状态
            if red_pos and green_pos:
                # 计算激光点间距离
                distance_px = np.linalg.norm(np.array(red_pos) - np.array(green_pos))
                
                # 判断是否失败状态
                is_failing = distance_px > success_distance_px
                
                if is_failing and failure_start is None:
                    failure_start = time.time()
                elif not is_failing:
                    failure_start = None
                
                # 判断初始成功
                if not initial_success and elapsed_time < initial_success_time_limit_s and not is_failing:
                    initial_success = True
                    logger.info(f"初始追踪成功！用时: {elapsed_time:.2f}s")
                    print('\a')
                
                # 判断任务失败
                if failure_start and (time.time() - failure_start) > continuous_failure_time_limit_s:
                    task_failed = True
                    logger.warning(f"任务失败：连续追踪失败超过 {continuous_failure_time_limit_s} 秒！")

            # 视觉反馈
            result_frame = frame.copy()
            
            # 绘制路径
            cv2.polylines(result_frame, [np.array(path_to_track)], True, (255, 255, 0), 2)
            
            # 绘制当前目标点
            if not tracker.paused:
                target = path_to_track[target_idx]
                cv2.circle(result_frame, target, 5, (0, 255, 255), -1)
            
            # 绘制激光点
            if red_pos: cv2.circle(result_frame, red_pos, 8, (0, 0, 255), -1)
            if green_pos: cv2.circle(result_frame, green_pos, 8, (0, 255, 0), -1)
            
            # 绘制状态信息
            if task_failed:
                cv2.putText(result_frame, "TASK FAILED", (50, 100), cv2.FONT_HERSHEY_TRIPLEX, 1.5, (0, 0, 255), 3)
            elif is_failing:
                fail_time = time.time() - failure_start if failure_start else 0
                cv2.putText(result_frame, f"FAILING: {fail_time:.1f}s", (50, 50), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 165, 255), 2)
            elif initial_success:
                cv2.putText(result_frame, "TRACKING OK", (50, 50), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            if tracker.paused:
                cv2.putText(result_frame, "PAUSED", 
                           (result_frame.shape[1]//2-100, result_frame.shape[0]//2), 
                           cv2.FONT_HERSHEY_TRIPLEX, 2, (0, 255, 255), 3)

            cv2.imshow('Dynamic Dual Tracking', result_frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'): break
            elif key == ord('p'): tracker.toggle_pause()
        
        # 任务结束后等待一段时间
        if task_failed:
            cv2.waitKey(2000)
            
    finally:
        tracker.stop_tracking_thread()
        tracker.update_dual_error(0, 0, 0, 0)
        cv2.destroyAllWindows()
        logger.info("--- 退出动态双激光追踪模式 ---")