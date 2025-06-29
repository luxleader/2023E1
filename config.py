# 配置文件，管理所有系统参数
class Config:
    # 通信协议常量
    SERIAL_PORT = "/dev/ttyAMA0"  # 默认串口，可通过命令行参数覆盖
    BAUDRATE = 115200
    FRAME_HEADER = 0x55
    FRAME_TRAILER = 0xBB
    CMD_ID_PID = 0x01
    CMD_ID_RESET = 0x02
    CMD_ID_DUAL_INFO = 0x03
    CMD_ID_DUAL_CONTROL = 0x04
    
    # 追踪模式枚举
    MODE_PID = 1          # 单激光PID控制模式
    MODE_DUAL_INFO = 2    # 双激光信息模式
    MODE_DUAL_CONTROL = 3 # 双激光独立控制模式
    
    # 追踪参数
    PIXELS_PER_CM = 20
    SUCCESS_DISTANCE_CM = 3.0
    INITIAL_SUCCESS_TIME_LIMIT_S = 2.0
    CONTINUOUS_FAILURE_TIME_LIMIT_S = 3.0
    SEND_INTERVAL_S = 0.01
    
    # 视觉参数
    ROI_SIZE = 100  # ROI区域大小，用于优化检测
    CONTOUR_MIN_AREA = 1  # 激光点最小面积
    
    # 颜色检测参数
    RED_LOWER_1 = (0, 100, 100)
    RED_UPPER_1 = (10, 255, 255)
    RED_LOWER_2 = (160, 100, 100)
    RED_UPPER_2 = (179, 255, 255)
    GREEN_LOWER = (40, 40, 40)
    GREEN_UPPER = (80, 255, 255)
    
    # 摄像头参数
    CAMERA_INDEX = 1
    CAMERA_WIDTH = 640
    CAMERA_HEIGHT = 480
    
    # 其他参数
    DEBUG_MODE = False
    MAX_SAVED_POINTS = 5
    PATH_POINTS_PER_EDGE = 50