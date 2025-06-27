import argparse

def parse_args():
    parser = argparse.ArgumentParser(description='激光追踪控制系统')
    parser.add_argument('--port', default='/dev/ttyAMA0', help='串口设备')
    parser.add_argument('--baudrate', type=int, default=115200, help='波特率')
    parser.add_argument('--camera', type=int, default=0, help='摄像头索引')
    parser.add_argument('--width', type=int, default=640, help='摄像头宽度')
    parser.add_argument('--height', type=int, default=480, help='摄像头高度')
    parser.add_argument('--debug', action='store_true', help='启用调试模式')
    return parser.parse_args()

def main():
    args = parse_args()
    
    # 使用命令行参数初始化系统
    tracker = LaserTracker(serial_port=args.port, baudrate=args.baudrate)
    
    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        print("无法打开摄像头")
        tracker.close_serial()
        return
        
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    
    # 剩余代码与原来类似，但使用args参数