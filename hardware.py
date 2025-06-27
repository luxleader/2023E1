import serial
import struct
import threading
import time
from config import Config
from utils.logger import get_logger

class LaserTracker:
    """
    负责与下位机进行串口通信的类。
    包含了多线程、二进制协议、暂停/恢复等高级功能。
    """
    def __init__(self, serial_port=None, baudrate=None):
        # 加载配置
        self.cfg = Config()
        
        # 初始化日志
        self.logger = get_logger()
        
        # 设置串口参数
        self.serial_port = serial_port or self.cfg.SERIAL_PORT
        self.baudrate = baudrate or self.cfg.BAUDRATE
        
        self.ser = None
        self._connect_serial()
        
        # 状态与数据
        self.paused = False
        self.tracking_active = False
        self.sender_thread = None
        self.error_lock = threading.Lock()
        self.current_mode = None
        
        # 错误数据
        self.last_error_x = 0
        self.last_error_y = 0
        self.last_green_x = 0
        self.last_green_y = 0
        self.last_green_error_x = 0
        self.last_green_error_y = 0
        
        self.receive_buffer = bytearray()

    def _connect_serial(self):
        """连接串口，并处理可能的异常"""
        try:
            self.ser = serial.Serial(
                port=self.serial_port,
                baudrate=self.baudrate,
                parity=serial.PARITY_NONE,
                stopbits=serial.STOPBITS_ONE,
                bytesize=serial.EIGHTBITS,
                timeout=0.1
            )
            self.logger.info(f"串口 {self.serial_port} 连接成功")
            try:
                self.ser.write(b"HOST_CONNECTED\n")
                self.logger.info("已发送连接成功通知至下位机")
            except Exception as e:
                self.logger.error(f"发送连接通知失败: {e}")
        except Exception as e:
            self.logger.error(f"串口连接失败: {e}")
            self.ser = None

    def _try_reconnect(self):
        """尝试重新连接串口"""
        if self.ser and self.ser.is_open:
            self.ser.close()
        
        retry_count = 0
        max_retries = 5
        
        while retry_count < max_retries:
            try:
                self.ser = serial.Serial(
                    port=self.serial_port,
                    baudrate=self.baudrate,
                    parity=serial.PARITY_NONE,
                    stopbits=serial.STOPBITS_ONE,
                    bytesize=serial.EIGHTBITS,
                    timeout=0.1
                )
                self.logger.info(f"串口 {self.serial_port} 重连成功")
                return True
            except Exception as e:
                retry_count += 1
                self.logger.warning(f"重连尝试 {retry_count}/{max_retries} 失败: {e}")
                time.sleep(1)
        
        self.logger.error("串口重连失败，请检查硬件连接")
        return False

    def toggle_pause(self):
        """切换暂停或恢复状态，并返回当前是否为暂停状态。"""
        with self.error_lock:
            self.paused = not self.paused
            self.logger.info(f"系统已 {'暂停' if self.paused else '恢复'}")
            return self.paused

    def update_target_error(self, error_x, error_y):
        """更新单激光误差值。"""
        with self.error_lock:
            self.last_error_x = error_x
            self.last_error_y = error_y

    def update_target_error_with_green(self, error_x, error_y, green_x=None, green_y=None):
        """更新红光误差和绿光绝对位置。"""
        with self.error_lock:
            self.last_error_x = error_x
            self.last_error_y = error_y
            self.last_green_x = green_x if green_x is not None else 0
            self.last_green_y = green_y if green_y is not None else 0

    def update_dual_error(self, red_error_x, red_error_y, green_error_x, green_error_y):
        """更新双激光的独立误差值。"""
        with self.error_lock:
            self.last_error_x = red_error_x
            self.last_error_y = red_error_y
            self.last_green_error_x = green_error_x
            self.last_green_error_y = green_error_y

    def _sending_loop(self):
        """统一的发送线程，根据当前模式发送不同数据"""
        while self.tracking_active:
            try:
                with self.error_lock:
                    is_paused = self.paused
                    mode = self.current_mode
                    
                    # 根据不同模式准备数据
                    if mode == self.cfg.MODE_PID:
                        data = (self.last_error_x, self.last_error_y)
                        cmd_id = self.cfg.CMD_ID_PID
                        pack_format = '>hh'
                    elif mode == self.cfg.MODE_DUAL_INFO:
                        data = (self.last_error_x, self.last_error_y, 
                                self.last_green_x, self.last_green_y)
                        cmd_id = self.cfg.CMD_ID_DUAL_INFO
                        pack_format = '>hhhh'
                    elif mode == self.cfg.MODE_DUAL_CONTROL:
                        data = (self.last_error_x, self.last_error_y,
                                self.last_green_error_x, self.last_green_error_y)
                        cmd_id = self.cfg.CMD_ID_DUAL_CONTROL
                        pack_format = '>hhhh'
                    else:
                        # 未知模式，使用默认PID模式
                        data = (self.last_error_x, self.last_error_y)
                        cmd_id = self.cfg.CMD_ID_PID
                        pack_format = '>hh'
                
                # 如果暂停，所有数据清零
                if is_paused:
                    data = tuple(0 for _ in data)
                
                # 发送命令
                if all(x is not None for x in data):
                    payload = struct.pack(pack_format, *[int(x) for x in data])
                    self._send_command(cmd_id, payload)
                
                time.sleep(self.cfg.SEND_INTERVAL_S)
            except Exception as e:
                self.logger.error(f"发送线程发生错误: {e}")
                # 如果发送错误，尝试重连
                if not self.ser or not self.ser.is_open:
                    self._try_reconnect()

    def start_tracking_thread(self, mode=None):
        """启动追踪线程，可指定模式"""
        if mode is not None:
            self.current_mode = mode
        elif self.current_mode is None:
            self.current_mode = self.cfg.MODE_PID  # 默认使用PID模式
            
        if not self.sender_thread:
            self.tracking_active = True
            self.sender_thread = threading.Thread(target=self._sending_loop, daemon=True)
            self.sender_thread.start()
            self.logger.info(f"已启动追踪线程，模式: {self.current_mode}")
        else:
            self.logger.warning("追踪线程已在运行")

    def stop_tracking_thread(self):
        """停止追踪线程"""
        if self.sender_thread:
            self.tracking_active = False
            try:
                self.sender_thread.join(timeout=0.5)
            except Exception as e:
                self.logger.warning(f"停止线程时发生异常: {e}")
            self.sender_thread = None
            self.logger.info("发送线程已停止")

    def _send_command(self, command_id, payload):
        """发送二进制命令到下位机"""
        if not (self.ser and self.ser.is_open):
            self.logger.warning("串口未连接，无法发送命令")
            return
            
        try:
            header = bytes([self.cfg.FRAME_HEADER])
            cmd_byte = struct.pack('B', command_id)
            trailer = struct.pack('B', self.cfg.FRAME_TRAILER)
            packet = header + cmd_byte + payload + trailer
            self.ser.write(packet)
            if self.cfg.DEBUG_MODE:
                self.logger.debug(f"发送命令: ID={command_id}, 负载大小={len(payload)}字节")
        except Exception as e:
            self.logger.error(f"串口发送失败: {e}")
            # 如果发送错误，尝试重连
            self._try_reconnect()

    def send_reset_command(self):
        """发送复位命令"""
        payload = struct.pack('>hh', 0, 0)
        self._send_command(self.cfg.CMD_ID_RESET, payload)
        self.logger.info("已发送复位命令")

    def receive_command(self):
        """接收并解析下位机命令"""
        if not (self.ser and self.ser.is_open and self.ser.in_waiting > 0):
            return None
            
        try:
            command_str = self.ser.readline().decode('utf-8').strip()
            parts = command_str.split()
            if len(parts) == 3 and parts[0].upper() == '0XA5' and parts[2].upper() == '0X5A' and len(parts[1]) == 1:
                self.logger.info(f"接收到有效串口指令: '{parts[1]}'")
                return parts[1]
        except Exception as e:
            self.logger.error(f"串口接收数据失败: {e}")
        return None

    def close_serial(self):
        """关闭串口连接"""
        self.stop_tracking_thread()
        if self.ser and self.ser.is_open:
            try:
                self.ser.close()
                self.logger.info("串口已关闭")
            except Exception as e:
                self.logger.error(f"关闭串口时发生错误: {e}")