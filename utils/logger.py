import logging
import os
from datetime import datetime

# 全局日志对象
_logger = None

def setup_logger(debug_mode=False, log_dir="logs"):
    """设置日志系统
    
    Args:
        debug_mode: 是否启用调试模式
        log_dir: 日志存储目录
        
    Returns:
        logger: 日志对象
    """
    global _logger
    
    if _logger is not None:
        return _logger
    
    # 创建日志目录
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    
    log_level = logging.DEBUG if debug_mode else logging.INFO
    log_filename = os.path.join(log_dir, f"laser_tracker_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
    
    _logger = logging.getLogger("LaserTracker")
    _logger.setLevel(log_level)
    
    # 清除已存在的处理器
    if _logger.handlers:
        _logger.handlers.clear()
    
    # 控制台处理器
    console_handler = logging.StreamHandler()
    console_handler.setLevel(log_level)
    console_format = logging.Formatter('%(levelname)s: %(message)s')
    console_handler.setFormatter(console_format)
    
    # 文件处理器
    file_handler = logging.FileHandler(log_filename)
    file_handler.setLevel(log_level)
    file_format = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(file_format)
    
    _logger.addHandler(console_handler)
    _logger.addHandler(file_handler)
    
    _logger.info(f"日志系统初始化完成，日志文件: {log_filename}")
    
    return _logger

def get_logger():
    """获取日志对象，如果未初始化则初始化"""
    global _logger
    if _logger is None:
        _logger = setup_logger()
    return _logger