import numpy as np
import cv2

class LaserKalmanFilter:
    """卡尔曼滤波器实现，用于平滑激光点轨迹"""
    def __init__(self):
        # 状态: [x, y, vx, vy]
        self.kalman = cv2.KalmanFilter(4, 2)
        self.kalman.measurementMatrix = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], np.float32)
        self.kalman.transitionMatrix = np.array([[1, 0, 1, 0], [0, 1, 0, 1], 
                                                [0, 0, 1, 0], [0, 0, 0, 1]], np.float32)
        self.kalman.processNoiseCov = np.array([[1, 0, 0, 0], [0, 1, 0, 0], 
                                              [0, 0, 1, 0], [0, 0, 0, 1]], np.float32) * 0.03
        self.initialized = False
    
    def update(self, point):
        """更新滤波器状态并获取平滑后的位置
        
        Args:
            point: 当前测量的点坐标，或None表示没有测量
            
        Returns:
            平滑后的点坐标，或None表示无法计算
        """
        if point is None:
            if not self.initialized:
                return None
            # 预测但不更新
            prediction = self.kalman.predict()
            return (int(prediction[0]), int(prediction[1]))
        
        # 转换为测量格式
        measurement = np.array([[point[0]], [point[1]]], dtype=np.float32)
        
        if not self.initialized:
            # 首次测量，初始化状态
            self.kalman.statePre = np.array([[point[0]], [point[1]], [0], [0]], dtype=np.float32)
            self.kalman.statePost = np.array([[point[0]], [point[1]], [0], [0]], dtype=np.float32)
            self.initialized = True
            return point
        
        # 预测并更新
        self.kalman.predict()
        correction = self.kalman.correct(measurement)
        return (int(correction[0]), int(correction[1]))