import numpy as np
from config import Config

cfg = Config()

def generate_path(points, num_per_edge=None):
    """根据一系列点生成平滑路径
    
    Args:
        points: 路径点列表，每个点是(x, y)坐标
        num_per_edge: 每条边的点数
        
    Returns:
        路径点列表
    """
    if num_per_edge is None:
        num_per_edge = cfg.PATH_POINTS_PER_EDGE
    
    if not points or len(points) < 2:
        return []
    
    path = []
    for i in range(len(points)):
        start, end = np.array(points[i]), np.array(points[(i + 1) % len(points)])
        for t in np.linspace(0, 1, num_per_edge, endpoint=False):
            path.append(tuple(((1 - t) * start + t * end).astype(int)))
    return path

def generate_centerline_path(outer_rect, inner_rect, num_per_edge=None):
    """根据内外矩形计算中线路径
    
    Args:
        outer_rect: 外部矩形轮廓
        inner_rect: 内部矩形轮廓
        num_per_edge: 每条边的点数
        
    Returns:
        中线路径点列表
    """
    if num_per_edge is None:
        num_per_edge = cfg.PATH_POINTS_PER_EDGE
        
    if outer_rect is None or inner_rect is None:
        return []
    
    outer_pts = outer_rect.reshape(4, 2).astype(np.float32)
    inner_pts = inner_rect.reshape(4, 2).astype(np.float32)

    # 匹配外部和内部矩形的对应顶点
    sorted_outer_pts = np.zeros_like(outer_pts)
    remaining_outer_pts = list(outer_pts)
    
    for i, p_inner in enumerate(inner_pts):
        distances = [np.linalg.norm(p_inner - p_outer) for p_outer in remaining_outer_pts]
        closest_idx = np.argmin(distances)
        sorted_outer_pts[i] = remaining_outer_pts.pop(closest_idx)

    # 计算中线点
    centerline_corners = [(inner_pts[i] + sorted_outer_pts[i]) / 2 for i in range(4)]
    
    # 生成路径
    return generate_path([tuple(p.astype(int)) for p in centerline_corners], num_per_edge)