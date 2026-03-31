import numpy as np
from scipy.spatial import KDTree
from loguru import logger as log
from typing import List, Tuple, TypeAlias

# [x, y] coordinates
Point: TypeAlias = np.ndarray
PathData: TypeAlias = np.ndarray  # (N, 2) shaped array

def minimize_moves_fast(paths: List[PathData]) -> List[PathData]:
    """
    使用 KDTree 實作高效的最近鄰搜尋 (Nearest Neighbor for TSP-like Path Ordering)
    
    Args:
        paths: 包含所有路徑座標的 List，每一項是 (N, 2) 的 Numpy Array。
    
    Returns:
        優化順序後的路徑清單。
    """
    if not paths:
        return []
        
    log.info(f"開始由 {len(paths)} 條路徑進行提筆路徑優化...")
    
    # 建立所有路徑端點的集庫
    # 每個路徑有 Start (0) 與 End (-1)
    # 用來追蹤哪些路徑還沒被使用
    unused_indices = set(range(len(paths)))
    ordered_paths = []
    
    # 從第一條路徑開始 (維持原有的第 0 條不變或隨機)
    current_idx = 0
    unused_indices.remove(current_idx)
    ordered_paths.append(paths[current_idx])
    
    while unused_indices:
        # 當前路徑的最後一個點
        last_point = ordered_paths[-1][-1]
        
        # 從剩下的路徑中，找出起始點或結束點最靠近 last_point 的
        remaining_indices = list(unused_indices)
        
        # 建立端點座標矩陣：包含剩餘路徑的 start 與 end
        endpoints = []
        for idx in remaining_indices:
            endpoints.append(paths[idx][0])   # Start
            endpoints.append(paths[idx][-1])  # End
        
        endpoints_arr = np.array(endpoints)
        tree = KDTree(endpoints_arr)
        
        # 搜尋最靠近當前結束點的端點
        dist, min_coord_idx = tree.query(last_point)
        
        # 判斷是哪條路徑及其方向
        target_path_idx = remaining_indices[min_coord_idx // 2]
        is_end_point = (min_coord_idx % 2 == 1)
        
        target_path = paths[target_path_idx]
        if is_end_point:
            # 如果是 End 點最接近，則翻轉該路徑
            target_path = target_path[::-1]
            
        ordered_paths.append(target_path)
        unused_indices.remove(target_path_idx)
        
    return ordered_paths

def fuse_points_vectorized(points: np.ndarray, d_sq: float) -> np.ndarray:
    """
    向量化濾除過於接近的點
    """
    if len(points) < 2:
        return points
    
    diff = np.diff(points, axis=0)
    dists_sq = np.sum(diff**2, axis=1)
    
    mask = np.ones(len(points), dtype=bool)
    # 簡單邏輯：若兩點距離小於閾值，標記下一個點為 False
    # 注意：這只是第一層。完整實作需要累積距離。
    # 這裡為簡便，先提供基本去重
    return points[np.concatenate(([True], dists_sq > d_sq))]
