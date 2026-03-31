"""
提筆路徑優化模組

使用 KDTree 空間索引實作最近鄰搜尋 (Nearest Neighbor Heuristic)，
將路徑重新排序以最小化提筆空走距離。
"""

import numpy as np
from scipy.spatial import KDTree
from loguru import logger as log
from typing import TypeAlias

# 型別定義
PathData: TypeAlias = np.ndarray  # (N, 2) shaped array


def minimize_moves_fast(paths: list[PathData]) -> list[PathData]:
    """
    使用一次性 KDTree 實作高效的最近鄰路徑排序

    一次建構所有路徑端點的 KDTree，再以貪心法逐一挑選
    距離當前位置最近的未使用路徑。若最近的是路徑終點，
    則自動翻轉該路徑方向。

    Args:
        paths: 包含所有路徑座標的 List，每一項是 (N, 2) 的 Numpy Array

    Returns:
        優化順序後的路徑清單
    """
    if not paths:
        return []

    n = len(paths)
    log.info(f"開始由 {n} 條路徑進行提筆路徑優化...")

    # 一次性建構所有端點的 KDTree
    # 索引規則: path i → start = 2*i, end = 2*i+1
    endpoints = np.empty((n * 2, 2))
    for i, p in enumerate(paths):
        endpoints[2 * i] = p[0]       # Start
        endpoints[2 * i + 1] = p[-1]  # End

    tree = KDTree(endpoints)

    # 貪心搜尋
    visited = np.zeros(n, dtype=bool)
    ordered: list[PathData] = []

    # 從第 0 條開始
    visited[0] = True
    ordered.append(paths[0])

    for _ in range(n - 1):
        last_point = ordered[-1][-1]

        # 從 KDTree 搜尋最近的 k 個端點
        # 最多需要查 2*n 個（但大部分情況只需少量）
        dists, indices = tree.query(last_point, k=min(n * 2, 2 * n))

        # 找到第一個屬於未 visited 路徑的端點
        for dist, idx in zip(dists, indices):
            path_idx = idx // 2
            if not visited[path_idx]:
                is_end = (idx % 2 == 1)
                target = paths[path_idx][::-1] if is_end else paths[path_idx]
                ordered.append(target)
                visited[path_idx] = True
                break

    return ordered
