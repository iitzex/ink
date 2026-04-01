"""
提筆路徑優化模組

使用 KDTree 空間索引實作最近鄰搜尋 (Nearest Neighbor Heuristic)，
將路徑重新排序以最小化提筆空走距離，並使用 2-opt 演算法進一步優化。
"""

import numpy as np
from scipy.spatial import KDTree
from loguru import logger as log
from typing import TypeAlias

# 型別定義
PathData: TypeAlias = np.ndarray  # (N, 2) shaped array


def minimize_moves_fast(paths: list[PathData]) -> list[PathData]:
    """
    使用一次性 KDTree 實作高效的最近鄰路徑排序，並以 2-opt 進一步優化

    一次建構所有端點的 KDTree，再以貪心法逐一挑選
    距離當前位置最近的未使用路徑。若最近的是路徑終點，
    則自動翻轉該路徑方向。最後使用輕量級 2-opt 演算法優化。

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
    # 索引規則：path i → start = 2*i, end = 2*i+1
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

    # 使用輕量級 2-opt 進一步優化
    log.info("執行 2-opt 路徑交換優化...")
    ordered = two_opt_lightweight(ordered)

    return ordered


def two_opt_lightweight(paths: list[PathData], max_iterations: int = 50) -> list[PathData]:
    """
    輕量級 2-opt 演算法：只優化「端點連接」而非路徑內部

    核心概念：
    - 只考慮路徑的起點和終點（端點）
    - 嘗試翻轉路徑或交換相鄰路徑來減少空走距離
    - 避免 O(n³) 的交叉交換，只保留 O(n) 的局部優化

    Args:
        paths: 已排序的路徑清單
        max_iterations: 最大迭代次數

    Returns:
        優化後的路徑清單
    """
    if len(paths) <= 2:
        return paths

    n = len(paths)

    # 預先計算所有路徑的起點和終點（避免重複存取）
    starts = np.array([p[0] for p in paths])
    ends = np.array([p[-1] for p in paths])

    # 路徑方向：False=正向，True=反向
    reversed_flags = [False] * n

    def get_start(i: int) -> np.ndarray:
        return ends[i] if reversed_flags[i] else starts[i]

    def get_end(i: int) -> np.ndarray:
        return starts[i] if reversed_flags[i] else ends[i]

    def calc_total_rapid_dist() -> float:
        """快速計算總空走距離（只算端點）"""
        total = 0.0
        for i in range(n - 1):
            total += np.linalg.norm(get_start(i + 1) - get_end(i))
        return total

    current_dist = calc_total_rapid_dist()
    improved = True
    iteration = 0

    while improved and iteration < max_iterations:
        improved = False
        iteration += 1

        # 策略 1: 嘗試翻轉單一路徑（只影響相鄰兩段的空走距離）
        for i in range(n):
            old_dist = 0.0
            if i > 0:
                old_dist += np.linalg.norm(get_start(i) - get_end(i - 1))
            if i < n - 1:
                old_dist += np.linalg.norm(get_start(i + 1) - get_end(i))

            # 翻轉後的距離
            new_start = get_end(i)
            new_end = get_start(i)
            new_dist = 0.0
            if i > 0:
                new_dist += np.linalg.norm(new_start - get_end(i - 1))
            if i < n - 1:
                new_dist += np.linalg.norm(get_start(i + 1) - new_end)

            if new_dist < old_dist - 1e-6:
                reversed_flags[i] = not reversed_flags[i]
                current_dist = current_dist - old_dist + new_dist
                improved = True
                log.debug(f"2-opt: 翻轉路徑 {i}, 距離減少 {old_dist - new_dist:.2f} mm")

        # 策略 2: 嘗試交換相鄰路徑順序（只影響局部）
        for i in range(n - 1):
            # 原始：... → end(i-1) → start(i) → end(i) → start(i+1) → end(i+1) → start(i+2) ...
            # 交換後：... → end(i-1) → start(i+1) → end(i+1) → start(i) → end(i) → start(i+2) ...

            old_dist = 0.0
            if i > 0:
                old_dist += np.linalg.norm(get_start(i) - get_end(i - 1))
            old_dist += np.linalg.norm(get_start(i + 1) - get_end(i))
            if i < n - 2:
                old_dist += np.linalg.norm(get_start(i + 2) - get_end(i + 1))

            # 交換後的距離
            new_dist = 0.0
            if i > 0:
                new_dist += np.linalg.norm(get_start(i + 1) - get_end(i - 1))
            new_dist += np.linalg.norm(get_start(i) - get_end(i + 1))
            if i < n - 2:
                new_dist += np.linalg.norm(get_start(i + 2) - get_end(i))

            if new_dist < old_dist - 1e-6:
                # 實際交換
                starts[i], starts[i + 1] = starts[i + 1], starts[i]
                ends[i], ends[i + 1] = ends[i + 1], ends[i]
                reversed_flags[i], reversed_flags[i + 1] = reversed_flags[i + 1], reversed_flags[i]
                current_dist = current_dist - old_dist + new_dist
                improved = True
                log.debug(f"2-opt: 交換路徑 {i} 與 {i+1}, 距離減少 {old_dist - new_dist:.2f} mm")

    # 根據優化結果重建路徑清單
    result: list[PathData] = []
    for i in range(n):
        p = paths[i].copy()
        if reversed_flags[i]:
            p = p[::-1]
        result.append(p)

    log.info(f"2-opt 完成：迭代 {iteration} 次，總空走距離 {current_dist:.2f} mm")
    return result
