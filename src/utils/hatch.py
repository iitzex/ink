"""
高效向量化填充引擎

使用 Numpy 矩陣運算進行像素級掃描取樣，支援零提筆 (Continuous)
連接路徑，大幅減少 Z 軸升降次數以提升 CNC 加工效率。
"""

import numpy as np
from PIL import Image


def generate_hatch_paths_v2(
    bw_image: Image.Image,
    hatch_mm: float,
    scale_mm: float,
    center: bool = True,
) -> list[np.ndarray]:
    """
    產生區域填充路徑（零提筆之字形模式）

    以水平掃描線逐行掃描二值化影像，在黑色像素區間內生成填充路徑。
    相鄰掃描線若距離在容差內，則自動鏈結為連續路徑，避免提筆。

    Args:
        bw_image: 灰階二值化影像 (mode='L')
        hatch_mm: 填充線間距 (mm)
        scale_mm: 每像素對應的公釐數
        center: 是否啟用置中偏移

    Returns:
        填充路徑清單，每項為 (N, 2) 的 numpy 陣列
    """
    if hatch_mm <= 0:
        return []

    w, h = bw_image.size
    hatch_px = hatch_mm / scale_mm

    offset_x = w / 2.0 if center else 0
    offset_y = h / 2.0 if center else 0

    # 一次性轉為 numpy 陣列
    img_arr = np.array(bw_image)

    y_coords = np.arange(0, h, hatch_px)
    all_paths: list[np.ndarray] = []
    current_chain: list[list[float]] = []

    for i, py in enumerate(y_coords):
        y_idx = int(py)
        if y_idx >= h:
            break

        # 向量化找出黑色區間邊界
        segments = _find_black_segments(img_arr[y_idx, :])
        if not segments:
            _flush_chain(current_chain, all_paths)
            current_chain = []
            continue

        # 奇數行反向掃描 (Zig-Zag)
        if i % 2 == 1:
            segments.reverse()

        for x1, x2 in segments:
            p_start, p_end = _to_cnc_coords(
                x1, x2, py, offset_x, offset_y, scale_mm, reverse=(i % 2 == 1),
            )

            # 零提筆鏈結判斷
            if not current_chain:
                current_chain.extend([p_start, p_end])
            else:
                dist = np.linalg.norm(
                    np.array(p_start) - np.array(current_chain[-1])
                )
                if dist < hatch_mm * 1.5:
                    current_chain.extend([p_start, p_end])
                else:
                    _flush_chain(current_chain, all_paths)
                    current_chain = [p_start, p_end]

    _flush_chain(current_chain, all_paths)
    return all_paths


def _find_black_segments(row: np.ndarray) -> list[tuple[int, int]]:
    """
    使用 Numpy 向量化找出單行像素中的黑色連續區間

    Args:
        row: 單行灰階像素值 (1D ndarray)

    Returns:
        黑色區間的 (start_x, end_x) 清單
    """
    padded = np.concatenate(([255], row, [255]))
    diff = np.diff((padded < 128).astype(np.int8))
    starts = np.where(diff == 1)[0]
    ends = np.where(diff == -1)[0] - 1
    return list(zip(starts, ends))


def _to_cnc_coords(
    x1: int, x2: int, py: float,
    offset_x: float, offset_y: float,
    scale: float, reverse: bool,
) -> tuple[list[float], list[float]]:
    """
    將像素座標轉換為 CNC 實體座標

    Args:
        x1: 區間起始像素
        x2: 區間結束像素
        py: Y 軸像素座標
        offset_x: X 偏移量
        offset_y: Y 偏移量
        scale: mm/pixel 比例
        reverse: 是否反轉方向（奇數行）

    Returns:
        (起點, 終點) 的 CNC 座標
    """
    cx1 = (x1 - offset_x) * scale
    cx2 = (x2 - offset_x) * scale
    cy = (offset_y - py) * scale

    if reverse:
        return [cx2, cy], [cx1, cy]
    return [cx1, cy], [cx2, cy]


def _flush_chain(
    chain: list[list[float]], target: list[np.ndarray]
) -> None:
    """
    將累積的連續路徑鏈提交至目標清單

    Args:
        chain: 當前累積的連續點位
        target: 輸出的路徑清單
    """
    if chain:
        target.append(np.array(chain))
