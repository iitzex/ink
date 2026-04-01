"""
核心處理器：負責圖片前處理、向量化與 G-Code 產出

將影像轉換為高品質的 CNC G-Code，支援透明底處理、
區域填充、路徑優化與精確座標映射。
"""

import tempfile
import subprocess
import numpy as np
from PIL import Image, ImageOps
from pathlib import Path
from loguru import logger as log
from svgpathtools import svg2paths2, Line, CubicBezier, QuadraticBezier, Arc
from simplification.cutil import simplify_coords

from config import AppConfig
from utils.path_opt import minimize_moves_fast
from utils.hatch import generate_hatch_paths_v2


class VectorProcessor:
    """
    核心處理器：負責圖片前處理、向量化與 G-Code 產出
    """

    def __init__(self, config: AppConfig):
        """
        初始化處理器

        Args:
            config: 應用程式參數配置物件
        """
        self.config = config
        self.paths_data: list[np.ndarray] = []
        self.bw_image: Image.Image | None = None

    # ── 前處理階段 ──────────────────────────────────────────

    def preprocess_image(self) -> Path:
        """
        圖片前處理：透明底處理、灰階、二值化、反相與旋轉

        Returns:
            暫存 BMP 檔案路徑，供向量化使用
        """
        log.info(f"正在處理圖片: {self.config.file}")
        img = Image.open(self.config.file)

        img = self._flatten_alpha(img)

        if self.config.invert:
            log.info("執行圖片反相 (Invert)...")
            img = ImageOps.invert(img)

        if self.config.rotate != 0:
            log.info(f"旋轉圖片: {self.config.rotate} 度")
            img = img.rotate(
                self.config.rotate, expand=True,
                resample=Image.BICUBIC, fillcolor=255,
            )

        # 二值化
        threshold_val = int(255 * (self.config.threshold / 100.0))
        bw = img.point(lambda p: 255 if p > threshold_val else 0, mode="1")

        # 保存二值化影像用於填充取樣
        self.bw_image = bw.convert("L")

        # 使用 tempfile 避免污染工作目錄
        temp_bmp = tempfile.NamedTemporaryFile(suffix=".bmp", delete=False)
        bw.save(temp_bmp.name)
        return Path(temp_bmp.name)

    @staticmethod
    def _flatten_alpha(img: Image.Image) -> Image.Image:
        """
        將帶有透明通道的影像扁平化至白底灰階

        Args:
            img: 原始 PIL Image

        Returns:
            灰階 (mode='L') 的 PIL Image
        """
        if img.mode in ("RGBA", "LA") or (img.mode == "P" and "transparency" in img.info):
            log.info(f"偵測到透明通道 ({img.mode})，執行透明底扁平化至白色背景...")
            background = Image.new("RGBA", img.size, (255, 255, 255, 255))
            rgba = img if img.mode == "RGBA" else img.convert("RGBA")
            background.paste(rgba, (0, 0), rgba)
            return background.convert("L")
        return img.convert("L")

    # ── 向量化階段 ──────────────────────────────────────────

    def vectorize(self, bmp_path: Path) -> Path:
        """
        調用 Potrace 或 Autotrace 進行向量化

        Args:
            bmp_path: 二值化 BMP 的暫存路徑

        Returns:
            SVG 向量檔暫存路徑
        """
        temp_svg = tempfile.NamedTemporaryFile(suffix=".svg", delete=False)
        svg_path = Path(temp_svg.name)

        if self.config.autotrace:
            log.info("使用 Autotrace (Centerline) 進行向量化...")
            temp_tga = tempfile.NamedTemporaryFile(suffix=".tga", delete=False)
            Image.open(bmp_path).save(temp_tga.name)
            cmd = f"autotrace -output-file {svg_path} --output-format svg --centerline {temp_tga.name}"
            subprocess.run(cmd.split(), check=True)
            Path(temp_tga.name).unlink(missing_ok=True)
        else:
            log.info("使用 Potrace (Outline) 進行向量化...")
            cmd = f"potrace -b svg -o {svg_path} -n {bmp_path}"
            subprocess.run(cmd.split(), check=True)

        return svg_path

    # ── SVG 解析階段 ────────────────────────────────────────

    def parse_svg(self, svg_path: Path) -> None:
        """
        解析 SVG 並將座標提取為 Numpy Arrays

        Args:
            svg_path: SVG 向量檔路徑
        """
        paths, _, _ = svg2paths2(str(svg_path))

        if self.bw_image is None:
            log.error("未找到二值化影像，無法解析 SVG 座標基準")
            return

        w, h = self.bw_image.size
        offset_x = w / 2.0 if self.config.center else 0
        offset_y = h / 2.0 if self.config.center else 0

        extracted_paths: list[np.ndarray] = []
        for path in paths:
            for sub_pts in self._extract_sub_paths(path):
                transformed = self._transform_points(sub_pts, offset_x, offset_y)
                if len(transformed) >= self.config.minpath:
                    extracted_paths.append(transformed)

        self.paths_data = extracted_paths

    def _extract_sub_paths(self, path) -> list[list[list[float]]]:
        """
        從 SVG path 中提取子路徑點位，偵測斷點以避免刮痕

        Args:
            path: svgpathtools 的 Path 物件

        Returns:
            各子路徑的點位列表
        """
        sub_paths: list[list[list[float]]] = []
        current_points: list[list[float]] = []

        for i, segment in enumerate(path):
            # 偵測斷點 (Discontinuity check)
            if i > 0 and abs(segment.start - path[i - 1].end) > 1e-5:
                if current_points:
                    sub_paths.append(current_points)
                current_points = []

            # 採樣座標點
            if isinstance(segment, Line):
                current_points.append([segment.start.real, segment.start.imag])
            elif isinstance(segment, (CubicBezier, QuadraticBezier, Arc)):
                for t in np.linspace(0, 1, 10, endpoint=False):
                    p = segment.point(t)
                    current_points.append([p.real, p.imag])

        if current_points:
            last_p = path[-1].end
            current_points.append([last_p.real, last_p.imag])
            sub_paths.append(current_points)

        return sub_paths

    def _transform_points(
        self, pts: list[list[float]], offset_x: float, offset_y: float
    ) -> np.ndarray:
        """
        將 Potrace 原始座標轉換為 CNC 實體座標 (mm)

        Args:
            pts: 原始點位清單
            offset_x: X 軸偏移 (px)
            offset_y: Y 軸偏移 (px)

        Returns:
            轉換後的 (N, 2) numpy 陣列
        """
        # Potrace 座標是 10 倍像素 (0.1pt)
        pts_arr = np.array(pts) / 10.0
        transformed = np.zeros_like(pts_arr)
        transformed[:, 0] = (pts_arr[:, 0] - offset_x) * self.config.mmperpixel
        transformed[:, 1] = (pts_arr[:, 1] - offset_y) * self.config.mmperpixel

        if self.config.simplify > 0:
            transformed = simplify_coords(transformed, self.config.simplify)

        return transformed

    # ── 填充與優化階段 ──────────────────────────────────────

    def generate_hatch_paths(self) -> None:
        """
        生成區域填充路徑，使用高效向量化引擎 (零提筆模式)
        """
        if self.config.hatch <= 0 or self.bw_image is None:
            return

        log.info(f"正在生成區域填充 (Hatch)... 間距: {self.config.hatch}mm (Continuous Mode)")
        hatch_paths = generate_hatch_paths_v2(
            self.bw_image, self.config.hatch,
            self.config.mmperpixel, self.config.center,
        )
        log.info(f"填充完成，產出了 {len(hatch_paths)} 個連續填充區塊 (零提筆模式)。")
        self.paths_data.extend(hatch_paths)

    def optimize_paths(self) -> None:
        """
        執行提筆路徑優化，減少空走距離
        """
        if self.config.minimize and len(self.paths_data) > 1:
            self.paths_data = minimize_moves_fast(self.paths_data)

    # ── 輸出階段 ────────────────────────────────────────────

    def save_results(self) -> None:
        """
        產出 G-Code 與 SVG 預覽（調度入口）
        """
        if not self.paths_data:
            log.warning("沒有可輸出的路徑數據")
            return

        self._apply_centering()

        output_dir = self.config.output_path
        output_dir.mkdir(parents=True, exist_ok=True)

        self._write_gcode(output_dir / "image.gcode")
        self._estimate_execution_time()
        self._write_svg_preview(output_dir / "final.svg")

    def _apply_centering(self) -> None:
        """
        計算物件包圍盒並將幾何中心對齊到 CNC (0, 0) 原點
        """
        if not self.config.center:
            return

        all_pts = np.vstack(self.paths_data)
        min_xy = np.min(all_pts, axis=0)
        max_xy = np.max(all_pts, axis=0)
        center = (min_xy + max_xy) / 2.0

        log.info(f"執行物件包圍盒置中... 偏移: X={-center[0]:.2f}, Y={-center[1]:.2f}")
        for i in range(len(self.paths_data)):
            self.paths_data[i] -= center

    def _write_gcode(self, gcode_path: Path) -> None:
        """
        將路徑數據寫入 G-Code 檔案

        使用 G00 快速定位進行空走移動（X/Y/Z 提筆），
        使用 G01 進行雕刻進給。

        Args:
            gcode_path: 輸出的 G-Code 檔案路徑
        """
        log.info(f"正在寫入 G-Code 至: {gcode_path}")
        cfg = self.config

        with open(gcode_path, "w") as f:
            f.write(f"M3 S{cfg.spindle_speed}\n")
            f.write(f"G00 Z{cfg.penup}\n")
            for path in self.paths_data:
                f.write(f"G00 Z{cfg.penup}\n")
                f.write(f"G00 X{path[0][0]:.3f} Y{path[0][1]:.3f}\n")
                f.write(f"G01 Z{cfg.pendown} F{cfg.feedrate}\n")
                for pt in path[1:]:
                    f.write(f"G01 X{pt[0]:.3f} Y{pt[1]:.3f}\n")
                f.write(f"G00 Z{cfg.penup}\n")
            f.write("M05\n")

    def _write_svg_preview(self, svg_path: Path) -> None:
        """
        建立 SVG 預覽檔，使用動態 viewBox 確保完整顯示

        Args:
            svg_path: 輸出的 SVG 預覽檔路徑
        """
        log.info(f"產出 SVG 預覽: {svg_path}")

        all_pts = np.vstack(self.paths_data)
        min_x, min_y = np.min(all_pts, axis=0)
        max_x, max_y = np.max(all_pts, axis=0)

        w_val, h_val = max_x - min_x, max_y - min_y
        padding = max(w_val, h_val) * 0.05
        vb_x, vb_y = min_x - padding, min_y - padding
        vb_w, vb_h = w_val + 2 * padding, h_val + 2 * padding

        with open(svg_path, "w") as f:
            f.write('<?xml version="1.0" encoding="utf-8" ?>\n')
            f.write(f'<svg xmlns="http://www.w3.org/2000/svg" '
                    f'viewBox="{vb_x} {vb_y} {vb_w} {vb_h}" '
                    f'width="800" height="800">\n')
            f.write('  <g transform="scale(1, -1)">\n')
            for path in self.paths_data:
                pts_str = " ".join(f"{pt[0]},{-pt[1]}" for pt in path)
                f.write(f'    <polyline points="{pts_str}" '
                        f'fill="none" stroke="black" stroke-width="0.5" />\n')
            f.write("  </g>\n")
            f.write("</svg>")

    def _estimate_execution_time(self) -> None:
        """
        計算並輸出 G-Code 的預估加工時間

        考量三種距離：雕刻路徑、跳轉空走、Z 軸升降，
        並加入 15% 安全係數以涵蓋機台加速度損耗。
        """
        if not self.paths_data:
            return

        # 雕刻距離
        total_work_dist = sum(
            np.sum(np.sqrt(np.sum(np.diff(p, axis=0) ** 2, axis=1)))
            for p in self.paths_data if len(p) >= 2
        )

        # 空走跳轉距離
        total_rapid_dist = sum(
            np.linalg.norm(self.paths_data[i + 1][0] - self.paths_data[i][-1])
            for i in range(len(self.paths_data) - 1)
        )

        # Z 軸垂直距離
        z_per_path = abs(self.config.penup - self.config.pendown) * 2
        total_z_dist = len(self.paths_data) * z_per_path

        # 時間計算 (分鐘) + 15% 安全係數
        t_work = total_work_dist / self.config.feedrate
        t_rapid = (total_rapid_dist + total_z_dist) / self.config.rapid_feedrate
        total_min = (t_work + t_rapid) * 1.15

        m, s = int(total_min), int((total_min % 1) * 60)
        log.info("--- 加工時間報表 ---")
        log.info(f"總雕刻距離: {total_work_dist:.2f} mm")
        log.info(f"總提筆移動: {total_rapid_dist + total_z_dist:.2f} mm")
        log.info(f"預估加工時間: <yellow>{m:02d}m {s:02d}s</yellow> (含 15% 安全係數)")
        log.info("-------------------")

    # ── 主流程 ──────────────────────────────────────────────

    def run_all(self) -> None:
        """
        執行完整的影像轉 G-Code 流程
        """
        bmp = self.preprocess_image()
        svg = self.vectorize(bmp)
        self.parse_svg(svg)
        self.generate_hatch_paths()
        self.optimize_paths()
        self.save_results()

        # 清理暫存檔
        bmp.unlink(missing_ok=True)
        svg.unlink(missing_ok=True)
        log.success("轉檔完成！")
