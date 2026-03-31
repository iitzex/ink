import subprocess
import numpy as np
from PIL import Image, ImageOps
from pathlib import Path
from loguru import logger as log
from svgpathtools import svg2paths2, Line, CubicBezier, QuadraticBezier, Arc
from simplification.cutil import simplify_coords
from shapely.geometry import Polygon, MultiLineString, LineString

from config import AppConfig
from utils.path_opt import minimize_moves_fast

class VectorProcessor:
    """
    核心處理器：負責圖片前處理、向量化與 G-Code 產出
    """
    def __init__(self, config: AppConfig):
        self.config = config
        self.paths_data: list[np.ndarray] = [] # List of (N, 2) arrays
        self.bw_image = None # 用於像素級填充取樣

    def preprocess_image(self) -> Path:
        """
        圖片前處理：透明底處理、灰階、二值化、反相與旋轉
        """
        log.info(f"正在處裡圖片: {self.config.file}")
        img = Image.open(self.config.file)
        
        # 處理透明通道 (Alpha Flattening)
        # 如果是 RGBA 或 LA，將其貼合至白色背景，避免透明處變黑
        if img.mode in ("RGBA", "LA") or (img.mode == "P" and "transparency" in img.info):
            log.info(f"偵測到透明通道 ({img.mode})，執行透明底扁平化至白色背景...")
            background = Image.new("RGBA", img.size, (255, 255, 255, 255))
            if img.mode == "RGBA":
                background.paste(img, (0, 0), img)
            else:
                background.paste(img.convert("RGBA"), (0, 0), img.convert("RGBA"))
            img = background.convert("L")
        else:
            img = img.convert("L")
            
        # 反相 (Invert)
        if self.config.invert:
            log.info("執行圖片反相 (Invert)...")
            img = ImageOps.invert(img)

        # 旋轉 (Rotate) - 在二值化前旋轉可以保持邊緣平滑
        if self.config.rotate != 0:
            log.info(f"旋轉圖片: {self.config.rotate} 度")
            img = img.rotate(self.config.rotate, expand=True, resample=Image.BICUBIC, fillcolor=255)

        # 二值化
        threshold_val = int(255 * (self.config.threshold / 100.0))
        bw = img.point(lambda p: 255 if p > threshold_val else 0, mode='1')
        
        # 行 44: 保存二值化影像用於填充取樣
        self.bw_image = bw.convert("L")
            
        temp_bmp = Path("temp_thresholded.bmp")
        bw.save(temp_bmp)
        return temp_bmp

    def vectorize(self, bmp_path: Path) -> Path:
        """
        調用 Potrace 或 Autotrace 進行向量化
        """
        svg_path = Path("temp_vector.svg")
        if self.config.autotrace:
            log.info("使用 Autotrace (Centerline) 進行向量化...")
            # Autotrace 暫存 TGA
            tga_path = Path("temp.tga")
            Image.open(bmp_path).save(tga_path)
            cmd = f"autotrace -output-file {svg_path} --output-format svg --centerline {tga_path}"
            subprocess.run(cmd.split(), check=True)
            tga_path.unlink(missing_ok=True)
        else:
            log.info("使用 Potrace (Outline) 進行向量化...")
            cmd = f"potrace -b svg -o {svg_path} -n {bmp_path}"
            subprocess.run(cmd.split(), check=True)
        
        return svg_path

    def parse_svg(self, svg_path: Path):
        """
        解析 SVG 並將座標提取為 Numpy Arrays
        """
        paths, attributes, svg_attributes = svg2paths2(str(svg_path))
        
        # 統一使用經過前處理（包括旋轉）的影像維度
        if self.bw_image is None:
            log.error("未找到二值化影像，無法解析 SVG 座標基準")
            return
            
        w, h = self.bw_image.size
        offset_x = w / 2.0 if self.config.center else 0
        offset_y = h / 2.0 if self.config.center else 0
        
        extracted_paths = []
        for path in paths:
            sub_paths_points = []
            current_points = []
            
            for i, segment in enumerate(path):
                # 偵測斷點 (Discontinuity check)
                if i > 0:
                    prev_end = path[i-1].end
                    # 如果端點不一致 (距離 > 1e-5)，則切斷為新路徑
                    if abs(segment.start - prev_end) > 1e-5:
                        if current_points:
                            sub_paths_points.append(current_points)
                        current_points = []
                
                # 採樣座標點
                if isinstance(segment, Line):
                    current_points.append([segment.start.real, segment.start.imag])
                elif isinstance(segment, (CubicBezier, QuadraticBezier, Arc)):
                    # 採樣精度提高至 10 以提升平滑度
                    for t in np.linspace(0, 1, 10, endpoint=False):
                        p = segment.point(t)
                        current_points.append([p.real, p.imag])
            # 加入最後一個點位並封裝子路徑
            if current_points:
                last_p = path[-1].end
                current_points.append([last_p.real, last_p.imag])
                sub_paths_points.append(current_points)
            
            # 對每個子路徑進行座標變換與簡化
            for pts in sub_paths_points:
                # 轉換為 Numpy 並應用比例與偏移 (標準 WYSIWYG 映射: X 向右, Y 向上)
                # Potrace 座標是 10 倍像素 (0.1pt)
                pts_arr = np.array(pts) / 10.0
                
                # 座標變換公式: X = (px - W/2), Y = (py - H/2)
                # 注意: Potrace 的 y 座標是從下往上，而影像像素 y 是從上往下
                transformed_pts = np.zeros_like(pts_arr)
                transformed_pts[:, 0] = (pts_arr[:, 0] - offset_x) * self.config.mmperpixel
                transformed_pts[:, 1] = (pts_arr[:, 1] - offset_y) * self.config.mmperpixel
                
                # 簡化路徑 (使用 simplification 庫)
                if self.config.simplify > 0:
                    transformed_pts = simplify_coords(transformed_pts, self.config.simplify)
                
                if len(transformed_pts) >= self.config.minpath:
                    extracted_paths.append(transformed_pts)
                    
        self.paths_data = extracted_paths

    def optimize_paths(self):
        """
        執行提筆路徑優化
        """
        if self.config.minimize and len(self.paths_data) > 1:
            self.paths_data = minimize_moves_fast(self.paths_data)

    def save_results(self):
        """
        產出 G-Code 與 SVG 預覽
        """
        output_dir = self.config.output_path
        output_dir.mkdir(parents=True, exist_ok=True)
        
        gcode_path = output_dir / "image.gc"
        log.info(f"正在寫入 G-Code 至: {gcode_path}")
        
        with open(gcode_path, "w") as f:
            # 主軸啟動 (M3) 與設定轉速 (S)
            f.write(f"M3 S{self.config.spindle_speed}\n")
            # 確保一開始是提筆狀態
            f.write(f"G01 Z{self.config.penup} F{self.config.rapid_feedrate}\n") 
            for path in self.paths_data:
                # 1. 先提筆再移動 (避免橫向位移時刮到工件)
                f.write(f"G01 Z{self.config.penup} F{self.config.rapid_feedrate}\n")
                # 2. 快速移動至路徑起點 (XY 平面移動)
                f.write(f"G01 X{path[0][0]:.3f} Y{path[0][1]:.3f}\n")
                # 3. 落筆與工作進給率 (Z 軸向進入)
                f.write(f"G01 Z{self.config.pendown} F{self.config.feedrate}\n")
                # 繪製路徑
                for pt in path[1:]:
                    f.write(f"G01 X{pt[0]:.3f} Y{pt[1]:.3f}\n")
                # 4. 每個路徑結束後提筆
                f.write(f"G01 Z{self.config.penup} F{self.config.rapid_feedrate}\n")
            # 停止主軸 (M5)
            f.write("M05\n")
        
        # 建立簡化的 SVG 預覽檔
        svg_preview = output_dir / "final.svg"
        log.info(f"產出 SVG 預覽: {svg_preview}")
        with open(svg_preview, "w") as f:
            f.write(f'<?xml version="1.0" encoding="utf-8" ?>\n')
            f.write(f'<svg xmlns="http://www.w3.org/2000/svg" width="100%" height="100%">\n')
            for path in self.paths_data:
                points_str = " ".join([f"{pt[0]},{pt[1]}" for pt in path])
                f.write(f'  <polyline points="{points_str}" fill="none" stroke="black" stroke-width="0.5" />\n')
            f.write("</svg>")

    def generate_hatch_paths(self):
        """
        生成區域填充 (Hatch/Fill) 路徑 - 像素取樣法 (Pixel-level Sampling)
        """
        if self.config.hatch <= 0 or self.bw_image is None:
            return
            
        log.info(f"正在生成區域填充 (Hatch)... 間距: {self.config.hatch}mm")
        
        w, h = self.bw_image.size
        # 計算 X/Y 軸的物理長度
        scale = self.config.mmperpixel
        
        # 填充線在像素空間的間距
        hatch_px = self.config.hatch / scale
        
        offset_x = w / 2.0 if self.config.center else 0
        offset_y = h / 2.0 if self.config.center else 0
        
        hatch_paths = []
        
        # 每隔 hatch_px 掃描一條水平線
        for i, py in enumerate(np.arange(0, h, hatch_px)):
            y_idx = int(py)
            if y_idx >= h: break
            
            row = np.array(self.bw_image.crop((0, y_idx, w, y_idx + 1)))[0]
            
            line_segments = []
            start_x = None
            for x in range(w):
                is_black = (row[x] < 128) 
                if is_black and start_x is None:
                    start_x = x
                elif not is_black and start_x is not None:
                    line_segments.append((start_x, x - 1))
                    start_x = None
            if start_x is not None:
                line_segments.append((start_x, w - 1))
            
            # 為了效率，奇數行反轉掃描方向 (Zig-Zag)
            if i % 2 == 1:
                line_segments.reverse()
                
            for x1, x2 in line_segments:
                # 奇數行內部的起點終點也要反轉
                if i % 2 == 1:
                    x1, x2 = x2, x1
                hatch_paths.append(self._create_hatch_segment(x1, x2, py, offset_x, offset_y))

        log.info(f"填充完成，新增了 {len(hatch_paths)} 條掃描線段。")
        self.paths_data.extend(hatch_paths)

    def _create_hatch_segment(self, x1, x2, py, offset_x, offset_y):
        """
        將像素座標段轉換為 CNC 座標段 (與 parse_svg 同步: X = x - offset_x, Y = offset_y - y)
        """
        cx1 = (x1 - offset_x) * self.config.mmperpixel
        cx2 = (x2 - offset_x) * self.config.mmperpixel
        cy = (offset_y - py) * self.config.mmperpixel
        
        return np.array([[cx1, cy], [cx2, cy]])

    def run_all(self):
        """
        執行完整流程
        """
        bmp = self.preprocess_image()
        svg = self.vectorize(bmp)
        self.parse_svg(svg)
        self.generate_hatch_paths() # 新增填充步驟
        self.optimize_paths()
        self.save_results()
        
        # 清理暫存檔
        bmp.unlink(missing_ok=True)
        svg.unlink(missing_ok=True)
        log.success("轉檔完成！")
