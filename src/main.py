"""
影像轉 G-Code CLI 工具入口點

解析命令列參數並啟動核心處理引擎。
"""

import argparse
import sys
from loguru import logger as log

from config import AppConfig
from utils.logger import setup_logger
from processor import VectorProcessor


def main() -> None:
    """
    CLI 主函式：解析參數、建立配置、啟動處理流程
    """
    parser = argparse.ArgumentParser(description="影像轉 G-Code (Ink) 核心工具")

    # 必填參數
    parser.add_argument("--file", required=True, help="輸入影像檔案路徑")

    # 選填參數
    parser.add_argument("--folder", default=".", help="輸出目錄 (預設為當前目錄)")
    parser.add_argument("--autotrace", action="store_true", help="使用 Autotrace 中心線轉檔")
    parser.add_argument("--threshold", type=float, default=60.0, help="二值化門檻 (0-100)")
    parser.add_argument("--simplify", type=float, default=0.1, help="路徑簡化程度 (RDP)")
    parser.add_argument("--mmperpixel", type=float, default=0.1, help="每像素對應之公釐數 (mm)")
    parser.add_argument("--center", action="store_true", default=True, help="是否將圖片置中於原點")
    parser.add_argument("--no-center", action="store_false", dest="center", help="停用置中功能")
    parser.add_argument("--rotate", type=float, default=0.0, help="入徑旋轉角度")
    parser.add_argument("--penup", type=float, default=3.0, help="提筆安全高度")
    parser.add_argument("--pendown", type=float, default=0.0, help="下筆雕刻深度")
    parser.add_argument("--feedrate", type=float, default=1200.0, help="工作進給率 (mm/min)")
    parser.add_argument("--rapid-feedrate", type=float, default=3000.0, help="快速移動進給率 (mm/min)")
    parser.add_argument("--spindle-speed", type=int, default=1000, help="主軸轉速 (S)")
    parser.add_argument("--invert", action="store_true", help="執行圖片反相 (白色部分視為實心)")
    parser.add_argument("--hatch", type=float, default=0.0, help="區域填充線間距 (mm)，0=不填充")
    parser.add_argument("--minpath", type=int, default=0, help="最小路徑長度 (濾除雜訊)")
    parser.add_argument("--debug", action="store_true", help="開啟 Debug 日誌模式")

    args = parser.parse_args()

    # 初始化日誌
    setup_logger(level="DEBUG" if args.debug else "INFO")

    try:
        # 直接建構 AppConfig（不使用全域單例）
        cfg = AppConfig(**vars(args))
        log.info(f"啟動轉檔任務: {cfg.file}")

        processor = VectorProcessor(cfg)
        processor.run_all()

    except Exception as e:
        log.exception(f"發生未預期錯誤: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
