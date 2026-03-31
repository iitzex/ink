from pydantic_settings import BaseSettings, SettingsConfigDict
from pathlib import Path
from typing import Optional

class AppConfig(BaseSettings):
    """
    影像轉 G-Code 的全域參數管理
    """
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

    file: str
    folder: str = "."
    animate: bool = False
    overwrite: bool = True
    skeleton: bool = False
    autotrace: bool = False
    minimize: bool = True
    
    # 影像處理參數
    threshold: float = 60.0
    simplify: float = 0.1
    prune: int = 0
    minpath: int = 0
    merge: int = 0
    junctiondist: float = 0.0
    seed: int = 0

    # 座標與比例
    penup: float = 3.0
    pendown: float = 0.0
    mmperpixel: float = 0.1
    center: bool = True
    rotate: float = 0.0
    debug: bool = False
    
    # 進給率 (Feedrate) mm/min
    feedrate: float = 1200.0
    rapid_feedrate: float = 3000.0
    spindle_speed: int = 1000
    invert: bool = False
    hatch: float = 0.0

    @property
    def output_path(self) -> Path:
        """
        計算輸出的子目錄路徑
        """
        filename = Path(self.file).name
        return Path(self.folder) / f"{filename}.img2gcode"

config = None

def get_config(**kwargs) -> AppConfig:
    global config
    if config is None or kwargs:
        config = AppConfig(**kwargs)
    return config
