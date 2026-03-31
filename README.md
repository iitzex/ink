# Ink-Convert: 工業級影像轉 G-Code 引擎

本專案是一個基於 Python 的專業影像處理工具，旨在將一般影像（PNG/JPG/BMP）轉換為適用於 CNC 雕刻機與繪圖機的高品質 G-Code。本工具採用現代化架構，整合了路徑優化、區域填充與精確的物理座標映射。

## 核心特性 (Key Features)

### 1. 專業路徑優化 (Path Optimization)

- 🚀 **提筆最小化 (Rapid-Move Optimization)**：利用 `KDTree` 空間索引進行 $O(N \log N)$ 鄰近點搜尋，自動重排雕刻順序，大幅減少空走時間。
- 📐 **子路徑斷點偵測**：精確識別 SVG 內的跳轉點，確保在不同物件（如文字、複雜邊框）之間移動時正確執行「提筆」動作，徹底杜絕表面刮痕。

### 2. 智能填充與像素分析 (Hatching & Pixel Sensing)

- 🏁 **像素級區域填充 (Pixel-level Hatching)**：捨棄傳統幾何運算，直接讀取影像像素 RGB/Alpha 值，實現「黑刻、白不刻」的 100% 精準度。
- ⚡ **之字形填充 (Zig-Zag)**：填充線採用連續往復路徑，減少 80% 的 Z 軸升降動作，提升效率並保護主軸馬達。
- 🖼️ **Alpha 通道平整化**：自動處理 RGBA/LA 影像，將透明背景轉換為物理白底，防止轉檔發黑。

### 3. 工業級指令控制 (CNC Control)

- 🔌 **主軸與進給率分離**：支援 `M3/M5` 主軸開關、進給率 (Work Feedrate) 與 快速移動 (Rapid Feedrate) 的獨立設定。
- 🔄 **所見即所得 (WYSIWYG)**：修正了傳統工具常見的座標鏡射問題，產出的 G-Code 方向與原始圖片完全一致。
- 🌪️ **貝茲採樣優化**：高精度曲線採樣，確保圓弧邊緣流暢平滑。

## 快速開始 (Quick Start)

### 安裝環境 (uv)

本專案建議使用 `uv` 進行隔離環境管理：

```bash
# 同步開發環境
uv sync
```

### 系統必備

- `potrace`: 用於輪廓向量化。
- `autotrace`: 用於中心線提取。

### 轉檔範例

```bash
# 1. 基本轉檔 (10px = 1mm, 中心對齊)
uv run python src/main.py --file image.png --mmperpixel 0.1

# 2. 實心區域填充 (0.5mm 間距)
uv run python src/main.py --file logo.png --hatch 0.5

# 3. 調整主軸轉速與進給率
uv run python src/main.py --file house.png --spindle-speed 1200 --feedrate 2000

# 4. 反相雕刻 (刻掉圖片白色部分)
uv run python src/main.py --file photo.jpg --invert
```

## CLI 參數說明 (Arguments)

| 參數               | 預設值  | 說明                               |
| :----------------- | :------ | :--------------------------------- |
| `--file`           | (必填)  | 輸入影像路徑                       |
| `--hatch`          | `0.0`   | 填充間距 (mm)，大於 0 啟動填充模式 |
| `--mmperpixel`     | `0.1`   | 物理比例 (1 像素對應多少 mm)       |
| `--invert`         | `False` | 是否執行顏色反相                   |
| `--rotate`         | `0.0`   | 座標旋轉角度 (0-360)               |
| `--feedrate`       | `1200`  | 雕刻時的進給速度 (mm/min)          |
| `--rapid-feedrate` | `3000`  | 提筆空走時的移動速度               |
| `--penup`          | `3.0`   | 提筆高度 (mm)                      |
| `--pendown`        | `0.0`   | 雕刻高度 (mm)                      |
| `--spindle-speed`  | `1000`  | 主軸轉速 (S 指令)                  |

## 目錄結構

```text
ink/
├── src/
│   ├── main.py          # 入口
│   ├── config.py        # Pydantic BaseSettings
│   ├── processor.py     # 核心引擎
│   └── utils/
│       ├── path_opt.py  # KDTree 優化邏輯
│       └── logger.py    # Loguru 日誌組態
├── pyproject.toml       # uv 配置與依賴清單
└── README.md            # 本文件
```
