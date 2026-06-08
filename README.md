# ⚡ SynCapture

<p align="center">
  <a href="#-english">English</a> •
  <a href="#-简体中文">简体中文</a>
</p>

---

## 🌐 English

**SynCapture** is a Streamlit-based synaptic event analysis tool tailored for whole-cell patch-clamp electrophysiology data. It supports automated event detection, manual review, and statistical exports of `.abf` files.

### Key Features

- 📂 **Multi-file Batch Processing**: Simultaneously upload and switch between multiple `.abf` files.
- 🔍 **Automated Event Detection**: High-performance mEPSC/mIPSC detection powered by `scipy.signal.find_peaks`, and **Action Potential** detection & feature extraction using the `eFEL` library.
- 📊 **Interactive WebGL Plots**: Rendered via Plotly WebGL (`go.Scattergl`), supporting smooth dragging, scrolling, custom zooming, panning, and instant double-click resetting.
- ✅ **Manual Review & Annotation**: Curate individual detected events by accepting or rejecting them directly in an editable event table.
- 📁 **Unified Exports**: Download all exports in a single structured ZIP file containing Prism-ready CSVs, high-quality vector trace & summary plots (PDF & SVG), and global review states.
- 🏷️ **Metadata Labeling**: Easily group and organize recordings by `group`, `individual`, and `cell_id` directly in the UI.

### Installation

#### 1. Clone the Repository

```bash
git clone https://github.com/JingjingCheng/syncapture.git
cd syncapture
```

#### 2. Install Dependencies

It is highly recommended to use a virtual environment (e.g., Conda or venv):

```bash
# Using conda
conda create -n syncapture python=3.10
conda activate syncapture

# Using venv
python -m venv .venv
source .venv/bin/activate  # macOS/Linux
# .venv\Scripts\activate   # Windows
```

Install the required packages using `requirements.txt`:

```bash
pip install -r requirements.txt
```

### Usage

Start the Streamlit application:

```bash
streamlit run main.py
```

The browser should automatically open `http://localhost:8501`.

#### Cloud Deployment (Optional)

You can deploy this application for free on [Streamlit Community Cloud](https://share.streamlit.io/):
1. Push this repository to your GitHub account.
2. Log in to Streamlit Community Cloud and click **"New app"**.
3. Select this repository, branch `main`, and main file path `main.py`.
4. Click **"Deploy!"**.

#### Step-by-Step Workflow

1. **Upload Files**: Click **"Upload .abf files"** in the sidebar to upload one or more `.abf` files.
2. **Configure Parameters**:
   - **Sweep / Time Window**: Select the sweep channel and specify the analysis start/end time window.
   - **LP Filter**: Apply a lowpass Butterworth filter (Hz). Enter `0` to disable filtering.
   - **Detection Params**: Adjust the peak amplitude threshold (pA) and minimum event spacing (ms).
3. **Analyze & Interact**: The main dashboard displays an interactive, high-fidelity trace plot with detected event markers. Drag to zoom in on specific regions, scroll to zoom, and double-click to reset the view.
4. **Curate Events**: In the event table, double-click or select checkboxes under the `accepted` column to manually include or exclude events. The interactive chart will update instantly.
5. **Add Metadata**: Set custom metadata (Group, Individual, Cell ID) under the **"Cell Labels"** panel.
6. **Export Data**: Expand the **"Summary & Export"** panel at the bottom to download a compiled ZIP bundle containing all results, charts, and Prism-friendly data.

### Dependencies

| Package | Purpose |
|---|---|
| `streamlit` | Modern Web UI framework |
| `pyabf` | High-fidelity reading of Axon Binary Format (.abf) files |
| `scipy` | Digital filtering (Butterworth) and peak detection |
| `pandas` | Structured tabular data processing |
| `matplotlib` | High-quality vector rendering (PDF/SVG) for trace & summary exports |
| `numpy` | High-performance numerical computations |
| `plotly` | Interactive WebGL-accelerated chart plotting |
| `efel` | Electrophys Feature Extraction Library (eFEL) for action potential features |

### Project Structure

```
syncapture/
├── main.py              # Main application logic and UI
├── requirements.txt     # Python package requirements
├── .gitignore           # Git ignore list
└── README.md            # Project documentation (this file)
```

---

## 🇨🇳 简体中文

**SynCapture** 是一款基于 Streamlit 的突触事件分析工具，专为全细胞膜片钳电生理（whole-cell patch-clamp）数据设计，支持对 `.abf` 文件进行自动事件检测、人工审核与统计导出。

### 功能特性

- 📂 **多文件批量处理**：支持同时上传多个 `.abf` 文件，并在文件之间便捷切换。
- 🔍 **自动事件检测**：基于 `scipy.signal.find_peaks` 的高性能 mEPSC/mIPSC 自动检测算法，以及基于 `eFEL` 库的 **动作电位（Action Potential）** 自动检测与特征提取。
- 📊 **交互式 WebGL 图表**：基于 Plotly WebGL (`go.Scattergl`) 渲染，支持极速拖拽缩放、滚轮缩放、双击重置和悬停提示。
- ✅ **人工审核**：可在编辑表格中直接勾选或修改每个检测事件的 `accepted` 状态，图表数据实时更新。
- 📁 **一键整合导出**：支持下载一键打包的 ZIP 文件，内含 Prism 格式 CSV、高品质矢量图（PDF & SVG，包含 trace 轨迹图与统计图）以及全局审核状态。
- 🏷️ **元数据标注**：可在 UI 中为每组记录设置组别（group）、个体（individual）和细胞 ID（cell_id）。

### 安装说明

#### 1. 克隆仓库

```bash
git clone https://github.com/JingjingCheng/syncapture.git
cd syncapture
```

#### 2. 安装依赖环境

推荐使用虚拟环境进行管理（如 Conda 或 venv）：

```bash
# 使用 conda
conda create -n syncapture python=3.10
conda activate syncapture

# 使用 venv
python -m venv .venv
source .venv/bin/activate  # macOS/Linux
# .venv\Scripts\activate   # Windows
```

通过 `requirements.txt` 安装所有依赖包：

```bash
pip install -r requirements.txt
```

### 使用指南

启动 Streamlit 应用程序：

```bash
streamlit run main.py
```

浏览器将自动打开 `http://localhost:8501`。

#### 云端部署（可选）

您可以在 [Streamlit Community Cloud](https://share.streamlit.io/) 上免费部署此应用：
1. 将此仓库推送至您的 GitHub 账号。
2. 登录 Streamlit Community Cloud 并点击 **"New app"**。
3. 选择此仓库、分支 `main`，并将主入口文件设为 `main.py`。
4. 点击 **"Deploy!"** 即可完成部署并获取专属链接。

#### 详细操作流程

1. **上传文件**：在左侧侧边栏中点击 **"Upload .abf files"** 选择一个或多个 `.abf` 电生理文件。
2. **配置参数**：
   - **Sweep / Time Window**：选择要分析的 Sweep 轨道和时间范围。
   - **LP Filter**：设置 Butterworth 低通滤波截止频率 (Hz)，输入 `0` 则不启用滤波。
   - **Detection Params**：调整事件检测的幅度阈值 (pA) 和最小事件间距 (ms)。
3. **交互分析**：主面板展示高精度的交互式轨迹图。可在图表上框选放大特定波形，或双击重置视图。
4. **审核标注**：在事件列表中直接双击修改 `accepted` 字段以排除噪点事件。
5. **添加标签**：在 **"Cell Labels"** 折叠面板下为当前记录配置 Group（组别）、Individual（个体）及 Cell ID。
6. **导出数据**：展开底部的 **"Summary & Export"** 面板一键下载全部数据的 ZIP 打包文件。

### 依赖清单

| 依赖包 | 主要用途 |
|---|---|
| `streamlit` | 响应式 Web 界面框架 |
| `pyabf` | 读取 Axon 二进制格式 (.abf) 电生理数据 |
| `scipy` | 数字信号滤波（Butterworth）与波峰检测 |
| `pandas` | 结构化表格数据读取与处理 |
| `matplotlib` | 矢量 trace 与 summary 导出渲染 (PDF/SVG) |
| `numpy` | 矩阵与高性能数值计算 |
| `plotly` | 交互式 WebGL 加速图表绘制 |
| `efel` | 细胞电生理特征提取库 (eFEL)，用于提取动作电位特征 |

### 项目结构

```
syncapture/
├── main.py              # 主应用程序代码与界面
├── requirements.txt     # Python 依赖清单
├── .gitignore           # Git 忽略配置
└── README.md            # 项目说明文档（本文件）
```
