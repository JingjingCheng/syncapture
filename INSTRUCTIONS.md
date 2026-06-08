# 📖 SynCapture: User Guide & Instructions

Welcome to **SynCapture**! This guide is designed to help new users quickly understand how to prepare electrophysiology data, run the analysis tool, customize event detection parameters, curate results, and export high-quality reports.

---

## 📌 Table of Contents
1. [⚙️ Installation & Setup](#-installation--setup)
2. [📂 Preparing Your Data](#-preparing-your-data)
3. [🚀 Step-by-Step Workflow](#-step-by-step-workflow)
   - [Step 1: Upload Files](#step-1-upload-files)
   - [Step 2: Choose Sweep & Filtering](#step-2-choose-sweep--filtering)
   - [Step 3: Event Detection Modes](#step-3-event-detection-modes)
   - [Step 4: Interactive Trace Inspection](#step-4-interactive-trace-inspection)
   - [Step 5: Event Curation (Table & Chart)](#step-5-event-curation-table--chart)
   - [Step 6: Metadata & Exporting](#step-6-metadata--exporting)
4. [💡 Pro-Tips & Troubleshooting](#-pro-tips--troubleshooting)

---

## ⚙️ Installation & Setup

If you are running SynCapture locally for the first time:

1. **Clone the repository & enter the folder**:
   ```bash
   git clone https://github.com/JingjingCheng/syncapture.git
   cd syncapture
   ```
2. **Create and activate a virtual environment**:
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # macOS / Linux
   # .venv\Scripts\activate   # Windows
   ```
3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```
4. **Run the app**:
   ```bash
   streamlit run main.py
   ```
   *Your browser will open to `http://localhost:8501` automatically.*

---

## 📂 Preparing Your Data

SynCapture reads standard **Axon Binary Format (.abf)** files recorded using Clampex/pCLAMP.
* **Compatibility**: Both ABF1 and ABF2 formats are supported via the `pyabf` parser.
* **File Naming**: Avoid spaces or special characters in filenames if you plan to import files into scripting pipelines later.
* **Units**: Make sure your sweep channels have clear units (e.g. `pA` for current, `mV` for voltage) so SynCapture can scale and label the axes properly.

---

## 🚀 Step-by-Step Workflow

### Step 1: Upload Files
* In the left sidebar, click the **"📂 Upload ABF Files"** panel.
* Drag and drop one or more `.abf` files.
* Once loaded, select the active file you want to analyze from the **"📂 File"** dropdown.

---

### Step 2: Choose Sweep & Filtering
* **Sweep**: Electrophysiology recordings often contain multiple sweeps. Select the specific sweep channel to analyze.
* **Low-pass Filter (Hz)**: If your trace is noisy, enter a low-pass filter frequency (e.g., `1000` Hz).
  * *Tip: Enter `0` to disable digital filtering and view the raw signal.*
* **Time Range**: Specify the **Start (s)** and **End (s)** time to isolate the region of interest.

---

### Step 3: Event Detection Modes

SynCapture has two distinct detection modes selected via the **Direction** dropdown in the sidebar:

#### A. Synaptic Event Mode (`inward (EPSC)` or `outward (IPSC)`)
Used for analyzing miniature/spontaneous excitatory or inhibitory postsynaptic currents.
* **Prominence (pA)**: The minimum height a peak must stand out above its local background noise to be counted. Increase this if too many noise peaks are detected.
* **Min IEI (ms)**: The minimum inter-event interval allowed between consecutive peaks (prevents double-counting single events).
* **Tau Rise / Decay (ms)**: Template constants for the synaptic waveform kinetics.

#### B. Action Potential (AP) Mode (`Action Potential`)
Used for analyzing spikes, firing rates, and waveform properties.
* **AP Threshold (mV)**: The minimum membrane voltage a spike must exceed (e.g., `-20 mV`).
* **AP Prominence (mV)**: The minimum height of the action potential spike relative to its base.
* **AP Min Width / Max Width (ms)**: Rejects events that are too narrow (noise spikes) or too wide (artifacts).
* **AP Min Dist (ms)**: Minimum duration allowed between two consecutive spikes.

*Click the **"⚡ Detect"** button to run the algorithm.*

---

### Step 4: Interactive Trace Inspection

The main dashboard renders an interactive WebGL-accelerated chart:
* **Zooming**: Click and drag a box over any region to zoom in.
* **Scrolling**: Use your mouse wheel or trackpad scroll gesture to zoom in and out dynamically.
* **Hovering**: Hover over markers or traces to read precise values (e.g., amplitude and timing).
* **Resetting**: Double-click anywhere on the chart area to reset zoom and return to the default range.

---

### Step 5: Event Curation (Table & Chart)

Automatic detection is rarely 100% perfect. SynCapture makes curation easy:
1. **Exclude/Reject an Event**:
   * Click any event marker directly on the interactive chart.
   * Or, uncheck the box under the `accepted` column in the **Event Curation Table** below the chart.
   * *Rejected events turn grey and are excluded from statistics.*
2. **Restore/Include an Event**:
   * Click any grey `x` marker on the chart to re-include it.
   * Or, check its `accepted` box in the curation table.
3. **Add Manual Events**:
   * If the algorithm missed a real event, double-click on the trace in the chart to place a manual event marker.

---

### Step 6: Metadata & Exporting

1. **Cell Labels**: Specify labels in the sidebar panel:
   * **Cell ID**: Unique identifier for the recorded cell (defaults to filename).
   * **Individual**: Animal ID or subject identifier.
   * **Group**: Experimental condition (e.g., `control`, `drug`, `naive`).
   * **Status**: Overall quality state (`accepted`, `needs_check`, or `rejected`).
2. **Save to Dataset**: Click **"✓ Save to dataset"** to store the current curation and labels in session memory.
3. **Export Package**: Expand the **"Summary & Export"** panel at the bottom to download a `.zip` bundle.

#### 📂 What's inside the Export ZIP?
* `per_cell_summary.csv`: Firing frequency, amplitudes, decay times, and metadata grouped per cell.
* `group_mean_sem.csv`: Compiled summary statistics ready for graphing.
* `Prism_amplitude_pA.csv` & `Prism_frequency_Hz.csv`: Columnar tables formatted specifically for copy-pasting directly into **GraphPad Prism**.
* `all_events.csv`: Raw, unaggregated data of every single event across all analyzed cells.
* `figures/`: Global statistics summary figures in vector formats (**PDF** and **SVG**).
* `traces/`: High-resolution vector traces of every analyzed sweep showing all detected and manual event markers.

---

## 💡 Pro-Tips & Troubleshooting

> [!TIP]
> **Autoplay/Autoscale Reset**
> If your Y-axes range looks squeezed or distorted after changing directions (e.g., switching from EPSC to AP mode), click the **"Reset Y to data"** button in the sidebar to autoscale the viewport instantly.

> [!IMPORTANT]
> **Saving Changes**
> Always click **"✓ Save to dataset"** before switching to another file, otherwise your manual event edits and cell metadata labels for the active file will be lost!

> [!NOTE]
> **Streamlit Cloud Deployment**
> If you deploy SynCapture on Streamlit Cloud, any custom QR code image uploaded through the donation dialog will only remain active until the cloud container recycles. To make it permanent, save your image as `donate.jpg` or `donate.png` in your local project root folder, commit, and push it to your GitHub repository.
