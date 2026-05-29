#!/usr/bin/env python3
"""
SynCapture — Synaptic Event Analysis Tool
Run: streamlit run syncapture.py
Dependencies: pip install streamlit pyabf scipy pandas matplotlib numpy plotly
"""
import io, json, shutil, zipfile, tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.signal import find_peaks, butter, filtfilt, fftconvolve
from scipy.ndimage import gaussian_filter1d
import streamlit as st
import streamlit.components.v1 as components
import plotly
import plotly.graph_objects as go

try:
    import pyabf
    HAS_PYABF = True
except Exception:
    HAS_PYABF = False

st.set_page_config(page_title='SynCapture', page_icon='⚡', layout='wide', initial_sidebar_state='expanded')

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300..700&display=swap');
html, body, [class*="css"] { font-family: 'Inter', sans-serif !important; }
:root {
  --bg:#ffffff; --surface:#f8f9fa; --border:rgba(0,0,0,0.08); --accent:#1a6b55;
  --accent-l:#eaf2ef; --text:#1a1a1a; --muted:#6b7280;
}
header[data-testid="stHeader"] { display: none !important; height: 0 !important; }
[data-testid="stAppViewContainer"] { top: 0 !important; margin-top: 0 !important; }
[data-testid="stMain"] > .block-container,
[data-testid="stMainBlockContainer"] { padding: 0.5rem 1.5rem 1rem 1.5rem !important; margin-top: 0 !important; max-width: 100% !important; }
section[data-testid="stSidebar"] { min-width: 300px !important; max-width: 320px !important; }
section[data-testid="stSidebar"] [data-testid="stVerticalBlockBorderWrapper"] { padding: 0; }
section[data-testid="stSidebar"] .block-container { padding-top: 0 !important; padding-bottom: 0.3rem !important; }
section[data-testid="stSidebar"] > div { padding-top: 0 !important; }
section[data-testid="stSidebar"] [data-testid="stSidebarContent"] { padding-top: 0.4rem !important; }
section[data-testid="stSidebar"] [data-testid="stSidebarUserContent"] { padding-top: 0 !important; }
section[data-testid="stSidebar"] [data-testid="stSidebarContent"] > div:first-child { margin-top: 0 !important; padding-top: 0 !important; }
section[data-testid="stSidebar"] [data-testid="stSidebarHeader"] { display: none !important; }
section[data-testid="stSidebar"] [data-testid="stSidebarUserContent"] { padding-top: 0 !important; }
/* Compact sidebar: reduce gaps between all widgets */
section[data-testid="stSidebar"] [data-testid="stVerticalBlock"] { gap: 0.35rem !important; }
section[data-testid="stSidebar"] hr { margin: 0.2rem 0 !important; }
section[data-testid="stSidebar"] .stSelectbox label,
section[data-testid="stSidebar"] .stNumberInput label,
section[data-testid="stSidebar"] .stSlider label,
section[data-testid="stSidebar"] .stFileUploader label,
section[data-testid="stSidebar"] .stTextInput label { font-size: 0.78rem !important; margin-bottom: 0 !important; }
section[data-testid="stSidebar"] .stSelectbox,
section[data-testid="stSidebar"] .stNumberInput,
section[data-testid="stSidebar"] .stSlider,
section[data-testid="stSidebar"] .stTextInput { margin-bottom: -0.3rem !important; }
section[data-testid="stSidebar"] [data-testid="stExpander"] { margin-top: 0 !important; margin-bottom: 0 !important; }
section[data-testid="stSidebar"] [data-testid="stExpander"] details { padding: 0 !important; }
/* Compact file uploader: just reduce extra padding */
section[data-testid="stSidebar"] .stFileUploader { margin-bottom: -0.2rem !important; }
/* Make the Plotly chart fill full width and available height */
[data-testid="stPlotlyChart"] { width: 100% !important; min-height: calc(100vh - 260px); }
[data-testid="stPlotlyChart"] > div { width: 100% !important; height: 100% !important; }
[data-testid="stPlotlyChart"] iframe { width: 100% !important; height: 100% !important; }
div[data-testid="stMetric"] { background: color-mix(in srgb, currentColor 6%, transparent); border:1px solid color-mix(in srgb, currentColor 12%, transparent); border-radius:8px; padding:0.5rem 0.7rem; }
.stButton > button { border-radius:6px; font-size:0.82rem; font-weight:500; }
.stButton > button[kind="primary"] { background:var(--accent); border:none; }
.stButton > button:hover { opacity:0.88; }
</style>
""", unsafe_allow_html=True)

def _init():
    defaults = {
        'files': {}, 'file_order': [], 'active': None, 'skipped': set(),
        'events': {}, 'settings': {}, 'records': [], 'custom_groups': [],
        'event_table_revisions': {},
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

_init()
S = st.session_state

DEFAULT_GROUP_OPTIONS = ['naive', 'ovx', 'control', 'treatment', 'other']
EVENT_COLUMNS = ['time_s', 'amplitude_pA', 'prominence', 'iei_s', 'accepted']

_PLOTLY_RELAYOUT_COMPONENT = None

def _plotly_relayout_component_html():
    return """
<!DOCTYPE html>
<html>
<head>
  <meta charset="UTF-8" />
  <script src="./plotly.min.js"></script>
  <style>
    html, body, #chart {
      width: 100%;
      height: 100%;
      margin: 0;
      padding: 0;
      overflow: hidden;
      background: white;
    }
    body {
      position: relative;
    }
    #chart-toast {
      position: absolute;
      left: 14px;
      bottom: 12px;
      z-index: 10;
      opacity: 0;
      transform: translateY(4px);
      transition: opacity 140ms ease, transform 140ms ease;
      border: 1px solid rgba(0,0,0,0.12);
      border-radius: 6px;
      background: rgba(255,255,255,0.94);
      color: #374151;
      font: 600 12px/1.2 Inter, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      padding: 7px 9px;
      pointer-events: none;
      box-shadow: 0 6px 18px rgba(0,0,0,0.08);
    }
    #chart-toast.visible {
      opacity: 1;
      transform: translateY(0);
    }
  </style>
</head>
<body>
  <div id="chart"></div>
  <div id="chart-toast"></div>
  <script>
    const chart = document.getElementById("chart");
    const chartToast = document.getElementById("chart-toast");
    let plotHandlersAttached = false;
    let lastPayload = "";
    let currentFileName = null;
    let currentFallbackRange = null;
    let toastTimer = null;

    function sendMessage(type, data) {
      window.parent.postMessage(
        Object.assign({ isStreamlitMessage: true, type: type }, data),
        "*"
      );
    }

    function setComponentReady() {
      sendMessage("streamlit:componentReady", { apiVersion: 1 });
    }

    function setFrameHeight(height) {
      sendMessage("streamlit:setFrameHeight", { height: height });
    }

    function setComponentValue(value) {
      sendMessage("streamlit:setComponentValue", {
        value: value,
        dataType: "json",
      });
    }

    function showToast(message) {
      chartToast.textContent = message;
      chartToast.classList.add("visible");
      if (toastTimer) {
        clearTimeout(toastTimer);
      }
      toastTimer = setTimeout(function() {
        chartToast.classList.remove("visible");
      }, 950);
    }

    function parseFigure(rawFigure) {
      if (typeof rawFigure === "string") {
        return JSON.parse(rawFigure);
      }
      return rawFigure || { data: [], layout: {} };
    }

    function numberOrNull(value) {
      const parsed = Number(value);
      return Number.isFinite(parsed) ? parsed : null;
    }

    function extractYRange(eventData, fallbackRange) {
      let yMin = numberOrNull(eventData["yaxis.range[0]"]);
      let yMax = numberOrNull(eventData["yaxis.range[1]"]);

      if ((yMin === null || yMax === null) && Array.isArray(eventData["yaxis.range"])) {
        yMin = numberOrNull(eventData["yaxis.range"][0]);
        yMax = numberOrNull(eventData["yaxis.range"][1]);
      }

      if ((yMin === null || yMax === null) && eventData["yaxis.autorange"] && Array.isArray(fallbackRange)) {
        yMin = numberOrNull(fallbackRange[0]);
        yMax = numberOrNull(fallbackRange[1]);
      }

      if (yMin === null || yMax === null || yMax <= yMin) {
        return null;
      }
      return { y_min: yMin, y_max: yMax };
    }

    function boolOrNull(value) {
      if (value === true || value === 1 || value === "1" || value === "true") {
        return true;
      }
      if (value === false || value === 0 || value === "0" || value === "false") {
        return false;
      }
      return null;
    }

    function pointMeta(point) {
      if (!point) {
        return null;
      }
      let customdata = Array.isArray(point.customdata) ? point.customdata : null;
      if (!customdata && point.data && Array.isArray(point.data.customdata)) {
        customdata = point.data.customdata[point.pointIndex];
      }
      let eventIndex = customdata ? Number(customdata[0]) : NaN;
      if ((!Number.isInteger(eventIndex) || eventIndex < 0) && point.id !== undefined) {
        eventIndex = Number(point.id);
      }
      if ((!Number.isInteger(eventIndex) || eventIndex < 0) && point.data && Array.isArray(point.data.ids)) {
        eventIndex = Number(point.data.ids[point.pointIndex]);
      }
      if (!Number.isInteger(eventIndex) || eventIndex < 0) {
        return null;
      }
      const accepted = customdata ? boolOrNull(customdata[1]) : null;
      return {
        event_index: eventIndex,
        accepted: accepted === null ? !(point.fullData && String(point.fullData.name || "").toLowerCase().includes("rejected")) : accepted,
        time_s: numberOrNull(point.x),
      };
    }

    function firstEventPoint(points) {
      const eventPoints = points || [];
      for (let i = 0; i < eventPoints.length; i += 1) {
        const meta = pointMeta(eventPoints[i]);
        if (meta) {
          return { point: eventPoints[i], meta: meta };
        }
      }
      return null;
    }

    function pointAction(meta, mouseEvent) {
      if (mouseEvent && (mouseEvent.shiftKey || mouseEvent.altKey)) {
        return "accept_event";
      }
      if (mouseEvent && (mouseEvent.ctrlKey || mouseEvent.metaKey)) {
        return "reject_event";
      }
      return "toggle_event";
    }

    function attachPlotHandlers() {
      if (plotHandlersAttached) {
        return;
      }
      plotHandlersAttached = true;

      chart.on("plotly_click", function(eventData) {
        const match = firstEventPoint(eventData && eventData.points);
        if (!match) {
          return;
        }
        const mouseEvent = eventData.event || {};
        const action = pointAction(match.meta, mouseEvent);
        const payload = {
          kind: "event_action",
          file_name: currentFileName,
          action: action,
          event_index: match.meta.event_index,
          time_s: match.meta.time_s,
          nonce: Date.now(),
        };
        const willAccept = action === "accept_event" || (action === "toggle_event" && match.meta.accepted === false);
        setComponentValue(payload);
        showToast(willAccept ? "Restored" : "Rejected");
        if (mouseEvent.preventDefault) {
          mouseEvent.preventDefault();
        }
      });

      chart.on("plotly_relayout", function(eventData) {
        const liveRange = chart._fullLayout && chart._fullLayout.yaxis && chart._fullLayout.yaxis.range;
        const range = extractYRange(eventData || {}, liveRange || currentFallbackRange);
        if (!range) {
          return;
        }
        const payload = {
          kind: "relayout",
          file_name: currentFileName,
          y_min: range.y_min,
          y_max: range.y_max,
          nonce: Date.now(),
        };
        const payloadKey = JSON.stringify({
          file_name: payload.file_name,
          y_min: payload.y_min,
          y_max: payload.y_max,
        });
        if (payloadKey === lastPayload) {
          return;
        }
        lastPayload = payloadKey;
        setComponentValue(payload);
      });
    }

    async function render(args) {
      const height = Number(args.height || 620);
      const figure = parseFigure(args.figure_json);
      const layout = Object.assign({}, figure.layout || {}, {
        height: height,
        autosize: true,
      });
      const config = Object.assign({ responsive: true }, args.config || {});
      const fallbackRange = layout.yaxis && layout.yaxis.range;
      currentFileName = args.file_name;
      currentFallbackRange = fallbackRange;

      await Plotly.react(chart, figure.data || [], layout, config);
      setFrameHeight(height);
      Plotly.Plots.resize(chart);
      attachPlotHandlers();
    }

    window.addEventListener("message", function(event) {
      if (!event.data || event.data.type !== "streamlit:render") {
        return;
      }
      render(event.data.args || {}).catch(function(error) {
        setComponentValue({
          error: error && error.message ? error.message : String(error),
          nonce: Date.now(),
        });
      });
    });

    window.addEventListener("resize", function() {
      Plotly.Plots.resize(chart);
    });

    setComponentReady();
  </script>
</body>
</html>
"""

def _get_plotly_relayout_component():
    global _PLOTLY_RELAYOUT_COMPONENT
    if _PLOTLY_RELAYOUT_COMPONENT is not None:
        return _PLOTLY_RELAYOUT_COMPONENT

    component_dir = Path(tempfile.gettempdir()) / 'syncapture_plotly_relayout_component'
    component_dir.mkdir(parents=True, exist_ok=True)
    html_path = component_dir / 'index.html'
    html = _plotly_relayout_component_html()
    if not html_path.exists() or html_path.read_text(encoding='utf-8') != html:
        html_path.write_text(html, encoding='utf-8')

    plotly_js_src = Path(plotly.__file__).parent / 'package_data' / 'plotly.min.js'
    plotly_js_dst = component_dir / 'plotly.min.js'
    if not plotly_js_dst.exists() or plotly_js_dst.stat().st_size != plotly_js_src.stat().st_size:
        shutil.copyfile(plotly_js_src, plotly_js_dst)

    _PLOTLY_RELAYOUT_COMPONENT = components.declare_component(
        'plotly_relayout_chart',
        path=str(component_dir),
    )
    return _PLOTLY_RELAYOUT_COMPONENT

def normalize_events_frame(events_df):
    if events_df is None or not isinstance(events_df, pd.DataFrame) or events_df.empty:
        return pd.DataFrame(columns=EVENT_COLUMNS)
    out = events_df.copy()
    for col in EVENT_COLUMNS:
        if col not in out.columns:
            out[col] = True if col == 'accepted' else np.nan
    for col in ['time_s', 'amplitude_pA', 'prominence', 'iei_s']:
        out[col] = pd.to_numeric(out[col], errors='coerce')
    out['accepted'] = out['accepted'].fillna(True).astype(bool)
    return out[EVENT_COLUMNS]

def events_frame_changed(before, after):
    before = normalize_events_frame(before).reset_index(drop=True)
    after = normalize_events_frame(after).reset_index(drop=True)
    if len(before) != len(after):
        return True
    if len(before) == 0:
        return False
    for col in ['time_s', 'amplitude_pA', 'prominence', 'iei_s']:
        left = before[col].to_numpy(dtype=float)
        right = after[col].to_numpy(dtype=float)
        if not np.allclose(left, right, equal_nan=True):
            return True
    return not np.array_equal(before['accepted'].to_numpy(dtype=bool), after['accepted'].to_numpy(dtype=bool))

def bump_event_table_revision(file_name):
    revisions = st.session_state.setdefault('event_table_revisions', {})
    revisions[file_name] = int(revisions.get(file_name, 0)) + 1

def apply_event_table_delta(file_name, editor_key, source_key):
    widget_state = st.session_state.get(editor_key)
    source = st.session_state.get(source_key)
    events_by_file = st.session_state.setdefault('events', {})
    all_events = events_by_file.get(file_name)
    if not isinstance(widget_state, dict) or not isinstance(source, pd.DataFrame):
        return
    if not isinstance(all_events, pd.DataFrame):
        all_events = pd.DataFrame(columns=EVENT_COLUMNS)

    edited = normalize_events_frame(source)
    edited_rows = widget_state.get('edited_rows', {}) or {}
    for row_key, changes in edited_rows.items():
        try:
            row_pos = int(row_key)
        except (TypeError, ValueError):
            continue
        if row_pos < 0 or row_pos >= len(edited):
            continue
        row_index = edited.index[row_pos]
        for col, value in (changes or {}).items():
            if col in EVENT_COLUMNS:
                edited.at[row_index, col] = value

    deleted_positions = []
    for row_key in widget_state.get('deleted_rows', []) or []:
        try:
            row_pos = int(row_key)
        except (TypeError, ValueError):
            continue
        if 0 <= row_pos < len(edited):
            deleted_positions.append(row_pos)
    if deleted_positions:
        edited = edited.drop(edited.index[deleted_positions])

    added_rows = widget_state.get('added_rows', []) or []
    if added_rows:
        added = normalize_events_frame(pd.DataFrame(added_rows))
        edited = pd.concat([edited, added], ignore_index=True)

    edited = normalize_events_frame(edited)
    outside = all_events.drop(index=source.index, errors='ignore')
    next_events = pd.concat([outside, edited], ignore_index=True).sort_values('time_s').reset_index(drop=True)
    if events_frame_changed(all_events, next_events):
        events_by_file[file_name] = next_events

def apply_chart_event_action(file_name, event_index, action):
    events_by_file = st.session_state.setdefault('events', {})
    events_df = events_by_file.get(file_name)
    if not isinstance(events_df, pd.DataFrame) or events_df.empty:
        return None
    try:
        event_index = int(event_index)
    except (TypeError, ValueError):
        return None
    if event_index not in events_df.index:
        return None

    events_df = normalize_events_frame(events_df)
    current = bool(events_df.at[event_index, 'accepted'])
    if action == 'accept_event':
        accepted = True
    elif action == 'reject_event':
        accepted = False
    elif action == 'toggle_event':
        accepted = not current
    else:
        return None

    if current == accepted:
        return {'changed': False, 'accepted': accepted}

    events_df.at[event_index, 'accepted'] = accepted
    events_by_file[file_name] = events_df
    bump_event_table_revision(file_name)
    st.session_state['_last_chart_event_action'] = {
        'file_name': file_name,
        'event_index': event_index,
        'accepted': accepted,
    }
    return {'changed': True, 'accepted': accepted}

def sync_plotly_chart_state():
    active = st.session_state.get('active')
    if not active:
        return
    event = st.session_state.get(f'plotly_chart_{active}')
    if not isinstance(event, dict):
        return
    if event.get('error'):
        return
    file_name = event.get('file_name') or active

    if event.get('kind') == 'event_action':
        action_key = json.dumps({
            'file_name': file_name,
            'action': event.get('action'),
            'event_index': event.get('event_index'),
            'nonce': event.get('nonce'),
        }, sort_keys=True)
        processed = st.session_state.setdefault('_processed_plotly_actions', {})
        if processed.get(file_name) == action_key:
            return
        processed[file_name] = action_key
        apply_chart_event_action(file_name, event.get('event_index'), event.get('action'))
        return

    y_min = event.get('y_min')
    y_max = event.get('y_max')
    if not is_valid_y_range(y_min, y_max):
        return
    settings = st.session_state.setdefault('settings', {})
    settings.setdefault(file_name, {})
    prev_min = settings[file_name].get('y_min')
    prev_max = settings[file_name].get('y_max')
    if prev_min == float(y_min) and prev_max == float(y_max):
        return
    reset_y_scale(file_name, y_min, y_max)

def plotly_relayout_chart(fig, file_name, config, height=620):
    component = _get_plotly_relayout_component()
    return component(
        figure_json=fig.to_json(),
        file_name=file_name,
        config=config,
        height=height,
        default=None,
        key=f'plotly_chart_{file_name}',
        on_change=sync_plotly_chart_state,
    )

def load_abf(uploaded_file):
    if not HAS_PYABF:
        raise RuntimeError('pyabf not found. Install: pip install pyabf')
    tmp_dir = Path(tempfile.gettempdir()) / 'syncapture'
    tmp_dir.mkdir(parents=True, exist_ok=True)
    tmp = tmp_dir / uploaded_file.name
    tmp.write_bytes(uploaded_file.getbuffer())
    abf = pyabf.ABF(str(tmp.resolve()))
    return abf, tmp

def abf_to_sweeps(abf):
    rows = []
    for sw in abf.sweepList:
        abf.setSweep(sw)
        rows.append(pd.DataFrame({'sweep': sw, 'time_s': abf.sweepX.copy(), 'signal': abf.sweepY.copy()}))
    return pd.concat(rows, ignore_index=True)

def gaussian_lowpass(data, cutoff_hz, fs_hz):
    """Applies a Gaussian filter converting the cutoff frequency to sigma."""
    nyq = 0.5 * fs_hz
    if cutoff_hz <= 0 or cutoff_hz >= nyq:
        return data
    # Calculate sigma in samples for a -3dB cutoff frequency
    sigma_samples = (0.1325 * fs_hz) / cutoff_hz
    return gaussian_filter1d(data, sigma=sigma_samples)

def create_mepsc_template(tau_rise_ms=0.5, tau_decay_ms=3.0, fs_hz=10000, length_ms=20):
    """Generates an idealized double-exponential mEPSC waveform."""
    t = np.arange(0, length_ms, 1000/fs_hz)
    template = (1 - np.exp(-t/tau_rise_ms)) * np.exp(-t/tau_decay_ms)
    return template / np.max(template)

def detect_synaptic_events(trace, time_s, direction, prominence, distance_ms, baseline_pct=20, tau_rise=0.5, tau_decay=3.0):
    # Baseline subtraction
    n_base = max(1, int(len(trace) * baseline_pct / 100))
    baseline = np.median(trace[:n_base])
    y = trace - baseline

    # Calculate sampling frequency
    dt = np.median(np.diff(time_s)) if len(time_s) > 1 else 0.0001
    fs_hz = 1.0 / dt

    # Generate reversed template for FFT cross-correlation
    template = create_mepsc_template(tau_rise_ms=tau_rise, tau_decay_ms=tau_decay, fs_hz=fs_hz)

    # NORMALIZE TEMPLATE SO AMPLITUDES REMAIN IN pA
    # The sum of the template must be 1, otherwise cross-correlation multiplies the signal size
    template_normalized = template / np.sum(template)
    template_reversed = template_normalized[::-1] 

    # Perform Deconvolution
    deconvolved_trace = fftconvolve(y, template_reversed, mode='same')

    # Invert signal if searching for inward currents
    y_detect = -deconvolved_trace if direction == 'inward (EPSC)' else deconvolved_trace

    # Find peaks on the deconvolved trace
    distance_pts = max(1, int((distance_ms / 1000) / dt))
    idx, props = find_peaks(y_detect, prominence=prominence, distance=distance_pts)

    if len(idx) == 0:
        return pd.DataFrame(columns=['time_s', 'amplitude_pA', 'prominence', 'iei_s', 'accepted'])

    # SHIFT COMPENSATION
    template_peak_idx = np.argmax(template_normalized)
    shift_offset = (len(template_normalized) // 2) - template_peak_idx

    corrected_idx = idx - shift_offset
    corrected_idx = np.clip(corrected_idx, 0, len(time_s) - 1)

    # Optional fine-tuning of the alignment
    search_window = max(1, int(1.0 / 1000 / dt)) # 1 ms window
    for i in range(len(corrected_idx)):
        win_start = max(0, corrected_idx[i] - search_window)
        win_end = min(len(y), corrected_idx[i] + search_window)
        if direction == 'inward (EPSC)':
            corrected_idx[i] = win_start + np.argmin(y[win_start:win_end])
        else:
            corrected_idx[i] = win_start + np.argmax(y[win_start:win_end])

    peak_times = time_s[corrected_idx]
    iei = np.diff(peak_times)
    iei = np.concatenate([[np.nan], iei])

    return pd.DataFrame({
        'time_s': peak_times,
        'amplitude_pA': y.iloc[corrected_idx] if isinstance(y, pd.Series) else y[corrected_idx],
        'prominence': props.get('prominences', np.full(len(idx), np.nan)),
        'iei_s': iei,
        'accepted': True,
    })

def summary_from_events(events_df, window_dur_s):
    acc = events_df[events_df['accepted'] == True] if not events_df.empty else events_df
    n = len(acc)
    freq_hz = n / window_dur_s if window_dur_s > 0 else np.nan
    amp_abs = acc['amplitude_pA'].abs() if n else pd.Series(dtype=float)
    return {
        'n_events': n,
        'freq_hz': freq_hz,
        'amp_mean_pA': amp_abs.mean() if n else np.nan,
        'amp_median_pA': amp_abs.median() if n else np.nan,
        'amp_sd_pA': amp_abs.std(ddof=1) if n > 1 else np.nan,
        'iei_mean_s': acc['iei_s'].mean() if n else np.nan,
    }

def json_safe(value):
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [json_safe(v) for v in value]
    if isinstance(value, np.generic):
        return json_safe(value.item())
    if value is pd.NA:
        return None
    if isinstance(value, float):
        return value if np.isfinite(value) else None
    return value

def infer_y_range(values, pad_frac=0.05):
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return -1.0, 1.0
    y_min = float(np.min(arr))
    y_max = float(np.max(arr))
    span = y_max - y_min
    pad = max(span * pad_frac, 1e-6) if span > 0 else max(abs(y_min) * pad_frac, 1.0)
    return y_min - pad, y_max + pad

def is_valid_y_range(y_min, y_max):
    try:
        y_min = float(y_min)
        y_max = float(y_max)
    except (TypeError, ValueError):
        return False
    return np.isfinite(y_min) and np.isfinite(y_max) and y_max > y_min

def reset_y_scale(file_name, y_min, y_max):
    settings = st.session_state.setdefault('settings', {})
    settings.setdefault(file_name, {})
    settings[file_name]['y_min'] = float(y_min)
    settings[file_name]['y_max'] = float(y_max)
    st.session_state[f'y_min_{file_name}'] = float(y_min)
    st.session_state[f'y_max_{file_name}'] = float(y_max)

def normalize_group_name(group):
    return group.strip() if isinstance(group, str) else ''

def remember_group_option(group):
    group = normalize_group_name(group)
    if not group:
        return ''
    custom_groups = st.session_state.setdefault('custom_groups', [])
    if group not in DEFAULT_GROUP_OPTIONS and group not in custom_groups:
        custom_groups.append(group)
    return group

def group_options_for(saved_group=None, current_group=None):
    options = list(DEFAULT_GROUP_OPTIONS)
    for rec in st.session_state.get('records', []):
        group = normalize_group_name(rec.get('group'))
        if group and group not in options:
            options.append(group)
    for group in st.session_state.get('custom_groups', []):
        group = normalize_group_name(group)
        if group and group not in options:
            options.append(group)
    for group in (saved_group, current_group):
        group = normalize_group_name(group)
        if group and group not in options:
            options.append(group)
    return options

sync_plotly_chart_state()

def make_trace_figure(sub, events_df, settings, file_name):
    direction = settings.get('direction', 'inward (EPSC)')
    marker_color = '#1a6b55' if 'EPSC' in direction else '#b91c1c'
    fig, ax = plt.subplots(figsize=(11, 3.5))
    fig.patch.set_facecolor('white')
    ax.set_facecolor('white')
    ax.spines[['top', 'right']].set_visible(False)
    ax.spines[['left', 'bottom']].set_color('#d1d5db')
    ax.tick_params(colors='#6b7280', labelsize=9)
    ax.set_xlabel('Time (s)', fontsize=9, color='#6b7280')
    ax.set_ylabel('Current (pA)', fontsize=9, color='#6b7280')
    ax.set_title(file_name, fontsize=10, color='#1a1a1a', pad=8)
    if not sub.empty:
        bl_end = sub['time_s'].min() + (sub['time_s'].max() - sub['time_s'].min()) * settings.get('baseline_pct', 20) / 100
        ax.axvspan(sub['time_s'].min(), bl_end, color='#f3f4f6', alpha=0.5, label='baseline region', zorder=0)
        ax.plot(sub['time_s'], sub['signal'], lw=0.7, color='#374151', zorder=1)
        if events_df is not None and not events_df.empty:
            nbase = max(1, int(len(sub) * settings.get('baseline_pct', 20) / 100))
            baseline = np.median(sub['signal'].values[:nbase])
            acc = events_df[events_df['accepted'] == True]
            rej = events_df[events_df['accepted'] != True]
            if not acc.empty:
                ax.scatter(acc['time_s'], acc['amplitude_pA'] + baseline, s=28, color=marker_color, zorder=3, label=f"{len(acc)} events")
            if not rej.empty:
                ax.scatter(rej['time_s'], rej['amplitude_pA'] + baseline, s=18, color='#9ca3af', zorder=2, marker='x', label=f"{len(rej)} rejected")
        ax.legend(fontsize=8, frameon=False, loc='upper right')
        if is_valid_y_range(settings.get('y_min'), settings.get('y_max')):
            ax.set_ylim(float(settings['y_min']), float(settings['y_max']))
    plt.tight_layout(pad=0.8)
    return fig

def fig_to_png_bytes(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=150, bbox_inches='tight')
    buf.seek(0)
    return buf.read()

def make_trace_figure_plotly(sub, events_df, settings, file_name):
    """Interactive Plotly figure with drag-zoom, box-select, scroll-zoom and double-click reset."""
    direction = settings.get('direction', 'inward (EPSC)')
    marker_color = '#1a6b55' if 'EPSC' in direction else '#b91c1c'
    xaxis_revision = f"{file_name}:{settings.get('sweep')}:{settings.get('t_start')}:{settings.get('t_end')}"
    yaxis_revision = f"{file_name}:{settings.get('y_min')}:{settings.get('y_max')}"
    fig = go.Figure()
    if not sub.empty:
        bl_end = sub['time_s'].min() + (sub['time_s'].max() - sub['time_s'].min()) * settings.get('baseline_pct', 20) / 100
        fig.add_vrect(
            x0=float(sub['time_s'].min()), x1=float(bl_end),
            fillcolor='#f3f4f6', opacity=0.5, line_width=0, layer='below',
            annotation_text='baseline', annotation_position='top left',
            annotation_font_size=9, annotation_font_color='#9ca3af',
        )
        fig.add_trace(go.Scattergl(
            x=sub['time_s'], y=sub['signal'],
            mode='lines', line=dict(color='#374151', width=0.8),
            name='Trace',
            hovertemplate='Time: %{x:.4f}s<br>Current: %{y:.2f}pA<extra></extra>',
        ))
        if events_df is not None and not events_df.empty:
            nbase = max(1, int(len(sub) * settings.get('baseline_pct', 20) / 100))
            baseline = float(np.median(sub['signal'].values[:nbase]))
            acc = events_df[events_df['accepted'] == True]
            rej = events_df[events_df['accepted'] != True]
            if not acc.empty:
                acc_custom = [[int(idx), 1, float(amp)] for idx, amp in zip(acc.index, acc['amplitude_pA'])]
                acc_ids = [str(int(idx)) for idx in acc.index]
                fig.add_trace(go.Scatter(
                    x=acc['time_s'], y=acc['amplitude_pA'] + baseline,
                    mode='markers',
                    marker=dict(color=marker_color, size=8, line=dict(color='white', width=0.8)),
                    ids=acc_ids,
                    customdata=acc_custom,
                    name=f'{len(acc)} events',
                    hovertemplate='Time: %{x:.4f}s<br>Amp: %{customdata[2]:.2f}pA<br>Click to reject<extra></extra>',
                ))
            if not rej.empty:
                rej_custom = [[int(idx), 0, float(amp)] for idx, amp in zip(rej.index, rej['amplitude_pA'])]
                rej_ids = [str(int(idx)) for idx in rej.index]
                fig.add_trace(go.Scatter(
                    x=rej['time_s'], y=rej['amplitude_pA'] + baseline,
                    mode='markers',
                    marker=dict(color='#9ca3af', size=7, symbol='x', line=dict(width=1.5)),
                    ids=rej_ids,
                    customdata=rej_custom,
                    name=f'{len(rej)} rejected',
                    hovertemplate='Time: %{x:.4f}s<br>Amp: %{customdata[2]:.2f}pA<br>Click to restore<extra></extra>',
                ))
    fig.update_layout(
        title=dict(text=file_name, font=dict(size=13, color='#1a1a1a')),
        xaxis=dict(
            title=dict(text='Time (s)', font=dict(size=11, color='#6b7280')),
            tickfont=dict(size=10, color='#6b7280'),
            showgrid=True, gridcolor='rgba(0,0,0,0.05)', zeroline=False,
            uirevision=xaxis_revision,
        ),
        yaxis=dict(
            title=dict(text='Current (pA)', font=dict(size=11, color='#6b7280')),
            tickfont=dict(size=10, color='#6b7280'),
            showgrid=True, gridcolor='rgba(0,0,0,0.05)', zeroline=False,
            uirevision=yaxis_revision,
        ),
        plot_bgcolor='white', paper_bgcolor='white',
        height=620,
        margin=dict(l=50, r=12, t=32, b=40),
        legend=dict(
            font=dict(size=10), bgcolor='rgba(255,255,255,0.8)',
            bordercolor='rgba(0,0,0,0.1)', borderwidth=1,
            x=1, y=1, xanchor='right', yanchor='top',
        ),
        dragmode='zoom',
        hovermode='x unified',
        modebar=dict(bgcolor='rgba(255,255,255,0.7)', color='#6b7280', activecolor='#1a6b55'),
        uirevision=file_name,
    )
    if is_valid_y_range(settings.get('y_min'), settings.get('y_max')):
        fig.update_yaxes(range=[float(settings['y_min']), float(settings['y_max'])], autorange=False)
    return fig

def lighten_hex(hex_color, frac):
    hex_color = hex_color.lstrip('#')
    r = int(hex_color[0:2], 16) / 255
    g = int(hex_color[2:4], 16) / 255
    b = int(hex_color[4:6], 16) / 255
    return (r * frac + (1 - frac), g * frac + (1 - frac), b * frac + (1 - frac))

# ───────────────────────────────────────────
#  SIDEBAR — controls, file upload, params
# ───────────────────────────────────────────
_run_detect = False
_clear_win = False
_save_btn = False

with st.sidebar:
    st.markdown("""
    <div style='display:flex;align-items:center;gap:8px;padding:0 0 2px 0;margin-bottom:16px'>
        <svg width="24" height="24" viewBox="0 0 36 36" fill="none" xmlns="http://www.w3.org/2000/svg">
          <rect width="36" height="36" rx="8" fill="#1a6b55"/>
          <polyline points="4,18 10,18 14,8 18,28 22,14 26,18 32,18" stroke="white" stroke-width="2.2" stroke-linejoin="round" stroke-linecap="round" fill="none"/>
        </svg>
        <div>
            <div style='font-size:13px;font-weight:700;color:#111827;line-height:1.1'>SynCapture</div>
            <div style='font-size:9px;color:#9ca3af'>Synaptic Event Analysis</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    uploaded = st.file_uploader('Upload ABF files', type=['abf'], accept_multiple_files=True, label_visibility='collapsed')
    if uploaded:
        for f in uploaded:
            if f.name not in S.files:
                try:
                    abf, tmp = load_abf(f)
                    df_sw = abf_to_sweeps(abf)
                    S.files[f.name] = {
                        'abf_path': str(tmp),
                        'meta': {
                            'file_name': f.name,
                            'sample_rate_hz': float(abf.dataRate),
                            'sweep_count': len(abf.sweepList),
                            'protocol': getattr(abf, 'protocol', ''),
                            'unit_y': getattr(abf, 'adcUnits', 'pA') if hasattr(abf, 'adcUnits') else 'pA',
                            'duration_s': float(df_sw.groupby('sweep')['time_s'].max().iloc[0]) if not df_sw.empty else 0.0,
                        },
                        'df': df_sw,
                    }
                    if f.name not in S.file_order:
                        S.file_order.append(f.name)
                    if f.name not in S.events:
                        S.events[f.name] = pd.DataFrame(columns=['time_s','amplitude_pA','prominence','iei_s','accepted'])
                    if f.name not in S.settings:
                        S.settings[f.name] = {}
                except Exception as e:
                    st.error(f'{f.name}: {e}')

    if S.file_order:
        selectable = [n for n in S.file_order if n not in S.skipped]
        if selectable:
            default_ix = selectable.index(S.active) if S.active in selectable else 0
            S.active = st.selectbox('📂 File', selectable, index=default_ix, key='file_select')
            st.markdown("<div style='height:0.35rem'></div>", unsafe_allow_html=True)
            if st.button('⊘ Skip this file', use_container_width=True):
                S.skipped.add(S.active)
                remaining = [n for n in S.file_order if n not in S.skipped]
                S.active = remaining[0] if remaining else None
                st.rerun()

    if S.active and S.active in S.files:
        fdata = S.files[S.active]
        meta = fdata['meta']
        df_all = fdata['df']
        sweeps_available = sorted(df_all['sweep'].unique().tolist())
        prev = S.settings.get(S.active, {})
        default_sweep = prev.get('sweep', sweeps_available[0])

        st.markdown("<p style='font-size:0.75rem;font-weight:600;color:#6b7280;text-transform:uppercase;letter-spacing:0.5px;margin:0.3rem 0 0.3rem 0'>Sweep & Window</p>", unsafe_allow_html=True)
        sweep = st.selectbox('Sweep', sweeps_available, index=sweeps_available.index(default_sweep) if default_sweep in sweeps_available else 0, key=f'sweep_{S.active}')
        sweep_df = df_all[df_all['sweep'] == sweep].copy()
        t_min, t_max = float(sweep_df['time_s'].min()), float(sweep_df['time_s'].max())

        sc1, sc2 = st.columns(2)
        with sc1:
            t_start = st.number_input('Start (s)', min_value=0.0, max_value=t_max, value=float(prev.get('t_start', t_min)), step=0.1, key=f't_start_{S.active}')
        with sc2:
            t_end = st.number_input('End (s)', min_value=0.0, max_value=t_max, value=float(prev.get('t_end', t_max)), step=0.1, key=f't_end_{S.active}')
        lp_hz = st.number_input('Low-pass (Hz)', min_value=0.0, value=float(prev.get('lp_hz', 1000.0)), step=50.0, help='Gaussian filter, 0 = off', key=f'lp_hz_{S.active}')

        y_window_df = sweep_df[(sweep_df['time_s'] >= t_start) & (sweep_df['time_s'] <= t_end)].copy() if t_end > t_start else sweep_df.copy()
        if y_window_df.empty:
            y_window_df = sweep_df.copy()
        y_values = y_window_df['signal'].to_numpy()
        if lp_hz > 0 and meta['sample_rate_hz'] > 0 and len(y_values) > 20:
            y_values = gaussian_lowpass(y_values, lp_hz, meta['sample_rate_hz'])
        default_y_min, default_y_max = infer_y_range(y_values)
        saved_y_min = prev.get('y_min', default_y_min)
        saved_y_max = prev.get('y_max', default_y_max)
        if not is_valid_y_range(saved_y_min, saved_y_max):
            saved_y_min, saved_y_max = default_y_min, default_y_max
        y_step = max((default_y_max - default_y_min) / 100, 0.001)
        y_min_key = f'y_min_{S.active}'
        y_max_key = f'y_max_{S.active}'
        if y_min_key not in st.session_state:
            st.session_state[y_min_key] = float(saved_y_min)
        if y_max_key not in st.session_state:
            st.session_state[y_max_key] = float(saved_y_max)

        st.markdown("<p style='font-size:0.75rem;font-weight:600;color:#6b7280;text-transform:uppercase;letter-spacing:0.5px;margin:0.3rem 0 0.3rem 0'>Y Scale</p>", unsafe_allow_html=True)
        yc1, yc2 = st.columns(2)
        with yc1:
            y_min = st.number_input('Y min (pA)', step=float(y_step), format='%.3f', key=y_min_key)
        with yc2:
            y_max = st.number_input('Y max (pA)', step=float(y_step), format='%.3f', key=y_max_key)
        st.markdown("<div style='height:0.35rem'></div>", unsafe_allow_html=True)
        st.button(
            'Reset Y to data',
            key=f'reset_y_{S.active}',
            use_container_width=True,
            on_click=reset_y_scale,
            args=(S.active, default_y_min, default_y_max),
        )
        if not is_valid_y_range(y_min, y_max):
            st.warning('Y max must be greater than Y min.')
            y_min, y_max = saved_y_min, saved_y_max

        st.markdown("<p style='font-size:0.75rem;font-weight:600;color:#6b7280;text-transform:uppercase;letter-spacing:0.5px;margin:0.3rem 0 0.3rem 0'>Detection</p>", unsafe_allow_html=True)
        direction = st.selectbox('Direction', ['inward (EPSC)', 'outward (IPSC)'], index=['inward (EPSC)', 'outward (IPSC)'].index(prev.get('direction', 'inward (EPSC)')) if prev.get('direction', 'inward (EPSC)') in ['inward (EPSC)', 'outward (IPSC)'] else 0, key=f'direction_{S.active}')
        baseline_pct = st.slider('Baseline %', 5, 50, int(prev.get('baseline_pct', 20)), 5, key=f'bl_pct_{S.active}')

        pc1, pc2 = st.columns(2)
        with pc1:
            prominence = st.number_input('Prom. (pA)', min_value=0.5, value=float(prev.get('prominence', 8.0)), step=0.5, key=f'prom_{S.active}')
            tau_rise = st.number_input('Tau Rise (ms)', min_value=0.1, value=float(prev.get('tau_rise', 0.5)), step=0.1, key=f'tau_rise_{S.active}')
        with pc2:
            distance_ms = st.number_input('Min IEI (ms)', min_value=0.1, value=float(prev.get('distance_ms', 5.0)), step=0.5, key=f'dist_{S.active}')
            tau_decay = st.number_input('Tau Decay (ms)', min_value=0.5, value=float(prev.get('tau_decay', 3.0)), step=0.5, key=f'tau_decay_{S.active}')

        st.markdown("<div style='height:0.35rem'></div>", unsafe_allow_html=True)
        bc1, bc2 = st.columns(2)
        with bc1:
            _run_detect = st.button('⚡ Detect', type='primary', key=f'detect_{S.active}', use_container_width=True)
        with bc2:
            _clear_win = st.button('🗑 Clear', key=f'clear_win_{S.active}', use_container_width=True)

        S.settings[S.active] = {'direction': direction, 'baseline_pct': baseline_pct, 'prominence': prominence, 'distance_ms': distance_ms, 'tau_rise': tau_rise, 'tau_decay': tau_decay, 'sweep': sweep, 't_start': t_start, 't_end': t_end, 'lp_hz': lp_hz, 'y_min': y_min, 'y_max': y_max}

        st.markdown("<p style='font-size:0.88rem;font-weight:600;color:inherit;margin:0.5rem 0 0.45rem 0'>🏷️ Cell Labels</p>", unsafe_allow_html=True)
        saved_rec = next((r for r in S.records if r.get('file_name') == S.active), {})
        cell_id = st.text_input('Cell ID', value=saved_rec.get('cell_id', Path(S.active).stem), key=f'cell_id_{S.active}')
        individual = st.text_input('Individual', value=saved_rec.get('individual', ''), key=f'individual_{S.active}')
        lc1, lc2 = st.columns(2)
        with lc1:
            saved_group = saved_rec.get('group', 'naive')
            group_key = f'group_{S.active}'
            current_group = remember_group_option(st.session_state.get(group_key))
            group_options = group_options_for(saved_group, current_group)
            group = st.selectbox(
                'Group',
                group_options,
                index=group_options.index(saved_group) if saved_group in group_options else 0,
                key=group_key,
                accept_new_options=True,
                placeholder='Select or type a group',
            )
            group = remember_group_option(group) or 'naive'
        with lc2:
            status_options = ['accepted', 'needs_check', 'rejected']
            saved_status = saved_rec.get('status', 'accepted')
            status = st.selectbox('Status', status_options, index=status_options.index(saved_status) if saved_status in status_options else 0, key=f'status_{S.active}')
        st.markdown("<div style='height:0.35rem'></div>", unsafe_allow_html=True)
        _save_btn = st.button('✓ Save to dataset', type='primary', key=f'save_{S.active}', use_container_width=True)

    if not HAS_PYABF:
        st.divider()
        st.warning('⚠ pyabf not installed. Run: `pip install pyabf`')

# ───────────────────────────────────────────
#  MAIN AREA — chart, metrics, events table
# ───────────────────────────────────────────
if not S.file_order:
    st.markdown("""
    <div style='text-align:center;padding:8rem 2rem 4rem 2rem'>
        <svg width="52" height="52" viewBox="0 0 36 36" fill="none" xmlns="http://www.w3.org/2000/svg" style="margin:0 auto 16px auto;display:block">
          <rect width="36" height="36" rx="8" fill="#1a6b55"/>
          <polyline points="4,18 10,18 14,8 18,28 22,14 26,18 32,18" stroke="white" stroke-width="2.2" stroke-linejoin="round" stroke-linecap="round" fill="none"/>
        </svg>
        <div style='font-size:1.5rem;font-weight:700;color:#111827;margin-bottom:4px'>SynCapture</div>
        <p style='color:#6b7280;font-size:0.95rem'>Upload ABF files in the sidebar to begin analysis</p>
    </div>
    """, unsafe_allow_html=True)

elif not S.active or S.active not in S.files:
    st.info('← Select a file from the sidebar to begin.')

else:
    # ---- data processing ----
    fdata = S.files[S.active]
    meta = fdata['meta']
    df_all = fdata['df']

    if t_end <= t_start:
        st.error('End time must be greater than start time. Adjust in the sidebar.')
        st.stop()

    sub = sweep_df[(sweep_df['time_s'] >= t_start) & (sweep_df['time_s'] <= t_end)].copy()
    if lp_hz > 0 and meta['sample_rate_hz'] > 0 and len(sub) > 20:
        sub['signal'] = gaussian_lowpass(sub['signal'].to_numpy(), lp_hz, meta['sample_rate_hz'])

    current_events = S.events.get(S.active, pd.DataFrame(columns=['time_s','amplitude_pA','prominence','iei_s','accepted']))

    # ---- handle sidebar actions ----
    if _run_detect and not sub.empty:
        new_ev = detect_synaptic_events(sub['signal'].to_numpy(), sub['time_s'].to_numpy(), direction, prominence, distance_ms, baseline_pct, tau_rise, tau_decay)
        outside = current_events[(current_events['time_s'] < t_start) | (current_events['time_s'] > t_end)] if not current_events.empty else pd.DataFrame(columns=new_ev.columns)
        S.events[S.active] = pd.concat([outside, new_ev], ignore_index=True).sort_values('time_s').reset_index(drop=True)
        bump_event_table_revision(S.active)
        st.rerun()
    if _clear_win:
        if not current_events.empty:
            S.events[S.active] = current_events[(current_events['time_s'] < t_start) | (current_events['time_s'] > t_end)].reset_index(drop=True)
            bump_event_table_revision(S.active)
        st.rerun()
    if _save_btn:
        all_ev_s = S.events.get(S.active, pd.DataFrame())
        dur_s = max(0.001, t_end - t_start)
        sm_s = summary_from_events(all_ev_s[(all_ev_s['time_s'] >= t_start) & (all_ev_s['time_s'] <= t_end)] if not all_ev_s.empty else all_ev_s, dur_s)
        rec = {
            'file_name': S.active, 'cell_id': cell_id, 'individual': individual, 'group': group,
            'status': status, 'sweep': sweep, 'window_start_s': t_start, 'window_end_s': t_end, 'window_dur_s': dur_s,
            **sm_s, 'lp_hz': lp_hz, 'direction': direction, 'sample_rate_hz': meta['sample_rate_hz'],
        }
        S.records = [r for r in S.records if r.get('file_name') != S.active]
        S.records.append(json_safe(rec))
        st.toast(f'✓ Saved {cell_id} → dataset ({len(S.records)} files)')

    # ---- interactive trace chart ----
    win_events = current_events[(current_events['time_s'] >= t_start) & (current_events['time_s'] <= t_end)] if not current_events.empty else current_events
    fig_plotly = make_trace_figure_plotly(sub, win_events if not win_events.empty else None, S.settings[S.active], S.active)
    plotly_relayout_chart(
        fig_plotly,
        S.active,
        config={'scrollZoom': True, 'displayModeBar': True, 'modeBarButtonsToAdd': ['drawrect', 'eraseshape']},
    )

    # ---- metrics row ----
    full_ev = S.events.get(S.active, pd.DataFrame())
    if not full_ev.empty:
        dur = max(0.001, t_end - t_start)
        window_full_ev = full_ev[(full_ev['time_s'] >= t_start) & (full_ev['time_s'] <= t_end)]
        sm = summary_from_events(window_full_ev, dur)
        m1, m2, m3, m4, m5 = st.columns(5)
        m1.metric('Events', sm['n_events'])
        m2.metric('Freq (Hz)', f"{sm['freq_hz']:.4f}" if pd.notna(sm['freq_hz']) else '—')
        m3.metric('Mean |Amp|', f"{sm['amp_mean_pA']:.2f} pA" if pd.notna(sm['amp_mean_pA']) else '—')
        m4.metric('Med |Amp|', f"{sm['amp_median_pA']:.2f} pA" if pd.notna(sm['amp_median_pA']) else '—')
        m5.metric('Mean IEI', f"{sm['iei_mean_s']:.4f} s" if pd.notna(sm['iei_mean_s']) else '—')

    # ---- editable event table ----
    if not S.events.get(S.active, pd.DataFrame()).empty:
        all_ev = S.events[S.active].copy()
        win_ev = all_ev[(all_ev['time_s'] >= t_start) & (all_ev['time_s'] <= t_end)].copy()
        total_acc = int((all_ev['accepted'] == True).sum()) if 'accepted' in all_ev.columns else 0
        total_rej = int((all_ev['accepted'] != True).sum()) if 'accepted' in all_ev.columns else 0
        table_revision = S.event_table_revisions.get(S.active, 0)
        editor_key = f'ev_table_{S.active}_{table_revision}'
        editor_source_key = f'{editor_key}_source'
        S[editor_source_key] = win_ev.copy()
        panel_key = f'events_panel_open_{S.active}'
        if panel_key not in S:
            S[panel_key] = False
        panel_icon = '▾' if S[panel_key] else '▸'
        if st.button(
            f'{panel_icon} 📋 {len(win_ev)} events in window · {total_acc} accepted · {total_rej} rejected total',
            key=f'events_panel_toggle_{S.active}',
            use_container_width=True,
        ):
            S[panel_key] = not S[panel_key]
        if S[panel_key]:
            st.data_editor(
                win_ev, num_rows='dynamic', use_container_width=True, key=editor_key,
                column_config={
                    'accepted': st.column_config.CheckboxColumn('Accept'),
                    'time_s': st.column_config.NumberColumn('Time (s)', format='%.4f'),
                    'amplitude_pA': st.column_config.NumberColumn('Amplitude (pA)', format='%.2f'),
                    'prominence': st.column_config.NumberColumn('Prominence', format='%.2f'),
                    'iei_s': st.column_config.NumberColumn('IEI (s)', format='%.4f'),
                },
                height=260,
                on_change=apply_event_table_delta,
                args=(S.active, editor_key, editor_source_key),
            )

# ───────────────────────────────────────────
#  EXPORT SECTION
# ───────────────────────────────────────────
if S.records:
    st.divider()
    st.markdown(f'**📊 Summary & Export ({len(S.records)} files)**')
    df_rec = pd.DataFrame(S.records).sort_values(['group', 'individual', 'cell_id'])
    st.dataframe(df_rec, use_container_width=True, height=200)
    clean = df_rec[df_rec['status'] == 'accepted'].copy()
    if not clean.empty:
        def sem(s):
            s = pd.Series(s).dropna()
            return s.std(ddof=1) / np.sqrt(len(s)) if len(s) > 1 else np.nan

        summary = clean.groupby('group').agg(
            n_cells=('cell_id', 'count'),
            amp_mean=('amp_mean_pA', 'mean'), amp_sem=('amp_mean_pA', sem),
            freq_mean=('freq_hz', 'mean'), freq_sem=('freq_hz', sem),
        ).reset_index()
        fig_sum, axes = plt.subplots(1, 2, figsize=(9, 4))
        fig_sum.patch.set_facecolor('white')
        order = sorted(clean['group'].unique())
        palette = {'naive': '#1a6b55', 'ovx': '#b91c1c', 'control': '#1d4ed8', 'treatment': '#b45309', 'other': '#6b7280'}
        rng = np.random.default_rng(42)
        panels = [('amp_mean_pA', 'Mean Amplitude (pA)', 'amp_mean', 'amp_sem'), ('freq_hz', 'Frequency (Hz)', 'freq_mean', 'freq_sem')]
        for ax_i, (col, label, mean_col, sem_col) in enumerate(panels):
            ax = axes[ax_i]
            ax.set_facecolor('white')
            ax.spines[['top', 'right']].set_visible(False)
            ax.spines[['left', 'bottom']].set_color('#d1d5db')
            ax.tick_params(colors='#6b7280', labelsize=9)
            ax.set_ylabel(label, fontsize=9, color='#374151')
            xs = np.arange(len(order))
            for xi, g in enumerate(order):
                gdf = clean[clean['group'] == g]
                grp_color = palette.get(g, '#6b7280')
                inds = sorted([i for i in gdf['individual'].dropna().unique().tolist()])
                ind_shades = {}
                for ii, ind in enumerate(inds):
                    frac = 0.35 + 0.5 * (ii / max(1, len(inds) - 1)) if len(inds) > 1 else 0.6
                    ind_shades[ind] = lighten_hex(grp_color, frac)
                gm = summary[summary['group'] == g]
                mean_val = gm[mean_col].values[0] if not gm.empty else np.nan
                sem_val = gm[sem_col].values[0] if not gm.empty else np.nan
                ax.bar(xi, mean_val, yerr=sem_val, capsize=4, color=grp_color, alpha=0.55, edgecolor=grp_color, linewidth=1, width=0.55, error_kw={'linewidth': 1.5})
                if inds:
                    for ind in inds:
                        idf = gdf[gdf['individual'] == ind][col].dropna()
                        j = rng.uniform(-0.1, 0.1, size=len(idf))
                        ax.scatter(np.full(len(idf), xi) + j, idf.values, color=ind_shades[ind], s=30, zorder=4, edgecolors=grp_color, linewidths=0.6)
                else:
                    vals = gdf[col].dropna()
                    j = rng.uniform(-0.1, 0.1, size=len(vals))
                    ax.scatter(np.full(len(vals), xi) + j, vals.values, color=lighten_hex(grp_color, 0.6), s=30, zorder=4, edgecolors=grp_color, linewidths=0.6)
            ax.set_xticks(xs)
            ax.set_xticklabels(order, fontsize=9)
        plt.tight_layout(pad=1.2)
        st.pyplot(fig_sum, clear_figure=True)
        plt.close(fig_sum)

        all_groups = sorted(clean['group'].unique())

        def prism_df(col):
            max_len = int(clean.groupby('group')[col].count().max())
            out = {}
            for g in all_groups:
                vals = clean[clean['group'] == g][col].reset_index(drop=True)
                out[g] = vals.reindex(range(max_len))
            return pd.DataFrame(out)

        prism_amp = prism_df('amp_mean_pA')
        prism_freq = prism_df('freq_hz')

        event_rows = []
        for rec in S.records:
            ev = S.events.get(rec['file_name'], pd.DataFrame())
            if not ev.empty:
                ev = ev.copy()
                ev['file_name'] = rec['file_name']
                ev['cell_id'] = rec['cell_id']
                ev['group'] = rec['group']
                ev['individual'] = rec['individual']
                event_rows.append(ev)
        events_all = pd.concat(event_rows, ignore_index=True) if event_rows else pd.DataFrame()

        img_bytes = {}
        for rec in S.records:
            fname = rec['file_name']
            fdata_exp = S.files.get(fname)
            if not fdata_exp:
                continue
            df_all2 = fdata_exp['df']
            sett = S.settings.get(fname, {})
            sw = sett.get('sweep', df_all2['sweep'].iloc[0])
            ts = sett.get('t_start', float(df_all2['time_s'].min()))
            te = sett.get('t_end', float(df_all2['time_s'].max()))
            sub_f = df_all2[(df_all2['sweep'] == sw) & (df_all2['time_s'] >= ts) & (df_all2['time_s'] <= te)].copy()
            lp = sett.get('lp_hz', 0)
            if lp > 0 and fdata_exp['meta']['sample_rate_hz'] > 0 and len(sub_f) > 20:
                sub_f['signal'] = gaussian_lowpass(sub_f['signal'].to_numpy(), lp, fdata_exp['meta']['sample_rate_hz'])
            ev_f = S.events.get(fname, pd.DataFrame())
            fig_f = make_trace_figure(sub_f, ev_f, sett, fname)
            img_bytes[fname] = fig_to_png_bytes(fig_f)
            plt.close(fig_f)

        zbuf = io.BytesIO()
        with zipfile.ZipFile(zbuf, 'w', zipfile.ZIP_DEFLATED) as zf:
            zf.writestr('per_cell_summary.csv', df_rec.to_csv(index=False))
            zf.writestr('group_mean_sem.csv', summary.to_csv(index=False))
            zf.writestr('Prism_amplitude_pA.csv', prism_amp.to_csv(index=False))
            zf.writestr('Prism_frequency_Hz.csv', prism_freq.to_csv(index=False))
            if not events_all.empty:
                zf.writestr('all_events.csv', events_all.to_csv(index=False))
            for fname, ibytes in img_bytes.items():
                zf.writestr(f'traces/{Path(fname).stem}_detected.png', ibytes)
            zf.writestr('review_state.json', json.dumps(json_safe(S.records), indent=2))
        st.download_button('⬇ Download all exports (ZIP)', data=zbuf.getvalue(), file_name='syncapture_exports.zip', mime='application/zip', type='primary')
        exp1, exp2, exp3 = st.columns(3)
        with exp1:
            st.download_button('Prism: Amplitude CSV', prism_amp.to_csv(index=False).encode(), file_name='Prism_amplitude_pA.csv', mime='text/csv')
        with exp2:
            st.download_button('Prism: Frequency CSV', prism_freq.to_csv(index=False).encode(), file_name='Prism_frequency_Hz.csv', mime='text/csv')
        with exp3:
            st.download_button('Per-cell summary CSV', df_rec.to_csv(index=False).encode(), file_name='per_cell_summary.csv', mime='text/csv')
