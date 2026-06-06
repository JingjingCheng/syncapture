#!/usr/bin/env python3
"""
SynCapture — Synaptic Event Analysis Tool
Run: streamlit run syncapture.py
Dependencies: pip install streamlit pyabf scipy pandas matplotlib numpy plotly
"""
import hashlib, io, json, shutil, zipfile, tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
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
  --accent:#1a6b55;
  --accent-l:#eaf2ef;
}
header[data-testid="stHeader"] { display: none !important; height: 0 !important; }
[data-testid="stAppViewContainer"] { top: 0 !important; margin-top: 0 !important; }
[data-testid="stMain"] > .block-container,
[data-testid="stMainBlockContainer"] { padding: 0.5rem 1.5rem 1rem 1.5rem !important; margin-top: 0 !important; max-width: 100% !important; }
section[data-testid="stSidebar"] { min-width: 292px !important; max-width: 312px !important; }
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
section[data-testid="stSidebar"] [data-testid="stExpander"] details {
  padding: 0 !important;
}
/* Compact controls */
section[data-testid="stSidebar"] .stFileUploader { margin-bottom: -0.2rem !important; }
section[data-testid="stSidebar"] .stFileUploader [data-testid="stFileUploaderDropzone"] {
  min-height: 2.15rem !important;
  padding: 0.22rem 0.35rem !important;
  border-style: solid !important;
}
section[data-testid="stSidebar"] .stFileUploader [data-testid="stFileUploaderDropzone"] button {
  background: var(--accent) !important;
  border-color: var(--accent) !important;
}
section[data-testid="stSidebar"] .stFileUploader [data-testid="stFileUploaderDropzone"] button * {
  background: transparent !important;
  color: #ffffff !important;
}
section[data-testid="stSidebar"] .stFileUploader [data-testid="stFileUploaderDropzone"] > div:first-child {
  display: none !important;
}
section[data-testid="stSidebar"] .stFileUploader [data-testid="stFileUploaderDropzone"] small {
  display: none !important;
}
section[data-testid="stSidebar"] .stNumberInput input,
section[data-testid="stSidebar"] .stTextInput input {
  min-height: 2.38rem !important;
  height: 2.38rem !important;
  padding-top: 0 !important;
  padding-bottom: 0 !important;
  font-size: 0.84rem !important;
  line-height: 1.25 !important;
}
section[data-testid="stSidebar"] .stSelectbox [data-baseweb="select"] > div {
  min-height: 2.55rem !important;
  height: 2.55rem !important;
  align-items: center !important;
  font-size: 0.84rem !important;
  line-height: 1.25 !important;
}
section[data-testid="stSidebar"] .stSelectbox [data-baseweb="select"] > div > div {
  min-height: 2.55rem !important;
  align-items: center !important;
  padding-top: 0 !important;
  padding-bottom: 0 !important;
  overflow: visible !important;
}
section[data-testid="stSidebar"] .stSelectbox [data-baseweb="select"] span,
section[data-testid="stSidebar"] .stSelectbox [data-baseweb="select"] p {
  line-height: 1.25 !important;
}
section[data-testid="stSidebar"] .stSelectbox [data-baseweb="select"] {
  width: 100% !important;
}
div[data-baseweb="popover"] [role="listbox"] {
  min-width: 320px !important;
}
section[data-testid="stSidebar"] .stSlider [data-testid="stThumbValue"] {
  font-size: 0.68rem !important;
}
section[data-testid="stSidebar"] .stButton > button {
  min-height: 1.85rem !important;
  padding: 0.2rem 0.45rem !important;
}
section[data-testid="stSidebar"] .stButton > button[kind="primary"] {
  background: var(--accent) !important;
  border-color: var(--accent) !important;
  color: #ffffff !important;
}
section[data-testid="stSidebar"] .stFileUploader button[data-testid="stBaseButton-secondary"] {
  background: var(--accent) !important;
  border-color: var(--accent) !important;
}
section[data-testid="stSidebar"] .stFileUploader button[data-testid="stBaseButton-secondary"] * {
  color: #ffffff !important;
}
/* Make the Plotly chart fill full width and available height */
[data-testid="stPlotlyChart"] { width: 100% !important; min-height: 0 !important; }
[data-testid="stPlotlyChart"] > div { width: 100% !important; height: 100% !important; }
[data-testid="stPlotlyChart"] iframe { width: 100% !important; height: 100% !important; }
div[data-testid="stMetric"] {
  background: color-mix(in srgb, currentColor 5%, transparent);
  border:1px solid color-mix(in srgb, currentColor 10%, transparent);
  border-radius:6px;
  padding:0.22rem 0.32rem;
  min-height: 2.35rem;
  overflow: hidden;
}
div[data-testid="stMetric"] label {
  font-size:0.58rem !important;
  line-height:1 !important;
  white-space: nowrap !important;
}
div[data-testid="stMetric"] [data-testid="stMetricValue"] {
  font-size:0.78rem !important;
  line-height:1.05 !important;
  white-space: nowrap !important;
}
.metric-inline-row {
  display: grid;
  grid-template-columns: repeat(6, minmax(0, 1fr));
  gap: 0.45rem;
  margin: -1.2rem 0 0.8rem 0;
  position: relative;
  z-index: 10;
}
.metric-inline-card {
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  gap: 0.25rem;
  min-width: 0;
  min-height: 3.4rem;
  padding: 0.4rem 0.42rem;
  overflow: hidden;
  white-space: nowrap;
  background: color-mix(in srgb, currentColor 5%, transparent);
  border: 1px solid color-mix(in srgb, currentColor 10%, transparent);
  border-radius: 6px;
}
.metric-inline-label {
  flex: 0 1 auto;
  min-width: 0;
  overflow: hidden;
  text-overflow: ellipsis;
  color: color-mix(in srgb, currentColor 65%, transparent);
  font-size: 0.85rem;
  font-weight: 600;
  line-height: 1;
}
.metric-inline-value {
  flex: 0 0 auto;
  color: inherit;
  font-size: 1.15rem;
  font-weight: 600;
  line-height: 1;
}
.compact-panel [data-testid="stVerticalBlock"] { gap: 0.35rem !important; }
.compact-panel .stDataFrame, .compact-panel [data-testid="stDataFrame"] { font-size: 0.78rem !important; }
.style-section-title {
  margin: 0.45rem 0 0.6rem 0;
  padding-top: 0.32rem;
  border-top: 1px solid color-mix(in srgb, currentColor 16%, transparent);
  color: color-mix(in srgb, currentColor 65%, transparent);
  font-size: 0.72rem;
  font-weight: 700;
  line-height: 1.1;
  text-transform: uppercase;
}
.style-subsection-title {
  margin: 0.18rem 0 0.18rem 0;
  color: color-mix(in srgb, currentColor 78%, transparent);
  font-size: 0.7rem;
  font-weight: 700;
  line-height: 1.1;
}
[data-testid="stMain"] [data-testid="stExpander"] [data-testid="stVerticalBlock"] {
  gap: 0.34rem !important;
}
[data-testid="stMain"] [data-testid="stExpander"] label p {
  color: color-mix(in srgb, currentColor 65%, transparent) !important;
  font-size: 0.72rem !important;
  font-weight: 600 !important;
  line-height: 1.08 !important;
  margin-bottom: 0.06rem !important;
}
[data-testid="stMain"] [data-testid="stExpander"] .stNumberInput input {
  min-height: 2.08rem !important;
  height: 2.08rem !important;
  padding: 0.12rem 0.42rem !important;
  font-size: 0.78rem !important;
}
[data-testid="stMain"] [data-testid="stExpander"] .stSelectbox [data-baseweb="select"] > div {
  min-height: 2.08rem !important;
  height: 2.08rem !important;
  font-size: 0.78rem !important;
}
[data-testid="stMain"] [data-testid="stExpander"] .stCheckbox {
  min-height: 2.08rem !important;
  display: flex !important;
  align-items: end !important;
}
[data-testid="stMain"] [data-testid="stExpander"] .stSlider {
  padding-top: 0 !important;
  margin-bottom: -0.2rem !important;
}
[data-testid="stMain"] .stDownloadButton > button {
  min-height: 2.2rem !important;
  padding: 0.34rem 0.72rem !important;
  border-radius: 6px !important;
  font-size: 0.82rem !important;
  font-weight: 650 !important;
  white-space: nowrap !important;
}
[data-testid="stMain"] .stDownloadButton {
  margin-bottom: 0 !important;
}
.stButton > button { border-radius:6px; font-size:0.82rem; font-weight:500; }
.stButton > button[kind="primary"] { background:var(--accent); border:none; }
.stButton > button:hover { opacity:0.88; }
</style>
""", unsafe_allow_html=True)

def _init():
    defaults = {
        'files': {}, 'file_order': [], 'active': None, 'skipped': set(),
        'events': {}, 'settings': {}, 'records': [], 'custom_groups': [],
        'event_table_revisions': {}, 'figure_style': {}, 'group_colors': {},
        'show_uploader': not bool(st.session_state.get('file_order', [])),
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

_init()
S = st.session_state

DEFAULT_GROUP_OPTIONS = ['naive', 'ovx', 'control', 'treatment', 'other']
EVENT_COLUMNS = ['time_s', 'amplitude_pA', 'prominence', 'iei_s', 'accepted', 'manual']
DEFAULT_GROUP_PALETTE = {
    'naive': '#1a6b55',
    'ovx': '#b91c1c',
    'control': '#1d4ed8',
    'treatment': '#b45309',
    'other': '#6b7280',
}
DEFAULT_FIGURE_STYLE = {
    'font_family': 'Arial',
    'font_size': 10,
    'axis_label_size': 11,
    'tick_label_size': 9,
    'axis_color': '#111827',
    'tick_color': '#374151',
    'grid_color': '#e5e7eb',
    'show_grid': False,
    'axis_line_width': 1.2,
    'x_label_rotation': 0,
    'bar_width': 0.55,
    'bar_line_width': 1.2,
    'error_line_width': 1.4,
    'point_size': 30,
    'show_individual_names': True,
    'individual_label_size': 7,
    'individual_label_color': '#111827',
    'trace_line_color': '#374151',
    'trace_line_width': 0.7,
    'trace_height_px': 520,
    'amp_y_auto': True,
    'amp_y_min': 0.0,
    'amp_y_max': 50.0,
    'freq_y_auto': True,
    'freq_y_min': 0.0,
    'freq_y_max': 10.0,
}

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
    #chart:focus {
      outline: none;
    }
  </style>
</head>
<body>
  <div id="chart" tabindex="0" aria-label="Trace chart"></div>
  <div id="chart-toast"></div>
  <script>
    const chart = document.getElementById("chart");
    const chartToast = document.getElementById("chart-toast");
    let plotHandlersAttached = false;
    let lastPayload = "";
    let currentFileName = null;
    let currentFallbackRange = null;
    let toastTimer = null;
    let pointClickTimer = null;
    let lastPlotMouseEvent = null;
    let lastAddEventAt = 0;

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
        manual: customdata ? boolOrNull(customdata[3]) === true : false,
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

    function plotCoordinates(mouseEvent) {
      if (!mouseEvent || !chart._fullLayout) {
        return null;
      }
      const xaxis = chart._fullLayout.xaxis;
      const yaxis = chart._fullLayout.yaxis;
      if (!xaxis || !yaxis || !Array.isArray(xaxis.range) || !Array.isArray(yaxis.range)) {
        return null;
      }
      const chartRect = chart.getBoundingClientRect();
      const size = chart._fullLayout._size || {};
      const dragRect = chart.querySelector(".nsewdrag");
      const plotRect = (
        Number.isFinite(size.l) && Number.isFinite(size.t) &&
        Number.isFinite(size.w) && Number.isFinite(size.h) &&
        size.w > 0 && size.h > 0
      ) ? {
        left: chartRect.left + size.l,
        top: chartRect.top + size.t,
        width: size.w,
        height: size.h,
      } : (dragRect ? dragRect.getBoundingClientRect() : null);
      if (!plotRect || !plotRect.width || !plotRect.height) {
        return null;
      }
      const xPixel = mouseEvent.clientX - plotRect.left;
      const yPixel = mouseEvent.clientY - plotRect.top;
      if (xPixel < 0 || yPixel < 0 || xPixel > plotRect.width || yPixel > plotRect.height) {
        return null;
      }
      const x0 = numberOrNull(xaxis.range[0]);
      const x1 = numberOrNull(xaxis.range[1]);
      const y0 = numberOrNull(yaxis.range[0]);
      const y1 = numberOrNull(yaxis.range[1]);
      if (x0 === null || x1 === null || y0 === null || y1 === null) {
        return null;
      }
      return {
        x: x0 + (xPixel / plotRect.width) * (x1 - x0),
        y: y1 - (yPixel / plotRect.height) * (y1 - y0),
      };
    }

    function rememberPlotMouseEvent(mouseEvent) {
      const coords = plotCoordinates(mouseEvent);
      if (!coords) {
        return;
      }
      lastPlotMouseEvent = {
        clientX: mouseEvent.clientX,
        clientY: mouseEvent.clientY,
      };
    }

    function addEventFromMouse(mouseEvent) {
      const now = Date.now();
      if (now - lastAddEventAt < 350) {
        return;
      }
      const coords = plotCoordinates(mouseEvent || lastPlotMouseEvent);
      if (!coords) {
        showToast("Double-click inside plot");
        return;
      }
      lastAddEventAt = now;
      if (pointClickTimer) {
        clearTimeout(pointClickTimer);
        pointClickTimer = null;
      }
      setComponentValue({
        kind: "event_action",
        file_name: currentFileName,
        action: "add_event",
        time_s: coords.x,
        current_pA: coords.y,
        nonce: now,
      });
      showToast("Added");
    }

    function focusChart() {
      if (document.activeElement !== chart && chart.focus) {
        chart.focus({ preventScroll: true });
      }
    }

    function axisRange(axisName) {
      const axis = chart._fullLayout && chart._fullLayout[axisName];
      if (!axis || !Array.isArray(axis.range)) {
        return null;
      }
      const min = numberOrNull(axis.range[0]);
      const max = numberOrNull(axis.range[1]);
      if (min === null || max === null || max <= min) {
        return null;
      }
      return [min, max];
    }

    function panAxis(axisName, direction, fraction) {
      const range = axisRange(axisName);
      if (!range) {
        return null;
      }
      const delta = (range[1] - range[0]) * fraction * direction;
      return [range[0] + delta, range[1] + delta];
    }

    function handleAxisKeydown(event) {
      const key = event.key;
      if (!["ArrowLeft", "ArrowRight", "ArrowUp", "ArrowDown"].includes(key)) {
        return;
      }
      const fraction = event.shiftKey ? 0.24 : (event.altKey ? 0.025 : 0.08);
      const update = {};
      if (key === "ArrowLeft" || key === "ArrowRight") {
        const range = panAxis("xaxis", key === "ArrowLeft" ? -1 : 1, fraction);
        if (!range) {
          return;
        }
        update["xaxis.range[0]"] = range[0];
        update["xaxis.range[1]"] = range[1];
      } else {
        const range = panAxis("yaxis", key === "ArrowUp" ? 1 : -1, fraction);
        if (!range) {
          return;
        }
        update["yaxis.range[0]"] = range[0];
        update["yaxis.range[1]"] = range[1];
      }
      event.preventDefault();
      event.stopPropagation();
      Plotly.relayout(chart, update);
    }

    function attachPlotHandlers() {
      if (plotHandlersAttached) {
        return;
      }
      plotHandlersAttached = true;

      chart.addEventListener("mouseenter", focusChart);
      chart.addEventListener("mousedown", focusChart, true);
      chart.addEventListener("keydown", handleAxisKeydown);
      chart.addEventListener("mousedown", rememberPlotMouseEvent, true);
      chart.addEventListener("click", rememberPlotMouseEvent, true);

      chart.on("plotly_click", function(eventData) {
        const match = firstEventPoint(eventData && eventData.points);
        if (!match) {
          return;
        }
        const mouseEvent = eventData.event || {};
        if (pointClickTimer) {
          clearTimeout(pointClickTimer);
        }
        pointClickTimer = setTimeout(function() {
          const payload = {
            kind: "event_action",
            file_name: currentFileName,
            action: "toggle_event",
            event_index: match.meta.event_index,
            time_s: match.meta.time_s,
            nonce: Date.now(),
          };
          const willAccept = match.meta.accepted === false;
          setComponentValue(payload);
          showToast(willAccept ? "Restored" : (match.meta.manual ? "Removed" : "Rejected"));
          pointClickTimer = null;
        }, 220);
        if (mouseEvent.preventDefault) {
          mouseEvent.preventDefault();
        }
      });

      chart.addEventListener("dblclick", function(mouseEvent) {
        rememberPlotMouseEvent(mouseEvent);
        addEventFromMouse(mouseEvent);
        if (mouseEvent.preventDefault) {
          mouseEvent.preventDefault();
        }
        if (mouseEvent.stopPropagation) {
          mouseEvent.stopPropagation();
        }
      }, true);

      chart.on("plotly_doubleclick", function(eventData) {
        addEventFromMouse((eventData && eventData.event) || lastPlotMouseEvent);
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
            if col == 'accepted':
                out[col] = True
            elif col == 'manual':
                out[col] = False
            else:
                out[col] = np.nan
    for col in ['time_s', 'amplitude_pA', 'prominence', 'iei_s']:
        out[col] = pd.to_numeric(out[col], errors='coerce')
    out['accepted'] = out['accepted'].fillna(True).astype(bool)
    out['manual'] = out['manual'].fillna(False).astype(bool)
    return out[EVENT_COLUMNS]

def recompute_event_stats(events_df):
    out = normalize_events_frame(events_df).sort_values('time_s').reset_index(drop=True)
    if out.empty:
        return out
    out['iei_s'] = np.nan
    acc_index = out.index[out['accepted'] == True]
    out.loc[acc_index, 'iei_s'] = out.loc[acc_index, 'time_s'].diff()
    return out

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
    for col in ['accepted', 'manual']:
        if not np.array_equal(before[col].to_numpy(dtype=bool), after[col].to_numpy(dtype=bool)):
            return True
    return False

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
    all_events = normalize_events_frame(all_events)

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
    edited = edited[~((edited['manual'] == True) & (edited['accepted'] != True))]
    outside = all_events.drop(index=source.index, errors='ignore')
    next_events = recompute_event_stats(pd.concat([outside, edited], ignore_index=True))
    if events_frame_changed(all_events, next_events):
        events_by_file[file_name] = next_events
        bump_event_table_revision(file_name)

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

    if bool(events_df.at[event_index, 'manual']) and not accepted:
        events_df = events_df.drop(index=event_index)
        events_by_file[file_name] = recompute_event_stats(events_df)
        bump_event_table_revision(file_name)
        st.session_state['_last_chart_event_action'] = {
            'file_name': file_name,
            'event_index': event_index,
            'accepted': False,
            'removed': True,
        }
        return {'changed': True, 'accepted': False, 'removed': True}

    events_df.at[event_index, 'accepted'] = accepted
    events_by_file[file_name] = recompute_event_stats(events_df)
    bump_event_table_revision(file_name)
    st.session_state['_last_chart_event_action'] = {
        'file_name': file_name,
        'event_index': event_index,
        'accepted': accepted,
    }
    return {'changed': True, 'accepted': accepted}

def add_chart_event(file_name, time_s):
    try:
        time_s = float(time_s)
    except (TypeError, ValueError):
        return None

    files = st.session_state.get('files', {})
    fdata = files.get(file_name)
    if not fdata:
        return None
    df_all = fdata.get('df')
    if not isinstance(df_all, pd.DataFrame) or df_all.empty:
        return None

    settings = st.session_state.get('settings', {}).get(file_name, {})
    sweeps = sorted(df_all['sweep'].unique().tolist())
    sweep = settings.get('sweep', sweeps[0] if sweeps else None)
    if sweep is None:
        return None
    t_start = float(settings.get('t_start', df_all['time_s'].min()))
    t_end = float(settings.get('t_end', df_all['time_s'].max()))
    sub = df_all[(df_all['sweep'] == sweep) & (df_all['time_s'] >= t_start) & (df_all['time_s'] <= t_end)].copy()
    if sub.empty:
        return None

    signal = sub['signal'].to_numpy()
    lp_hz = float(settings.get('lp_hz', 0) or 0)
    sample_rate_hz = float(fdata.get('meta', {}).get('sample_rate_hz', 0) or 0)
    if lp_hz > 0 and sample_rate_hz > 0 and len(signal) > 20:
        signal = gaussian_lowpass(signal, lp_hz, sample_rate_hz)

    times = sub['time_s'].to_numpy(dtype=float)
    nearest_pos = int(np.argmin(np.abs(times - time_s)))
    event_time = float(times[nearest_pos])
    signal_at_event = float(signal[nearest_pos])
    baseline_pct = float(settings.get('baseline_pct', 20) or 20)
    nbase = max(1, int(len(signal) * baseline_pct / 100))
    baseline = float(np.median(signal[:nbase]))
    amplitude = signal_at_event - baseline

    events_by_file = st.session_state.setdefault('events', {})
    events_df = normalize_events_frame(events_by_file.get(file_name, pd.DataFrame(columns=EVENT_COLUMNS)))
    new_event = pd.DataFrame([{
        'time_s': event_time,
        'amplitude_pA': amplitude,
        'prominence': abs(amplitude),
        'iei_s': np.nan,
        'accepted': True,
        'manual': True,
    }])
    events_by_file[file_name] = recompute_event_stats(pd.concat([events_df, new_event], ignore_index=True))
    bump_event_table_revision(file_name)
    return {'changed': True, 'time_s': event_time}

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
        if event.get('action') == 'add_event':
            add_chart_event(file_name, event.get('time_s'))
        else:
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

    raw_amps = y.iloc[corrected_idx] if isinstance(y, pd.Series) else y[corrected_idx]
    valid = np.array((raw_amps <= -prominence) if direction == 'inward (EPSC)' else (raw_amps >= prominence))

    if not np.any(valid):
        return pd.DataFrame(columns=['time_s', 'amplitude_pA', 'prominence', 'iei_s', 'accepted'])

    proms = props.get('prominences', np.full(len(idx), np.nan))

    corrected_idx = corrected_idx[valid]
    idx = idx[valid]
    if isinstance(raw_amps, pd.Series):
        raw_amps = raw_amps.to_numpy()
    raw_amps = raw_amps[valid]
    proms = proms[valid]

    peak_times = time_s[corrected_idx]
    iei = np.diff(peak_times)
    iei = np.concatenate([[np.nan], iei])

    return pd.DataFrame({
        'time_s': peak_times,
        'amplitude_pA': raw_amps,
        'prominence': proms,
        'iei_s': iei,
        'accepted': True,
    })

def summary_from_events(events_df, window_dur_s):
    events_df = normalize_events_frame(events_df)
    acc = events_df[events_df['accepted'] == True].sort_values('time_s') if not events_df.empty else events_df
    n = len(acc)
    n_manual = int((acc['manual'] == True).sum()) if n else 0
    n_detected = n - n_manual
    freq_hz = n / window_dur_s if window_dur_s > 0 else np.nan
    amp_abs = acc['amplitude_pA'].abs() if n else pd.Series(dtype=float)
    iei = acc['time_s'].diff() if n else pd.Series(dtype=float)
    return {
        'n_events': n,
        'n_detected_events': n_detected,
        'n_manual_events': n_manual,
        'freq_hz': freq_hz,
        'amp_mean_pA': amp_abs.mean() if n else np.nan,
        'amp_median_pA': amp_abs.median() if n else np.nan,
        'amp_sd_pA': amp_abs.std(ddof=1) if n > 1 else np.nan,
        'iei_mean_s': iei.mean() if n else np.nan,
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

# ───────────────────────────────────────────
#  ANALYTICS COUNTERS
# ───────────────────────────────────────────
import requests
import datetime

ABACUS_NAMESPACE = "syncapture_prod_v1"

@st.cache_data(ttl=300)
def get_counter(key):
    try:
        r = requests.get(f"https://abacus.jasoncameron.dev/get/{ABACUS_NAMESPACE}/{key}", timeout=2)
        if r.status_code == 200:
            return r.json().get('value', 0)
    except:
        pass
    return 0

def inc_counter(key):
    try:
        r = requests.get(f"https://abacus.jasoncameron.dev/hit/{ABACUS_NAMESPACE}/{key}", timeout=2)
        if r.status_code == 200:
            return r.json().get('value', 0)
    except:
        pass
    return 0

def record_event(event_type):
    today = datetime.date.today().isoformat()
    inc_counter(event_type)
    return inc_counter(f"{event_type}_{today}")

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

def get_figure_style():
    style = DEFAULT_FIGURE_STYLE.copy()
    style.update(st.session_state.get('figure_style', {}) or {})
    widget_key_map = {
        'fig_font_family': 'font_family',
        'fig_font_size': 'font_size',
        'fig_axis_label_size': 'axis_label_size',
        'fig_tick_label_size': 'tick_label_size',
        'fig_bar_width': 'bar_width',
        'fig_bar_line_width': 'bar_line_width',
        'fig_axis_line_width': 'axis_line_width',
        'fig_error_line_width': 'error_line_width',
        'fig_point_size': 'point_size',
        'fig_x_label_rotation': 'x_label_rotation',
        'fig_show_individual_names': 'show_individual_names',
        'fig_individual_label_size': 'individual_label_size',
        'fig_amp_y_auto': 'amp_y_auto',
        'fig_amp_y_min': 'amp_y_min',
        'fig_amp_y_max': 'amp_y_max',
        'fig_freq_y_auto': 'freq_y_auto',
        'fig_freq_y_min': 'freq_y_min',
        'fig_freq_y_max': 'freq_y_max',
    }
    for widget_key, style_key in widget_key_map.items():
        if widget_key in st.session_state:
            style[style_key] = st.session_state[widget_key]
    style['axis_color'] = DEFAULT_FIGURE_STYLE['axis_color']
    style['tick_color'] = DEFAULT_FIGURE_STYLE['tick_color']
    style['trace_line_color'] = DEFAULT_FIGURE_STYLE['trace_line_color']
    style['trace_line_width'] = DEFAULT_FIGURE_STYLE['trace_line_width']
    style['individual_label_color'] = DEFAULT_FIGURE_STYLE['individual_label_color']
    style['trace_height_px'] = DEFAULT_FIGURE_STYLE['trace_height_px']
    return style

def group_color_key(group):
    digest = hashlib.sha1(str(group).encode('utf-8')).hexdigest()[:10]
    return f'group_color_{digest}'

def default_group_color(group):
    if group in DEFAULT_GROUP_PALETTE:
        return DEFAULT_GROUP_PALETTE[group]
    digest = hashlib.sha1(str(group).encode('utf-8')).hexdigest()
    hue = int(digest[:2], 16) / 255
    palette = ['#0f766e', '#7c3aed', '#dc2626', '#2563eb', '#ca8a04', '#be185d', '#4b5563']
    return palette[int(hue * (len(palette) - 1))]

def group_colors_for(groups):
    colors = {}
    saved = st.session_state.setdefault('group_colors', {})
    for group in groups:
        colors[group] = saved.get(group) or default_group_color(group)
    return colors

def record_for_file(file_name):
    return next((r for r in st.session_state.get('records', []) if r.get('file_name') == file_name), {})

def format_trace_title(file_name, record=None):
    record = record or record_for_file(file_name)
    cell = normalize_group_name(record.get('cell_id')) or Path(file_name).stem
    individual = normalize_group_name(record.get('individual'))
    group = normalize_group_name(record.get('group'))
    title = cell
    if individual:
        title = f'{individual} - {title}'
    if group:
        title = f'{title} ({group})'
    return title

sync_plotly_chart_state()

def make_trace_figure(sub, events_df, settings, file_name, record=None, figure_style=None):
    figure_style = figure_style or get_figure_style()
    direction = settings.get('direction', 'inward (EPSC)')
    marker_color = '#1a6b55' if 'EPSC' in direction else '#b91c1c'
    fig, ax = plt.subplots(figsize=(11, 3.5))
    fig.patch.set_facecolor('white')
    ax.set_facecolor('white')
    ax.spines[['top', 'right']].set_visible(False)
    ax.spines[['left', 'bottom']].set_color(figure_style['axis_color'])
    ax.spines[['left', 'bottom']].set_linewidth(float(figure_style['axis_line_width']))
    ax.tick_params(colors=figure_style['tick_color'], labelsize=14, width=float(figure_style['axis_line_width']))
    fdata = S.files.get(file_name, {}) if 'files' in S else {}
    meta = fdata.get('meta', {})
    unit_x = meta.get('unit_x', 's')
    unit_y = meta.get('unit_y', 'pA')
    
    unit_y_lower = str(unit_y).lower()
    if 'v' in unit_y_lower:
        y_label = f'Voltage ({unit_y})'
    elif 'a' in unit_y_lower:
        y_label = f'Current ({unit_y})'
    else:
        y_label = f'Signal ({unit_y})'

    ax.set_xlabel(f'Time ({unit_x})', fontsize=18, color=figure_style['axis_color'], fontfamily=figure_style['font_family'])
    ax.set_ylabel(y_label, fontsize=18, color=figure_style['axis_color'], fontfamily=figure_style['font_family'])
    ax.set_title(format_trace_title(file_name, record), fontsize=18, color=figure_style['axis_color'], pad=8, fontfamily=figure_style['font_family'])
    if figure_style.get('show_grid'):
        ax.grid(True, color=figure_style['grid_color'], linewidth=0.6, alpha=0.8)
    if not sub.empty:
        bl_end = sub['time_s'].min() + (sub['time_s'].max() - sub['time_s'].min()) * settings.get('baseline_pct', 20) / 100
        ax.axvspan(sub['time_s'].min(), bl_end, color='#f3f4f6', alpha=0.5, label='baseline region', zorder=0)
        ax.plot(sub['time_s'], sub['signal'], lw=float(figure_style['trace_line_width']), color=figure_style['trace_line_color'], zorder=1)
        if events_df is not None and not events_df.empty:
            events_df = normalize_events_frame(events_df)
            nbase = max(1, int(len(sub) * settings.get('baseline_pct', 20) / 100))
            baseline = np.median(sub['signal'].values[:nbase])
            acc = events_df[events_df['accepted'] == True]
            detected_acc = acc[acc['manual'] != True]
            manual_acc = acc[acc['manual'] == True]
            rej = events_df[(events_df['accepted'] != True) & (events_df['manual'] != True)]
            if not detected_acc.empty:
                ax.scatter(detected_acc['time_s'], detected_acc['amplitude_pA'] + baseline, s=28, color=marker_color, zorder=3, label=f"{len(detected_acc)} detected")
            if not manual_acc.empty:
                ax.scatter(manual_acc['time_s'], manual_acc['amplitude_pA'] + baseline, s=34, color='#2563eb', zorder=4, marker='D', label=f"{len(manual_acc)} manual")
            if not rej.empty:
                ax.scatter(rej['time_s'], rej['amplitude_pA'] + baseline, s=18, color='#9ca3af', zorder=2, marker='x', label=f"{len(rej)} rejected")
        ax.legend(fontsize=max(7, int(figure_style['tick_label_size']) - 1), frameon=False, loc='upper right')
        if is_valid_y_range(settings.get('y_min'), settings.get('y_max')):
            ax.set_ylim(float(settings['y_min']), float(settings['y_max']))
    plt.tight_layout(pad=0.8)
    return fig

def fig_to_png_bytes(fig, tight=True):
    buf = io.BytesIO()
    save_kwargs = {'format': 'png', 'dpi': 150}
    if tight:
        save_kwargs['bbox_inches'] = 'tight'
    fig.savefig(buf, **save_kwargs)
    buf.seek(0)
    return buf.read()

def make_trace_figure_plotly(sub, events_df, settings, file_name, record=None, figure_style=None):
    """Interactive Plotly figure with drag-zoom, box-select, scroll-zoom and double-click reset."""
    figure_style = figure_style or get_figure_style()
    direction = settings.get('direction', 'inward (EPSC)')
    marker_color = '#1a6b55' if 'EPSC' in direction else '#b91c1c'
    xaxis_revision = f"{file_name}:{settings.get('sweep')}:{settings.get('t_start')}:{settings.get('t_end')}"
    yaxis_revision = f"{file_name}:{settings.get('y_min')}:{settings.get('y_max')}"
    fig = go.Figure()

    fdata = S.files.get(file_name, {}) if 'files' in S else {}
    meta = fdata.get('meta', {})
    unit_x = meta.get('unit_x', 's')
    unit_y = meta.get('unit_y', 'pA')
    
    unit_y_lower = str(unit_y).lower()
    if 'v' in unit_y_lower:
        y_label = f'Voltage ({unit_y})'
    elif 'a' in unit_y_lower:
        y_label = f'Current ({unit_y})'
    else:
        y_label = f'Signal ({unit_y})'
        
    x_label = f'Time ({unit_x})'

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
            mode='lines', line=dict(color=figure_style['trace_line_color'], width=float(figure_style['trace_line_width'])),
            name='Trace',
            hovertemplate=f'Time: %{{x:.4f}}{unit_x}<br>{y_label.split(" ")[0]}: %{{y:.2f}}{unit_y}<extra></extra>',
        ))
        if events_df is not None and not events_df.empty:
            events_df = normalize_events_frame(events_df)
            nbase = max(1, int(len(sub) * settings.get('baseline_pct', 20) / 100))
            baseline = float(np.median(sub['signal'].values[:nbase]))
            acc = events_df[events_df['accepted'] == True]
            detected_acc = acc[acc['manual'] != True]
            manual_acc = acc[acc['manual'] == True]
            rej = events_df[(events_df['accepted'] != True) & (events_df['manual'] != True)]
            if not detected_acc.empty:
                acc_custom = [[int(idx), 1, float(row['amplitude_pA']), 0] for idx, row in detected_acc.iterrows()]
                acc_ids = [str(int(idx)) for idx in detected_acc.index]
                fig.add_trace(go.Scatter(
                    x=detected_acc['time_s'], y=detected_acc['amplitude_pA'] + baseline,
                    mode='markers',
                    marker=dict(color=marker_color, size=8, line=dict(color='white', width=0.8)),
                    ids=acc_ids,
                    customdata=acc_custom,
                    name=f'{len(detected_acc)} detected',
                    hovertemplate=f'Time: %{{x:.4f}}{unit_x}<br>Amp: %{{customdata[2]:.2f}}{unit_y}<br>Click to reject<extra></extra>',
                ))
            if not manual_acc.empty:
                manual_custom = [[int(idx), 1, float(row['amplitude_pA']), 1] for idx, row in manual_acc.iterrows()]
                manual_ids = [str(int(idx)) for idx in manual_acc.index]
                fig.add_trace(go.Scatter(
                    x=manual_acc['time_s'], y=manual_acc['amplitude_pA'] + baseline,
                    mode='markers',
                    marker=dict(color='#2563eb', size=9, symbol='diamond', line=dict(color='white', width=0.8)),
                    ids=manual_ids,
                    customdata=manual_custom,
                    name=f'{len(manual_acc)} manual',
                    hovertemplate=f'Time: %{{x:.4f}}{unit_x}<br>Amp: %{{customdata[2]:.2f}}{unit_y}<br>Click to remove<extra></extra>',
                ))
            if not rej.empty:
                rej_custom = [[int(idx), 0, float(row['amplitude_pA']), int(bool(row['manual']))] for idx, row in rej.iterrows()]
                rej_ids = [str(int(idx)) for idx in rej.index]
                fig.add_trace(go.Scatter(
                    x=rej['time_s'], y=rej['amplitude_pA'] + baseline,
                    mode='markers',
                    marker=dict(color='#9ca3af', size=7, symbol='x', line=dict(width=1.5)),
                    ids=rej_ids,
                    customdata=rej_custom,
                    name=f'{len(rej)} rejected',
                    hovertemplate=f'Time: %{{x:.4f}}{unit_x}<br>Amp: %{{customdata[2]:.2f}}{unit_y}<br>Click to restore<extra></extra>',
                ))
    fig.update_layout(
        title=dict(text=format_trace_title(file_name, record), font=dict(size=18, color=figure_style['axis_color'], family=figure_style['font_family'])),
        xaxis=dict(
            title=dict(text=x_label, font=dict(size=18, color=figure_style['axis_color'], family=figure_style['font_family'])),
            tickfont=dict(size=14, color=figure_style['tick_color'], family=figure_style['font_family']),
            showgrid=bool(figure_style.get('show_grid')), gridcolor=figure_style['grid_color'], zeroline=False,
            linecolor=figure_style['axis_color'], linewidth=float(figure_style['axis_line_width']), mirror=False,
            uirevision=xaxis_revision,
        ),
        yaxis=dict(
            title=dict(text=y_label, font=dict(size=18, color=figure_style['axis_color'], family=figure_style['font_family'])),
            tickfont=dict(size=14, color=figure_style['tick_color'], family=figure_style['font_family']),
            showgrid=bool(figure_style.get('show_grid')), gridcolor=figure_style['grid_color'], zeroline=False,
            linecolor=figure_style['axis_color'], linewidth=float(figure_style['axis_line_width']), mirror=False,
            uirevision=yaxis_revision,
        ),
        plot_bgcolor='white', paper_bgcolor='white',
        height=int(figure_style.get('trace_height_px', 520)),
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

def make_summary_figure(clean, summary, figure_style, group_colors):
    fig_sum, axes = plt.subplots(1, 2, figsize=(10.5, 4.4), constrained_layout=False)
    fig_sum.patch.set_facecolor('white')
    order = sorted(clean['group'].dropna().unique())
    rng = np.random.default_rng(42)
    panels = [
        ('amp_mean_pA', 'Mean Amplitude (pA)', 'amp_mean', 'amp_sem', 'amp'),
        ('freq_hz', 'Frequency (Hz)', 'freq_mean', 'freq_sem', 'freq'),
    ]
    for ax_i, (col, label, mean_col, sem_col, scale_key) in enumerate(panels):
        ax = axes[ax_i]
        ax.set_facecolor('white')
        ax.spines[['top', 'right']].set_visible(False)
        ax.spines[['left', 'bottom']].set_color(figure_style['axis_color'])
        ax.spines[['left', 'bottom']].set_linewidth(float(figure_style['axis_line_width']))
        ax.tick_params(
            colors=figure_style['tick_color'],
            labelsize=int(figure_style['tick_label_size']),
            width=float(figure_style['axis_line_width']),
            direction='out',
        )
        ax.set_ylabel(
            label,
            fontsize=int(figure_style['axis_label_size']),
            color=figure_style['axis_color'],
            fontfamily=figure_style['font_family'],
        )
        if figure_style.get('show_grid'):
            ax.yaxis.grid(True, color=figure_style['grid_color'], linewidth=0.6, alpha=0.8)
            ax.set_axisbelow(True)

        xs = np.arange(len(order))
        individual_handles = []
        individual_seen = set()
        for xi, group in enumerate(order):
            gdf = clean[clean['group'] == group].copy()
            grp_color = group_colors.get(group, default_group_color(group))
            individuals = sorted([normalize_group_name(i) for i in gdf['individual'].dropna().unique().tolist() if normalize_group_name(i)])
            ind_shades = {}
            for ii, individual in enumerate(individuals):
                frac = 0.35 + 0.5 * (ii / max(1, len(individuals) - 1)) if len(individuals) > 1 else 0.6
                ind_shades[individual] = lighten_hex(grp_color, frac)

            gm = summary[summary['group'] == group]
            mean_val = gm[mean_col].values[0] if not gm.empty else np.nan
            sem_val = gm[sem_col].values[0] if not gm.empty else np.nan
            if pd.notna(mean_val):
                ax.bar(
                    xi,
                    mean_val,
                    yerr=sem_val if pd.notna(sem_val) else None,
                    capsize=4,
                    color=grp_color,
                    alpha=0.55,
                    edgecolor=grp_color,
                    linewidth=float(figure_style['bar_line_width']),
                    width=float(figure_style['bar_width']),
                    error_kw={'linewidth': float(figure_style['error_line_width'])},
                )

            point_rows = gdf[[col, 'individual', 'cell_id']].dropna(subset=[col])
            for _, row in point_rows.iterrows():
                point_x = xi + float(rng.uniform(-0.1, 0.1))
                individual = normalize_group_name(row.get('individual'))
                cell_label = normalize_group_name(row.get('cell_id'))
                label_text = individual or cell_label
                point_color = ind_shades.get(individual, lighten_hex(grp_color, 0.6))
                ax.scatter(
                    [point_x],
                    [float(row[col])],
                    color=point_color,
                    s=float(figure_style['point_size']),
                    zorder=4,
                    edgecolors=grp_color,
                    linewidths=0.6,
                )
                if figure_style.get('show_individual_names') and label_text:
                    legend_key = (group, label_text)
                    if legend_key not in individual_seen:
                        individual_seen.add(legend_key)
                        individual_handles.append(Line2D(
                            [0], [0],
                            marker='o',
                            linestyle='',
                            markersize=max(4, int(figure_style['individual_label_size']) * 0.75),
                            markerfacecolor=point_color,
                            markeredgecolor=grp_color,
                            markeredgewidth=0.6,
                            label=label_text,
                        ))

        ax.set_xticks(xs)
        ax.set_xticklabels(
            order,
            fontsize=int(figure_style['tick_label_size']),
            color=figure_style['tick_color'],
            rotation=int(figure_style['x_label_rotation']),
            fontfamily=figure_style['font_family'],
        )
        if len(order) > 0:
            x_margin = 0.9
            ax.set_xlim(-x_margin, (len(order) - 1) + x_margin)
        if figure_style.get('show_individual_names') and individual_handles:
            ax.legend(
                handles=individual_handles,
                loc='upper right',
                frameon=False,
                fontsize=int(figure_style['individual_label_size']),
                handlelength=0.8,
                handletextpad=0.35,
                borderpad=0.15,
                labelspacing=0.25,
            )
        auto_key = f'{scale_key}_y_auto'
        min_key = f'{scale_key}_y_min'
        max_key = f'{scale_key}_y_max'
        if not figure_style.get(auto_key, True) and is_valid_y_range(figure_style.get(min_key), figure_style.get(max_key)):
            ax.set_ylim(float(figure_style[min_key]), float(figure_style[max_key]))
    # Keep the two summary panels away from each other and from the image edges.
    fig_sum.subplots_adjust(left=0.18, right=0.82, bottom=0.18, top=0.88, wspace=0.38)
    return fig_sum

# ───────────────────────────────────────────
#  DONATION DIALOG
# ───────────────────────────────────────────
@st.dialog("☕ Support the Developer")
def show_donation_dialog():
    st.markdown("<p style='font-size: 1rem;'>Thank you for using <b>SynCapture</b>! Your support helps keep this project free and open-source.</p>", unsafe_allow_html=True)
    
    st.markdown("##### 🌍 International Users")
    st.markdown("""
    <a href="https://buymeacoffee.com/skyblingbling" target="_blank" style="
        display: flex;
        align-items: center;
        justify-content: center;
        width: 100%;
        background-color: #FFDD00;
        color: #000000;
        padding: 0.5rem 1rem;
        border-radius: 8px;
        text-decoration: none;
        font-weight: 600;
        font-size: 1rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    "><span style='font-size: 1.2em; margin-right: 6px;'>☕</span> Buy me a coffee</a>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    st.markdown("##### 🇨🇳 国内用户 (WeChat / Alipay)")
    import os
    if os.path.exists('donate.JPG'):
        _, center_col, _ = st.columns([1, 2, 1])
        with center_col:
            st.image('donate.JPG', use_container_width=True)
    elif os.path.exists('donate.jpg'):
        _, center_col, _ = st.columns([1, 2, 1])
        with center_col:
            st.image('donate.jpg', use_container_width=True)
    else:
        st.info("Donation QR code not found.")

# ───────────────────────────────────────────
#  SUGGESTION DIALOG
# ───────────────────────────────────────────
@st.dialog("💡 Send a Suggestion")
def show_suggestion_dialog():
    st.markdown("We'd love to hear your feedback or feature requests!")
    
    email = st.text_input("Your Email (optional)", placeholder="name@example.com")
    suggestion = st.text_area("Your Suggestion", height=150, placeholder="What can we improve?")
    
    if st.button("Submit", type='primary', use_container_width=True):
        if not suggestion.strip():
            st.error("Please enter a suggestion.")
        else:
            # TODO: Replace with your actual Formspree endpoint URL
            # e.g., "https://formspree.io/f/your_form_id"
            webhook_url = "https://formspree.io/f/mpqeqlwd" 
            
            try:
                import requests
                data = {
                    "email": email if email.strip() else "anonymous",
                    "message": suggestion
                }
                if "PLACEHOLDER" in webhook_url:
                    st.warning("⚠️ Developer note: Please replace the `webhook_url` in main.py with your real Formspree endpoint (formspree.io).")
                    st.success("Your suggestion was collected (Test Mode).")
                else:
                    response = requests.post(webhook_url, json=data, timeout=5)
                    if response.status_code in [200, 201]:
                        st.success("🎉 Thank you! Your suggestion has been sent directly to the developer.")
                    else:
                        st.error(f"Failed to send. Error code: {response.status_code}")
            except Exception as e:
                st.error(f"Network error: {e}")

# ───────────────────────────────────────────
#  ADMIN DASHBOARD
# ───────────────────────────────────────────
@st.dialog("📊 Admin Dashboard", width="large")
def show_admin_dashboard():
    st.markdown("### Daily Analytics (Last 7 Days)")
    import datetime
    import pandas as pd
    today = datetime.date.today()
    dates = [(today - datetime.timedelta(days=i)).isoformat() for i in range(7)]
    dates.reverse()
    
    visits_data = []
    likes_data = []
    
    for d in dates:
        visits_data.append(get_counter(f"visits_{d}"))
        likes_data.append(get_counter(f"likes_{d}"))
        
    df = pd.DataFrame({
        "Date": dates,
        "Visits": visits_data,
        "Likes": likes_data
    })
    
    st.bar_chart(df.set_index("Date"))
    st.dataframe(df, use_container_width=True)
    
    st.markdown("### All-Time Totals")
    c1, c2 = st.columns(2)
    c1.metric("Total Visits", get_counter("visits"))
    c2.metric("Total Likes", get_counter("likes"))

# ───────────────────────────────────────────
#  SIDEBAR — controls, file upload, params
# ───────────────────────────────────────────
_run_detect = False
_clear_win = False
_save_btn = False

if 'has_visited' not in S:
    S.has_visited = True
    S.visits_count = inc_counter('visits')
else:
    if 'visits_count' not in S:
        S.visits_count = get_counter('visits')

if 'likes_count' not in S:
    S.likes_count = get_counter('likes')

with st.sidebar:
    st.markdown("""
    <div style='display:flex;align-items:center;gap:12px;padding:0 0 2px 0;margin-bottom:16px'>
        <svg width="48" height="48" viewBox="0 0 36 36" fill="none" xmlns="http://www.w3.org/2000/svg">
          <rect width="36" height="36" rx="8" fill="#1a6b55"/>
          <polyline points="4,18 10,18 14,8 18,28 22,14 26,18 32,18" stroke="white" stroke-width="2.2" stroke-linejoin="round" stroke-linecap="round" fill="none"/>
        </svg>
        <div>
            <div style='font-size:26px;font-weight:700;color:inherit;line-height:1.1'>SynCapture</div>
            <div style='font-size:18px;color:color-mix(in srgb,currentColor 65%,transparent)'>Synaptic Event Analysis</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    upload_expander_state = len(S.file_order) == 0
    with st.expander("📂 Upload ABF Files", expanded=upload_expander_state):
        st.markdown("<div style='height:0.2rem'></div>", unsafe_allow_html=True)
        uploaded = st.file_uploader('Select or drag files here', type=['abf'], accept_multiple_files=True, label_visibility='collapsed')
        
    if uploaded:
        loaded_new_file = False
        for f in uploaded:
            if f.name not in S.files:
                try:
                    abf, tmp = load_abf(f)
                    df_sw = abf_to_sweeps(abf)
                    # Extract units
                    unit_y = 'pA'
                    if hasattr(abf, 'sweepUnitsY'):
                        unit_y = abf.sweepUnitsY
                    elif hasattr(abf, 'adcUnits'):
                        unit_y = abf.adcUnits[0] if isinstance(abf.adcUnits, list) and len(abf.adcUnits) > 0 else abf.adcUnits
                    if not isinstance(unit_y, str):
                        unit_y = str(unit_y)

                    unit_x = 's'
                    if hasattr(abf, 'sweepUnitsX'):
                        unit_x = abf.sweepUnitsX
                    if not isinstance(unit_x, str):
                        unit_x = str(unit_x)

                    S.files[f.name] = {
                        'abf_path': str(tmp),
                        'meta': {
                            'file_name': f.name,
                            'sample_rate_hz': float(abf.dataRate),
                            'sweep_count': len(abf.sweepList),
                            'protocol': getattr(abf, 'protocol', ''),
                            'unit_x': unit_x,
                            'unit_y': unit_y,
                            'duration_s': float(df_sw.groupby('sweep')['time_s'].max().iloc[0]) if not df_sw.empty else 0.0,
                        },
                        'df': df_sw,
                    }
                    if f.name not in S.file_order:
                        S.file_order.append(f.name)
                    if f.name not in S.events:
                        S.events[f.name] = pd.DataFrame(columns=EVENT_COLUMNS)
                    if f.name not in S.settings:
                        S.settings[f.name] = {}
                    loaded_new_file = True
                except Exception as e:
                    st.error(f'{f.name}: {e}')
        if loaded_new_file:
            st.rerun()

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

        st.markdown("<p style='font-size:0.75rem;font-weight:600;color:color-mix(in srgb,currentColor 65%,transparent);text-transform:uppercase;letter-spacing:0.5px;margin:0.3rem 0 0.3rem 0'>Sweep & Window</p>", unsafe_allow_html=True)
        sw_col1, sw_col2 = st.columns(2)
        with sw_col1:
            sweep = st.selectbox('Sweep', sweeps_available, index=sweeps_available.index(default_sweep) if default_sweep in sweeps_available else 0, key=f'sweep_{S.active}')
        with sw_col2:
            lp_hz = st.number_input('Low-pass (Hz)', min_value=0.0, value=float(prev.get('lp_hz', 1000.0)), step=50.0, help='Gaussian filter, 0 = off', key=f'lp_hz_{S.active}')
        sweep_df = df_all[df_all['sweep'] == sweep].copy()
        t_min, t_max = float(sweep_df['time_s'].min()), float(sweep_df['time_s'].max())

        sc1, sc2 = st.columns(2)
        with sc1:
            t_start = st.number_input('Start (s)', min_value=0.0, max_value=t_max, value=float(prev.get('t_start', t_min)), step=0.1, key=f't_start_{S.active}')
        with sc2:
            t_end = st.number_input('End (s)', min_value=0.0, max_value=t_max, value=float(prev.get('t_end', t_max)), step=0.1, key=f't_end_{S.active}')

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

        st.markdown("<p style='font-size:0.75rem;font-weight:600;color:color-mix(in srgb,currentColor 65%,transparent);text-transform:uppercase;letter-spacing:0.5px;margin:0.3rem 0 0.3rem 0'>Y Scale</p>", unsafe_allow_html=True)
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

        st.markdown("<p style='font-size:0.75rem;font-weight:600;color:color-mix(in srgb,currentColor 65%,transparent);text-transform:uppercase;letter-spacing:0.5px;margin:0.3rem 0 0.3rem 0'>Detection</p>", unsafe_allow_html=True)
        direction = st.selectbox('Direction', ['inward (EPSC)', 'outward (IPSC)'], index=0, key='global_direction')
        baseline_pct = st.slider('Baseline %', 5, 50, 20, 5, key='global_bl_pct', help="Initial percentage of the trace used to calculate the 0 pA baseline.")

        pc1, pc2 = st.columns(2)
        with pc1:
            prominence = st.number_input('Prom. (pA)', min_value=0.5, value=8.0, step=0.5, key='global_prominence', help="Minimum amplitude a peak must stand out above the surrounding background noise.")
            tau_rise = st.number_input('Tau Rise (ms)', min_value=0.1, value=0.5, step=0.1, key='global_tau_rise', help="Time constant for the rising phase of the template waveform (e.g., 0.5 ms for AMPA, 1.0 ms for GABA).")
        with pc2:
            distance_ms = st.number_input('Min IEI (ms)', min_value=0.1, value=5.0, step=0.5, key='global_distance_ms', help="Minimum inter-event interval; the shortest allowed time between two consecutive peaks.")
            tau_decay = st.number_input('Tau Decay (ms)', min_value=0.5, value=3.0, step=0.5, key='global_tau_decay', help="Time constant for the decaying phase of the template waveform (e.g., 3.0 ms for AMPA, 8.0 ms for GABA).")

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
            group = st.text_input('Group name', value=saved_group or 'naive', key=f'group_name_{S.active}', placeholder='type a group name')
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

    st.markdown("""
    <style>
    [data-testid="stSidebar"] [data-testid="stButton"] button,
    [data-testid="stSidebar"] [data-testid="stLinkButton"] a {
        min-height: 2.2rem !important;
        height: 2.2rem !important;
        padding: 0.1rem 0.2rem !important;
        white-space: nowrap !important;
        overflow: hidden !important;
        text-overflow: ellipsis !important;
        display: flex !important;
        align-items: center !important;
        justify-content: center !important;
    }
    [data-testid="stSidebar"] [data-testid="stButton"] button p,
    [data-testid="stSidebar"] [data-testid="stLinkButton"] a p {
        margin: 0 !important;
        padding: 0 !important;
        line-height: 1 !important;
    }
    </style>
    """, unsafe_allow_html=True)

    with st.expander("🎵 Background Music (007 Theme)", expanded=False):
        st.markdown("<p style='font-size:0.75rem;color:gray;margin-bottom:0.4rem;line-height:1.2'>Listen to the James Bond Theme or paste your own MP3 URL! (Maintains playback position across uploads/actions)</p>", unsafe_allow_html=True)
        music_url = st.text_input("Audio URL", "https://archive.org/download/tvtunes_4619/James%20Bond.mp3", label_visibility="collapsed")
        
        player_html = f"""
        <div style="font-family: -apple-system, BlinkMacSystemFont, sans-serif; font-size: 0.8rem; color: #374151; display: flex; flex-direction: column; gap: 8px;">
            <audio id="bg-audio" loop preload="auto">
                <source src="{music_url}" type="audio/mp3">
            </audio>
            <div style="display: flex; align-items: center; gap: 10px; margin-top: 2px;">
                <button id="play-btn" style="background: #1a6b55; color: white; border: none; padding: 5px 12px; border-radius: 4px; cursor: pointer; font-size: 0.75rem; font-weight: 600; min-width: 60px; outline: none;">Play</button>
                <span id="status" style="font-size: 0.75rem; color: #6b7280; font-weight: 500;">Stopped</span>
                <div style="display: flex; align-items: center; gap: 6px; margin-left: auto; padding-right: 5px;">
                    <span style="font-size: 0.7rem; color: #9ca3af; user-select: none;">Vol</span>
                    <input id="volume-slider" type="range" min="0" max="1" step="0.05" value="0.5" style="width: 70px; height: 4px; cursor: pointer; -webkit-appearance: none; background: #e5e7eb; border-radius: 2px; outline: none;">
                </div>
            </div>
            <style>
                #volume-slider::-webkit-slider-thumb {{
                    -webkit-appearance: none;
                    appearance: none;
                    width: 10px;
                    height: 10px;
                    border-radius: 50%;
                    background: #1a6b55;
                    cursor: pointer;
                }}
                #volume-slider::-moz-range-thumb {{
                    width: 10px;
                    height: 10px;
                    border-radius: 50%;
                    background: #1a6b55;
                    cursor: pointer;
                }}
            </style>
            <script>
                const audio = document.getElementById('bg-audio');
                const playBtn = document.getElementById('play-btn');
                const status = document.getElementById('status');
                const volumeSlider = document.getElementById('volume-slider');

                // Check URL change
                const currentUrl = "{music_url}";
                const savedUrl = localStorage.getItem('syncapture_music_url');
                let savedTime = localStorage.getItem('syncapture_music_time');
                
                if (savedUrl !== currentUrl) {{
                    localStorage.setItem('syncapture_music_url', currentUrl);
                    localStorage.setItem('syncapture_music_time', '0');
                    savedTime = '0';
                }}

                const savedPlaying = localStorage.getItem('syncapture_music_playing');
                const savedVolume = localStorage.getItem('syncapture_music_volume');

                if (savedVolume) {{
                    audio.volume = parseFloat(savedVolume);
                    volumeSlider.value = savedVolume;
                }} else {{
                    audio.volume = 0.5;
                }}

                function restoreTime() {{
                    if (savedTime && isFinite(savedTime) && parseFloat(savedTime) > 0) {{
                        try {{
                            audio.currentTime = parseFloat(savedTime);
                            savedTime = null; // restore once
                        }} catch (e) {{
                            console.log('Error seeking:', e);
                        }}
                    }}
                }}

                function tryAutoplay() {{
                    restoreTime();
                    if (savedPlaying === 'true') {{
                        audio.play().then(() => {{
                            playBtn.textContent = 'Pause';
                            status.textContent = 'Playing';
                        }}).catch(err => {{
                            console.log('Autoplay blocked. Click Play.', err);
                            playBtn.textContent = 'Play';
                            status.textContent = 'Click Play';
                        }});
                    }}
                }}

                audio.addEventListener('loadedmetadata', restoreTime);
                audio.addEventListener('canplay', restoreTime);
                
                // Try immediate autoplay
                tryAutoplay();

                playBtn.addEventListener('click', () => {{
                    if (audio.paused) {{
                        audio.play().then(() => {{
                            playBtn.textContent = 'Pause';
                            status.textContent = 'Playing';
                            localStorage.setItem('syncapture_music_playing', 'true');
                        }}).catch(err => {{
                            console.error('Play failed:', err);
                        }});
                    }} else {{
                        audio.pause();
                        playBtn.textContent = 'Play';
                        status.textContent = 'Paused';
                        localStorage.setItem('syncapture_music_playing', 'false');
                    }}
                }});

                volumeSlider.addEventListener('input', (e) => {{
                    const vol = parseFloat(e.target.value);
                    audio.volume = vol;
                    localStorage.setItem('syncapture_music_volume', vol);
                }});

                // Save time periodically
                setInterval(() => {{
                    if (!audio.paused) {{
                        localStorage.setItem('syncapture_music_time', audio.currentTime);
                    }}
                }}, 800);
            </script>
        </div>
        """
        import streamlit.components.v1 as components
        components.html(player_html, height=52)

    st.markdown("""
    <div style='height:1rem'></div>
    <p style='font-size:0.85rem;font-weight:600;color:inherit;margin-bottom:0.4rem'>💖 Support & Feedback</p>
    """, unsafe_allow_html=True)
    
    fc1, fc2 = st.columns(2)
    with fc1:
        if st.button('👍 Like', use_container_width=True, key='like_btn'):
            record_event('likes')
            get_counter.clear()  # clear cache so others see it soon
            st.toast('Thank you for your support! ❤️', icon='🎉')
            st.rerun()
    with fc2:
        st.link_button('⭐ GitHub', 'https://github.com/JingjingCheng/syncapture', use_container_width=True)
        
    fc3, fc4 = st.columns(2)
    with fc3:
        if st.button('☕ Buy me a coffee', use_container_width=True):
            show_donation_dialog()
    with fc4:
        if st.button('💡 Suggestions', use_container_width=True):
            show_suggestion_dialog()
            
    if st.query_params.get("admin") == "syncapture":
        st.markdown("<div style='height:1rem'></div>", unsafe_allow_html=True)
        if st.button("📊 Open Admin Dashboard", type="primary", use_container_width=True):
            show_admin_dashboard()

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
        <div style='font-size:1.5rem;font-weight:700;color:inherit;margin-bottom:4px'>SynCapture</div>
        <p style='color:color-mix(in srgb,currentColor 65%,transparent);font-size:0.95rem'>Upload ABF files in the sidebar to begin analysis</p>
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

    current_events = normalize_events_frame(S.events.get(S.active, pd.DataFrame(columns=EVENT_COLUMNS)))

    # ---- handle sidebar actions ----
    if _run_detect and not sub.empty:
        new_ev = detect_synaptic_events(sub['signal'].to_numpy(), sub['time_s'].to_numpy(), direction, prominence, distance_ms, baseline_pct, tau_rise, tau_decay)
        outside = current_events[(current_events['time_s'] < t_start) | (current_events['time_s'] > t_end)] if not current_events.empty else pd.DataFrame(columns=new_ev.columns)
        S.events[S.active] = recompute_event_stats(pd.concat([outside, new_ev], ignore_index=True))
        bump_event_table_revision(S.active)
        st.rerun()
    if _clear_win:
        if not current_events.empty:
            S.events[S.active] = recompute_event_stats(current_events[(current_events['time_s'] < t_start) | (current_events['time_s'] > t_end)])
            bump_event_table_revision(S.active)
        st.rerun()
    if _save_btn:
        all_ev_s = normalize_events_frame(S.events.get(S.active, pd.DataFrame(columns=EVENT_COLUMNS)))
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

    figure_style = get_figure_style()

    # ---- interactive trace chart ----
    win_events = current_events[(current_events['time_s'] >= t_start) & (current_events['time_s'] <= t_end)] if not current_events.empty else current_events
    fig_plotly = make_trace_figure_plotly(sub, win_events if not win_events.empty else None, S.settings[S.active], S.active, record_for_file(S.active), figure_style)
    plotly_relayout_chart(
        fig_plotly,
        S.active,
        config={'scrollZoom': True, 'displayModeBar': True, 'doubleClick': False, 'modeBarButtonsToAdd': ['drawrect', 'eraseshape']},
        height=int(figure_style.get('trace_height_px', 520)),
    )

    # ---- metrics row ----
    full_ev = S.events.get(S.active, pd.DataFrame())
    full_ev = normalize_events_frame(full_ev)
    dur = max(0.001, t_end - t_start)
    window_full_ev = full_ev[(full_ev['time_s'] >= t_start) & (full_ev['time_s'] <= t_end)] if not full_ev.empty else pd.DataFrame(columns=EVENT_COLUMNS)
    sm = summary_from_events(window_full_ev, dur)
    metric_items = [
        ('Events', sm['n_events']),
        ('Manual', sm['n_manual_events']),
        ('Freq (Hz)', f"{sm['freq_hz']:.4f}" if pd.notna(sm['freq_hz']) else '—'),
        ('Mean |Amp|', f"{sm['amp_mean_pA']:.2f} pA" if pd.notna(sm['amp_mean_pA']) else '—'),
        ('Med |Amp|', f"{sm['amp_median_pA']:.2f} pA" if pd.notna(sm['amp_median_pA']) else '—'),
        ('Mean IEI', f"{sm['iei_mean_s']:.4f} s" if pd.notna(sm['iei_mean_s']) else '—'),
    ]
    metric_cards = ''.join(
        f"<div class='metric-inline-card'><span class='metric-inline-label'>{label}</span><span class='metric-inline-value'>{value}</span></div>"
        for label, value in metric_items
    )
    st.markdown(f"<div class='metric-inline-row'>{metric_cards}</div>", unsafe_allow_html=True)

    # ---- editable event table ----
    if not S.events.get(S.active, pd.DataFrame()).empty:
        all_ev = normalize_events_frame(S.events[S.active])
        win_ev = all_ev[(all_ev['time_s'] >= t_start) & (all_ev['time_s'] <= t_end)].copy()
        total_acc = int((all_ev['accepted'] == True).sum()) if 'accepted' in all_ev.columns else 0
        total_manual = int(((all_ev['accepted'] == True) & (all_ev['manual'] == True)).sum()) if 'accepted' in all_ev.columns else 0
        total_rej = int(((all_ev['accepted'] != True) & (all_ev['manual'] != True)).sum()) if 'accepted' in all_ev.columns else 0
        table_revision = S.event_table_revisions.get(S.active, 0)
        editor_key = f'ev_table_{S.active}_{table_revision}'
        editor_source_key = f'{editor_key}_source'
        S[editor_source_key] = win_ev.copy()
        with st.expander(f'📋 {len(win_ev)} events in window · {total_acc} accepted · {total_manual} manual · {total_rej} rejected total', expanded=False):
            st.data_editor(
                win_ev, num_rows='dynamic', use_container_width=True, key=editor_key,
                column_config={
                    'manual': None,
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
    with st.expander(f'📊 Summary & Export ({len(S.records)} files)', expanded=False):
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
            all_groups = sorted(clean['group'].unique())
            figure_style = get_figure_style()
            amp_y_min_default, amp_y_max_default = infer_y_range(clean['amp_mean_pA'].dropna(), pad_frac=0.12)
            freq_y_min_default, freq_y_max_default = infer_y_range(clean['freq_hz'].dropna(), pad_frac=0.12)
            if not S.figure_style:
                figure_style['amp_y_min'] = min(0.0, float(amp_y_min_default))
                figure_style['amp_y_max'] = float(amp_y_max_default)
                figure_style['freq_y_min'] = min(0.0, float(freq_y_min_default))
                figure_style['freq_y_max'] = float(freq_y_max_default)

            with st.expander('Figure Style', expanded=True):
                style_tabs = st.tabs(['Text & Axes', 'Bars & Points', 'Y-Axis Ranges', 'Group Colors'])
                
                with style_tabs[0]:
                    t_cols = st.columns([1.5, 1, 1, 1, 1, 1])
                    with t_cols[0]:
                        fonts = ['Arial', 'Helvetica', 'DejaVu Sans', 'Times New Roman']
                        figure_style['font_family'] = st.selectbox('Font', fonts, index=fonts.index(figure_style['font_family']) if figure_style['font_family'] in fonts else 0, key='fig_font_family')
                    with t_cols[1]:
                        figure_style['font_size'] = st.number_input('Title Size', min_value=7, max_value=24, value=int(figure_style['font_size']), step=1, key='fig_font_size')
                    with t_cols[2]:
                        figure_style['axis_label_size'] = st.number_input('Axis Lbl', min_value=7, max_value=24, value=int(figure_style['axis_label_size']), step=1, key='fig_axis_label_size')
                    with t_cols[3]:
                        figure_style['tick_label_size'] = st.number_input('Tick Lbl', min_value=6, max_value=20, value=int(figure_style['tick_label_size']), step=1, key='fig_tick_label_size')
                    with t_cols[4]:
                        figure_style['axis_line_width'] = st.number_input('Axis Line', min_value=0.1, max_value=10.0, value=float(figure_style['axis_line_width']), step=0.1, key='fig_axis_line_width')
                    with t_cols[5]:
                        figure_style['x_label_rotation'] = st.number_input('X-Ang', min_value=-360, max_value=360, value=int(figure_style['x_label_rotation']), step=1, key='fig_x_label_rotation')

                with style_tabs[1]:
                    m_cols = st.columns([1, 1, 1, 1, 1.2, 1])
                    with m_cols[0]:
                        figure_style['bar_width'] = st.number_input('Bar W', min_value=0.01, max_value=2.0, value=float(figure_style['bar_width']), step=0.05, key='fig_bar_width')
                    with m_cols[1]:
                        figure_style['bar_line_width'] = st.number_input('Bar Line W', min_value=0.0, max_value=10.0, value=float(figure_style['bar_line_width']), step=0.1, key='fig_bar_line_width')
                    with m_cols[2]:
                        figure_style['error_line_width'] = st.number_input('Err Line W', min_value=0.1, max_value=10.0, value=float(figure_style['error_line_width']), step=0.1, key='fig_error_line_width')
                    with m_cols[3]:
                        figure_style['point_size'] = st.number_input('Point Size', min_value=1, max_value=200, value=int(figure_style['point_size']), step=1, key='fig_point_size')
                    with m_cols[4]:
                        st.markdown("<div style='margin-top:28px'></div>", unsafe_allow_html=True)
                        show_names = st.checkbox('Ind Legend', value=bool(figure_style.get('show_individual_names')), key='fig_show_individual_names')
                        figure_style['show_individual_names'] = show_names
                    with m_cols[5]:
                        figure_style['individual_label_size'] = st.number_input('Leg Text', min_value=5, max_value=18, value=int(figure_style['individual_label_size']), step=1, key='fig_individual_label_size', disabled=not bool(figure_style['show_individual_names']))

                with style_tabs[2]:
                    r_cols = st.columns(6)
                    with r_cols[0]:
                        st.markdown("<div style='margin-top:28px'></div>", unsafe_allow_html=True)
                        figure_style['amp_y_auto'] = st.checkbox('Amp Auto Y', value=bool(figure_style.get('amp_y_auto', True)), key='fig_amp_y_auto')
                    with r_cols[1]:
                        figure_style['amp_y_min'] = st.number_input('Amp Y min', value=float(figure_style.get('amp_y_min', amp_y_min_default)), step=1.0, key='fig_amp_y_min', disabled=bool(figure_style['amp_y_auto']))
                    with r_cols[2]:
                        figure_style['amp_y_max'] = st.number_input('Amp Y max', value=float(figure_style.get('amp_y_max', amp_y_max_default)), step=1.0, key='fig_amp_y_max', disabled=bool(figure_style['amp_y_auto']))
                    with r_cols[3]:
                        st.markdown("<div style='margin-top:28px'></div>", unsafe_allow_html=True)
                        figure_style['freq_y_auto'] = st.checkbox('Freq Auto Y', value=bool(figure_style.get('freq_y_auto', True)), key='fig_freq_y_auto')
                    with r_cols[4]:
                        figure_style['freq_y_min'] = st.number_input('Freq Y min', value=float(figure_style.get('freq_y_min', freq_y_min_default)), step=0.1, key='fig_freq_y_min', disabled=bool(figure_style['freq_y_auto']))
                    with r_cols[5]:
                        figure_style['freq_y_max'] = st.number_input('Freq Y max', value=float(figure_style.get('freq_y_max', freq_y_max_default)), step=0.1, key='fig_freq_y_max', disabled=bool(figure_style['freq_y_auto']))

                with style_tabs[3]:
                    c_cols = st.columns(min(8, max(1, len(all_groups))))
                    group_colors = group_colors_for(all_groups)
                    for i, group in enumerate(all_groups):
                        with c_cols[i % len(c_cols)]:
                            picked = st.color_picker(f'{group}', value=group_colors[group], key=group_color_key(group))
                            S.group_colors[group] = picked
                            group_colors[group] = picked

            S.figure_style = figure_style
            fig_sum = make_summary_figure(clean, summary, figure_style, group_colors)
            fig_sum_bytes = fig_to_png_bytes(fig_sum, tight=False)
            st.image(fig_sum_bytes, width='stretch')
            plt.close(fig_sum)

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
                if not ev_f.empty:
                    ev_f = normalize_events_frame(ev_f)
                    ev_f = ev_f[(ev_f['time_s'] >= ts) & (ev_f['time_s'] <= te)]
                fig_f = make_trace_figure(sub_f, ev_f, sett, fname, rec, figure_style)
                img_bytes[fname] = fig_to_png_bytes(fig_f)
                plt.close(fig_f)

            zbuf = io.BytesIO()
            with zipfile.ZipFile(zbuf, 'w', zipfile.ZIP_DEFLATED) as zf:
                zf.writestr('per_cell_summary.csv', df_rec.to_csv(index=False))
                zf.writestr('group_mean_sem.csv', summary.to_csv(index=False))
                zf.writestr('Prism_amplitude_pA.csv', prism_amp.to_csv(index=False))
                zf.writestr('Prism_frequency_Hz.csv', prism_freq.to_csv(index=False))
                zf.writestr('figures/summary_prism_style.png', fig_sum_bytes)
                if not events_all.empty:
                    zf.writestr('all_events.csv', events_all.to_csv(index=False))
                for fname, ibytes in img_bytes.items():
                    zf.writestr(f'traces/{Path(fname).stem}_detected.png', ibytes)
                zf.writestr('review_state.json', json.dumps(json_safe(S.records), indent=2))
            export_cols = st.columns([1.15, 1.0, 1.0, 1.0, 3.2], gap='small')
            with export_cols[0]:
                st.download_button('Download ZIP', data=zbuf.getvalue(), file_name='syncapture_exports.zip', mime='application/zip', type='primary')
            with export_cols[1]:
                st.download_button('Amplitude CSV', prism_amp.to_csv(index=False).encode(), file_name='Prism_amplitude_pA.csv', mime='text/csv')
            with export_cols[2]:
                st.download_button('Frequency CSV', prism_freq.to_csv(index=False).encode(), file_name='Prism_frequency_Hz.csv', mime='text/csv')
            with export_cols[3]:
                st.download_button('Summary CSV', df_rec.to_csv(index=False).encode(), file_name='per_cell_summary.csv', mime='text/csv')
