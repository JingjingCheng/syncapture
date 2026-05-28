#!/usr/bin/env python3
"""
SynCapture — Synaptic Event Analysis Tool
Run: streamlit run syncapture_updated.py
Dependencies: pip install streamlit pyabf scipy pandas matplotlib numpy
"""

import io, json, zipfile, tempfile
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.signal import find_peaks, butter, filtfilt
import streamlit as st

try:
    import pyabf
    HAS_PYABF = True
except Exception:
    HAS_PYABF = False

st.set_page_config(page_title='SynCapture', page_icon='⚡', layout='wide', initial_sidebar_state='collapsed')

st.markdown("""
<style>
.block-container {padding-top: 1rem; padding-bottom: 1rem; max-width: 1400px;}
div[data-testid="stMetric"] {background: #fff; border: 1px solid #e5e7eb; padding: 10px 12px; border-radius: 14px;}
div[data-testid="stFileUploader"] {background: #fff; border: 1px solid #e5e7eb; padding: 8px 10px; border-radius: 14px;}
</style>
""", unsafe_allow_html=True)


def _init():
    defaults = {
        'files': {},
        'file_order': [],
        'active': None,
        'skipped': set(),
        'events': {},
        'settings': {},
        'records': [],
        'template': None,
        'template_name': None,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v
_init()
S = st.session_state


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


def butter_lowpass(data, cutoff_hz, fs_hz, order=4):
    nyq = 0.5 * fs_hz
    if cutoff_hz >= nyq:
        return data
    b, a = butter(order, cutoff_hz / nyq, btype='low')
    return filtfilt(b, a, data)


def parse_atf_template(uploaded_file):
    text = uploaded_file.getvalue().decode(errors='ignore')
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    rows = []
    for ln in lines:
        if ln.startswith('ATF'):
            continue
        if 'Signals=' in ln or 'SweepStartTimesMS=' in ln:
            continue
        parts = [p for p in ln.replace(',', '\t').split('\t') if p.strip()]
        vals = []
        for p in parts:
            try:
                vals.append(float(p))
            except Exception:
                pass
        if len(vals) >= 2:
            rows.append(vals[:2])
    if len(rows) < 5:
        return None
    arr = np.array(rows, dtype=float)
    y = arr[:, 1]
    y = y - np.nanmean(y)
    sd = np.nanstd(y)
    if sd > 0:
        y = y / sd
    return y


def normalized_xcorr(window, template):
    w = np.asarray(window, dtype=float)
    t = np.asarray(template, dtype=float)
    if len(w) != len(t):
        return np.nan
    w = w - np.nanmean(w)
    ws = np.nanstd(w)
    ts = np.nanstd(t)
    if ws == 0 or ts == 0:
        return np.nan
    return np.mean((w / ws) * (t / ts))


def detect_synaptic_events(trace, time_s, direction, prominence, distance_ms, baseline_pct=20, template=None, corr_threshold=0.6, method='Threshold + Template'):
    n_base = max(1, int(len(trace) * baseline_pct / 100))
    baseline = np.median(trace[:n_base])
    y = trace - baseline
    dt = np.median(np.diff(time_s)) if len(time_s) > 1 else 0.0001
    distance_pts = max(1, int((distance_ms / 1000) / dt))
    y_detect = -y if direction == 'inward (EPSC)' else y
    idx_thr, props = find_peaks(y_detect, prominence=prominence, distance=distance_pts)

    template_hits = []
    if template is not None and len(template) >= 5:
        L = len(template)
        for i in range(0, len(y) - L):
            seg = y[i:i+L]
            corr = normalized_xcorr(-seg if direction == 'inward (EPSC)' else seg, template)
            if np.isfinite(corr) and corr >= corr_threshold:
                j = i + int(np.argmax(y_detect[i:i+L]))
                amp = y[j]
                template_hits.append((j, corr, amp))

    template_map = {}
    for j, corr, amp in template_hits:
        if j not in template_map or corr > template_map[j][0]:
            template_map[j] = (corr, amp)

    accepted_idx = []
    out_corr = []
    out_prom = []

    if method == 'Threshold only':
        for k, j in enumerate(idx_thr):
            accepted_idx.append(j)
            out_prom.append(props.get('prominences', np.full(len(idx_thr), np.nan))[k])
            out_corr.append(np.nan)
    elif method == 'Template only':
        sorted_hits = sorted(template_map.items(), key=lambda x: x[0])
        last = -10**9
        for j, (corr, amp) in sorted_hits:
            if j - last >= distance_pts:
                accepted_idx.append(j)
                out_prom.append(np.nan)
                out_corr.append(corr)
                last = j
    else:
        for k, j in enumerate(idx_thr):
            nearby = [kk for kk in template_map if abs(kk - j) <= distance_pts]
            if nearby:
                best = max(nearby, key=lambda kk: template_map[kk][0])
                accepted_idx.append(j)
                out_prom.append(props.get('prominences', np.full(len(idx_thr), np.nan))[k])
                out_corr.append(template_map[best][0])

    if len(accepted_idx) == 0:
        return pd.DataFrame(columns=['time_s', 'amplitude_pA', 'prominence', 'corr', 'accepted', 'iei_s'])

    peak_times = time_s[np.array(accepted_idx)]
    iei = np.diff(peak_times)
    iei = np.concatenate([[np.nan], iei])
    return pd.DataFrame({
        'time_s': peak_times,
        'amplitude_pA': y[np.array(accepted_idx)],
        'prominence': out_prom,
        'corr': out_corr,
        'iei_s': iei,
        'accepted': True,
    }).sort_values('time_s').reset_index(drop=True)


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

    bl_end = sub['time_s'].min() + (sub['time_s'].max() - sub['time_s'].min()) * settings.get('baseline_pct', 20) / 100
    ax.axvspan(sub['time_s'].min(), bl_end, color='#f3f4f6', alpha=0.5, label='baseline region', zorder=0)
    ax.plot(sub['time_s'], sub['signal'], lw=0.7, color='#374151', zorder=1)

    if events_df is not None and not events_df.empty:
        base_val = np.median(sub['signal'].values[:max(1, int(len(sub) * settings.get('baseline_pct', 20) / 100))])
        acc = events_df[events_df['accepted'] == True]
        rej = events_df[events_df['accepted'] != True]
        if not acc.empty:
            ax.scatter(acc['time_s'], acc['amplitude_pA'] + base_val, s=28, color=marker_color, zorder=3, label=f"{len(acc)} events")
        if not rej.empty:
            ax.scatter(rej['time_s'], rej['amplitude_pA'] + base_val, s=18, color='#9ca3af', zorder=2, marker='x', label='rejected')
        ax.legend(fontsize=8, frameon=False, loc='upper right')

    x0, x1 = settings.get('xlim', (sub['time_s'].min(), sub['time_s'].max()))
    y0, y1 = settings.get('ylim', (sub['signal'].min(), sub['signal'].max()))
    ax.set_xlim(x0, x1)
    ax.set_ylim(y0, y1)
    plt.tight_layout(pad=0.8)
    return fig


def fig_to_png_bytes(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=150, bbox_inches='tight')
    buf.seek(0)
    return buf.read()


col_logo, col_title = st.columns([1, 14])
with col_logo:
    st.markdown("<div style='width:42px;height:42px;border-radius:12px;background:#0f766e;color:white;display:flex;align-items:center;justify-content:center;font-weight:700'>⚡</div>", unsafe_allow_html=True)
with col_title:
    st.markdown("""
    <div style='padding-top:2px'>
      <div style='font-size:26px;font-weight:800;color:#111827'>SynCapture</div>
      <div style='font-size:13px;color:#6b7280'>Whole-cell patch-clamp synaptic event analysis</div>
    </div>
    """, unsafe_allow_html=True)

left, right = st.columns([0.9, 2.1], gap='large')

with left:
    st.subheader('Load files')
    uploaded_abfs = st.file_uploader('ABF files', type=['abf'], accept_multiple_files=True, label_visibility='collapsed')
    uploaded_template = st.file_uploader('Template ATF file', type=['atf', 'txt'], accept_multiple_files=False, help='Provide a clean representative event template in ATF format.')

    if uploaded_template is not None:
        tpl = parse_atf_template(uploaded_template)
        if tpl is None:
            st.error('Could not parse the template ATF file.')
        else:
            S.template = tpl
            S.template_name = uploaded_template.name
            st.success(f'Template loaded: {uploaded_template.name}')

    if uploaded_abfs:
        for uf in uploaded_abfs:
            if uf.name in S.files:
                continue
            abf, path = load_abf(uf)
            df_sweeps = abf_to_sweeps(abf)
            S.files[uf.name] = {
                'abf_path': str(path),
                'meta': {'sample_rate_hz': float(abf.dataRate), 'units_y': abf.sweepLabelY, 'units_x': abf.sweepLabelX, 'n_sweeps': len(abf.sweepList)},
                'df_sweeps': df_sweeps,
            }
            S.file_order.append(uf.name)
            if S.active is None:
                S.active = uf.name

    st.subheader('Files')
    for fname in S.file_order:
        if st.button(fname, key=f'file_{fname}', use_container_width=True):
            S.active = fname

with right:
    if S.active is None:
        st.info('Upload ABF files to begin.')
    else:
        fname = S.active
        payload = S.files[fname]
        df = payload['df_sweeps']
        sweep_ids = sorted(df['sweep'].unique().tolist())
        sett = S.settings.get(fname, {})

        top1, top2, top3, top4 = st.columns([1,1,1,1])
        with top1:
            sweep = st.selectbox('Sweep', sweep_ids, index=sweep_ids.index(sett.get('sweep', sweep_ids[0])) if sett.get('sweep', sweep_ids[0]) in sweep_ids else 0)
        with top2:
            direction = st.selectbox('Direction', ['inward (EPSC)', 'outward (IPSC)'], index=0 if sett.get('direction', 'inward (EPSC)') == 'inward (EPSC)' else 1)
        with top3:
            cutoff = st.number_input('Low-pass (Hz)', min_value=1, value=int(sett.get('cutoff', 1000)), step=100)
        with top4:
            baseline_pct = st.slider('Baseline %', min_value=1, max_value=50, value=int(sett.get('baseline_pct', 20)))

        sub = df[df['sweep'] == sweep].copy().reset_index(drop=True)
        fs = float(payload['meta']['sample_rate_hz'])
        sub['signal_filt'] = butter_lowpass(sub['signal'].values, cutoff, fs)

        detect1, detect2, detect3, detect4 = st.columns([1,1,1,1])
        with detect1:
            prominence = st.number_input('Threshold / prominence', min_value=0.1, value=float(sett.get('prominence', 5.0)), step=0.5)
        with detect2:
            distance_ms = st.number_input('Min distance (ms)', min_value=0.1, value=float(sett.get('distance_ms', 5.0)), step=0.5)
        with detect3:
            method = st.selectbox('Detection method', ['Threshold only', 'Template only', 'Threshold + Template'], index=['Threshold only', 'Template only', 'Threshold + Template'].index(sett.get('method', 'Threshold + Template')))
        with detect4:
            corr_threshold = st.number_input('Template corr', min_value=0.0, max_value=1.0, value=float(sett.get('corr_threshold', 0.60)), step=0.05)

        st.caption('Preview zoom works like Clampfit-style axis scaling: narrow the visible x/y window without changing the page layout.')
        zoom1, zoom2, zoom3, zoom4 = st.columns([1,1,1,1])
        tmin, tmax = float(sub['time_s'].min()), float(sub['time_s'].max())
        ymin0, ymax0 = float(sub['signal_filt'].min()), float(sub['signal_filt'].max())
        with zoom1:
            x_start = st.number_input('X start (s)', value=float(sett.get('x_start', tmin)), format='%.4f')
        with zoom2:
            x_end = st.number_input('X end (s)', value=float(sett.get('x_end', tmax)), format='%.4f')
        with zoom3:
            y_min = st.number_input('Y min (pA)', value=float(sett.get('y_min', ymin0)), format='%.3f')
        with zoom4:
            y_max = st.number_input('Y max (pA)', value=float(sett.get('y_max', ymax0)), format='%.3f')

        if x_end <= x_start:
            x_end = x_start + max(np.median(np.diff(sub['time_s'])), 0.001)
        if y_max <= y_min:
            y_max = y_min + 1

        zoom_reset = st.button('Reset preview zoom', key=f'reset_{fname}')
        if zoom_reset:
            x_start, x_end, y_min, y_max = tmin, tmax, ymin0, ymax0

        sett = {
            'sweep': sweep,
            'direction': direction,
            'cutoff': cutoff,
            'baseline_pct': baseline_pct,
            'prominence': prominence,
            'distance_ms': distance_ms,
            'method': method,
            'corr_threshold': corr_threshold,
            'x_start': x_start,
            'x_end': x_end,
            'y_min': y_min,
            'y_max': y_max,
            'xlim': (x_start, x_end),
            'ylim': (y_min, y_max),
        }
        S.settings[fname] = sett

        if method in ('Template only', 'Threshold + Template') and S.template is None:
            st.warning('Please provide a template ATF file to use template-assisted detection.')

        events_df = detect_synaptic_events(
            sub['signal_filt'].values,
            sub['time_s'].values,
            direction,
            prominence,
            distance_ms,
            baseline_pct,
            template=S.template,
            corr_threshold=corr_threshold,
            method=method,
        )
        S.events[fname] = events_df

        fig = make_trace_figure(sub.rename(columns={'signal_filt': 'signal'}), events_df, sett, fname)
        st.pyplot(fig, use_container_width=True)

        dur = float(sub['time_s'].max() - sub['time_s'].min())
        summary = summary_from_events(events_df, dur)
        m1, m2, m3, m4 = st.columns(4)
        m1.metric('Detected events', int(summary['n_events']))
        m2.metric('Frequency (Hz)', f"{summary['freq_hz']:.3f}" if pd.notna(summary['freq_hz']) else '—')
        m3.metric('Mean amp (pA)', f"{summary['amp_mean_pA']:.3f}" if pd.notna(summary['amp_mean_pA']) else '—')
        m4.metric('Template', S.template_name if S.template_name else 'none')

        st.subheader('Detected events')
        st.dataframe(events_df, use_container_width=True, height=260)

        rec = {
            'file': fname,
            'sweep': sweep,
            'direction': direction,
            'method': method,
            'template_file': S.template_name,
            'n_events': summary['n_events'],
            'freq_hz': summary['freq_hz'],
            'amp_mean_pA': summary['amp_mean_pA'],
            'amp_median_pA': summary['amp_median_pA'],
            'amp_sd_pA': summary['amp_sd_pA'],
            'iei_mean_s': summary['iei_mean_s'],
            'x_start_s': x_start,
            'x_end_s': x_end,
            'y_min_pA': y_min,
            'y_max_pA': y_max,
        }
        S.records = [r for r in S.records if not (r.get('file') == fname and r.get('sweep') == sweep)] + [rec]

        st.subheader('Exports')
        rec_df = pd.DataFrame(S.records).sort_values(['file', 'sweep']) if S.records else pd.DataFrame()
        ev_all = []
        for k, v in S.events.items():
            if v is not None and not v.empty:
                tmp = v.copy()
                tmp.insert(0, 'file', k)
                ev_all.append(tmp)
        ev_df = pd.concat(ev_all, ignore_index=True) if ev_all else pd.DataFrame()

        c1, c2 = st.columns(2)
        with c1:
            if not rec_df.empty:
                st.download_button('Download summary CSV', rec_df.to_csv(index=False).encode(), file_name='syncapture_summary.csv', mime='text/csv')
        with c2:
            if not ev_df.empty:
                st.download_button('Download events CSV', ev_df.to_csv(index=False).encode(), file_name='syncapture_events.csv', mime='text/csv')
