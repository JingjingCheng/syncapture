#!/usr/bin/env python3
"""
SynCapture — Synaptic Event Analysis Tool
Run: streamlit run syncapture.py
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
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300..700&display=swap');
html, body, [class*="css"] { font-family: 'Inter', sans-serif !important; }
:root {
  --bg:#ffffff; --surface:#f8f9fa; --border:rgba(0,0,0,0.08); --accent:#1a6b55;
  --accent-l:#eaf2ef; --text:#1a1a1a; --muted:#6b7280; --red:#b91c1c; --blue:#1d4ed8;
}
.main .block-container { padding: 1.5rem 2rem 2rem 2rem; max-width: 1280px; }
h1 { font-size: 1.4rem; font-weight: 700; letter-spacing: -0.02em; color: var(--text); }
h2 { font-size: 1.05rem; font-weight: 600; color: var(--text); margin-top: 1.2rem; }
h3 { font-size: 0.9rem; font-weight: 600; color: var(--muted); text-transform: uppercase; letter-spacing: 0.06em; margin-bottom: 0.4rem; }
.step-badge { display:inline-block; background:var(--accent); color:white; border-radius:50%; width:22px; height:22px; font-size:11px; font-weight:700; text-align:center; line-height:22px; margin-right:6px; }
.file-pill { display:inline-flex; align-items:center; gap:6px; background:var(--surface); border:1px solid var(--border); border-radius:20px; padding:3px 10px; font-size:0.78rem; margin:3px; }
.file-pill.active { background:var(--accent-l); border-color:var(--accent); font-weight:600; }
div[data-testid="stMetric"] { background:var(--surface); border:1px solid var(--border); border-radius:8px; padding:0.6rem 0.8rem; }
.stButton > button { border-radius:6px; font-size:0.82rem; font-weight:500; }
.stButton > button[kind="primary"] { background:var(--accent); border:none; }
.stButton > button:hover { opacity:0.88; }
hr { border:none; border-top:1px solid var(--border); margin:1.2rem 0; }
</style>
""", unsafe_allow_html=True)

def _init():
    defaults = {
        'files': {}, 'file_order': [], 'active': None, 'skipped': set(),
        'events': {}, 'settings': {}, 'records': [],
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
    if cutoff_hz <= 0 or cutoff_hz >= nyq:
        return data
    b, a = butter(order, cutoff_hz / nyq, btype='low')
    return filtfilt(b, a, data)

def detect_synaptic_events(trace, time_s, direction, prominence, distance_ms, baseline_pct=20):
    n_base = max(1, int(len(trace) * baseline_pct / 100))
    baseline = np.median(trace[:n_base])
    y = trace - baseline
    dt = np.median(np.diff(time_s)) if len(time_s) > 1 else 0.0001
    distance_pts = max(1, int((distance_ms / 1000) / dt))
    y_detect = -y if direction == 'inward (EPSC)' else y
    idx, props = find_peaks(y_detect, prominence=prominence, distance=distance_pts)
    if len(idx) == 0:
        return pd.DataFrame(columns=['time_s', 'amplitude_pA', 'prominence', 'iei_s', 'accepted'])
    peak_times = time_s[idx]
    iei = np.diff(peak_times)
    iei = np.concatenate([[np.nan], iei])
    return pd.DataFrame({
        'time_s': peak_times,
        'amplitude_pA': y[idx],
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
                ax.scatter(rej['time_s'], rej['amplitude_pA'] + baseline, s=18, color='#9ca3af', zorder=2, marker='x', label='rejected')
        ax.legend(fontsize=8, frameon=False, loc='upper right')
    plt.tight_layout(pad=0.8)
    return fig

def fig_to_png_bytes(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=150, bbox_inches='tight')
    buf.seek(0)
    return buf.read()

def lighten_hex(hex_color, frac):
    hex_color = hex_color.lstrip('#')
    r = int(hex_color[0:2], 16) / 255
    g = int(hex_color[2:4], 16) / 255
    b = int(hex_color[4:6], 16) / 255
    return (r * frac + (1 - frac), g * frac + (1 - frac), b * frac + (1 - frac))

col_logo, col_title = st.columns([1, 14])
with col_logo:
    st.markdown("""
    <svg width="36" height="36" viewBox="0 0 36 36" fill="none" xmlns="http://www.w3.org/2000/svg" aria-label="SynCapture logo">
      <rect width="36" height="36" rx="8" fill="#1a6b55"/>
      <polyline points="4,18 10,18 14,8 18,28 22,14 26,18 32,18" stroke="white" stroke-width="2.2" stroke-linejoin="round" stroke-linecap="round" fill="none"/>
    </svg>
    """, unsafe_allow_html=True)
with col_title:
    st.markdown("<h1 style='margin:0;padding-top:6px'>SynCapture <span style='font-weight:300;color:#6b7280;font-size:0.85rem;'>Synaptic Event Analysis</span></h1>", unsafe_allow_html=True)

st.markdown("<hr>", unsafe_allow_html=True)
st.markdown("<h2><span class='step-badge'>1</span>Upload ABF files</h2>", unsafe_allow_html=True)
uploaded = st.file_uploader('Drag & drop ABF files or click to browse', type=['abf'], accept_multiple_files=True, label_visibility='collapsed')

if uploaded:
    for f in uploaded:
        if f.name not in S.files:
            with st.spinner(f'Loading {f.name}…'):
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
    st.markdown("<h2><span class='step-badge'>2</span>File queue</h2>", unsafe_allow_html=True)
    queue_cols = st.columns([6, 2])
    with queue_cols[0]:
        pills_html = ''
        for name in S.file_order:
            active_cls = 'active' if name == S.active else ''
            skip_mark = '⊘ ' if name in S.skipped else ''
            pills_html += f"<span class='file-pill {active_cls}'>{skip_mark}{name}</span>"
        st.markdown(f"<div style='margin-bottom:8px'>{pills_html}</div>", unsafe_allow_html=True)
        selectable = [n for n in S.file_order if n not in S.skipped]
        if selectable:
            default_ix = selectable.index(S.active) if S.active in selectable else 0
            S.active = st.selectbox('Select file to review', selectable, index=default_ix, key='file_select', label_visibility='collapsed')
    with queue_cols[1]:
        if S.active and st.button('Skip this file'):
            S.skipped.add(S.active)
            remaining = [n for n in S.file_order if n not in S.skipped]
            S.active = remaining[0] if remaining else None
            st.rerun()

if S.active and S.active in S.files:
    fdata = S.files[S.active]
    meta = fdata['meta']
    df_all = fdata['df']
    sweeps_available = sorted(df_all['sweep'].unique().tolist())
    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown("<h2><span class='step-badge'>3</span>Select sweep & time window</h2>", unsafe_allow_html=True)
    col_sw, col_t0, col_t1, col_lp = st.columns([1, 1, 1, 1])
    prev = S.settings.get(S.active, {})
    default_sweep = prev.get('sweep', sweeps_available[0])
    with col_sw:
        sweep = st.selectbox('Sweep', sweeps_available, index=sweeps_available.index(default_sweep) if default_sweep in sweeps_available else 0, key=f'sweep_{S.active}')
    sweep_df = df_all[df_all['sweep'] == sweep].copy()
    t_min, t_max = float(sweep_df['time_s'].min()), float(sweep_df['time_s'].max())
    with col_t0:
        t_start = st.number_input('Start (s)', min_value=0.0, max_value=t_max, value=float(prev.get('t_start', t_min)), step=0.1, key=f't_start_{S.active}')
    with col_t1:
        t_end = st.number_input('End (s)', min_value=0.0, max_value=t_max, value=float(prev.get('t_end', t_max)), step=0.1, key=f't_end_{S.active}')
    with col_lp:
        lp_hz = st.number_input('Low-pass filter (Hz)', min_value=0.0, value=float(prev.get('lp_hz', 1000.0)), step=50.0, help='0 = no filter', key=f'lp_hz_{S.active}')
    if t_end <= t_start:
        st.error('End time must be greater than start time.')
        st.stop()
    sub = sweep_df[(sweep_df['time_s'] >= t_start) & (sweep_df['time_s'] <= t_end)].copy()
    if lp_hz > 0 and meta['sample_rate_hz'] > 0 and len(sub) > 20:
        sub['signal'] = butter_lowpass(sub['signal'].to_numpy(), lp_hz, meta['sample_rate_hz'])

    st.markdown("<h2><span class='step-badge'>4</span>Baseline & event detection</h2>", unsafe_allow_html=True)
    dc1, dc2, dc3, dc4 = st.columns([1, 1, 1, 1])
    with dc1:
        direction = st.selectbox('Current direction', ['inward (EPSC)', 'outward (IPSC)'], index=['inward (EPSC)', 'outward (IPSC)'].index(prev.get('direction', 'inward (EPSC)')) if prev.get('direction', 'inward (EPSC)') in ['inward (EPSC)', 'outward (IPSC)'] else 0, key=f'direction_{S.active}')
    with dc2:
        baseline_pct = st.slider('Baseline (first % of window)', 5, 50, int(prev.get('baseline_pct', 20)), 5, key=f'bl_pct_{S.active}')
    with dc3:
        prominence = st.number_input('Threshold prominence (pA)', min_value=0.5, value=float(prev.get('prominence', 8.0)), step=0.5, key=f'prom_{S.active}')
    with dc4:
        distance_ms = st.number_input('Min inter-event (ms)', min_value=0.1, value=float(prev.get('distance_ms', 5.0)), step=0.5, key=f'dist_{S.active}')
    S.settings[S.active] = {'direction': direction, 'baseline_pct': baseline_pct, 'prominence': prominence, 'distance_ms': distance_ms, 'sweep': sweep, 't_start': t_start, 't_end': t_end, 'lp_hz': lp_hz}

    current_events = S.events.get(S.active, pd.DataFrame(columns=['time_s','amplitude_pA','prominence','iei_s','accepted']))
    win_events = current_events[(current_events['time_s'] >= t_start) & (current_events['time_s'] <= t_end)] if not current_events.empty else current_events
    fig_trace = make_trace_figure(sub, win_events if not win_events.empty else None, S.settings[S.active], S.active)
    st.pyplot(fig_trace, clear_figure=True)
    plt.close(fig_trace)

    st.markdown("<h2><span class='step-badge'>5</span>Detect events & review</h2>", unsafe_allow_html=True)
    btn1, btn2, _ = st.columns([1, 1, 5])
    with btn1:
        run_detect = st.button('⚡ Detect events', type='primary', key=f'detect_{S.active}')
    with btn2:
        clear_win = st.button('Clear window', key=f'clear_win_{S.active}')
    if run_detect and not sub.empty:
        new_ev = detect_synaptic_events(sub['signal'].to_numpy(), sub['time_s'].to_numpy(), direction, prominence, distance_ms, baseline_pct)
        outside = current_events[(current_events['time_s'] < t_start) | (current_events['time_s'] > t_end)] if not current_events.empty else pd.DataFrame(columns=new_ev.columns)
        S.events[S.active] = pd.concat([outside, new_ev], ignore_index=True).sort_values('time_s').reset_index(drop=True)
        st.rerun()
    if clear_win:
        if not current_events.empty:
            S.events[S.active] = current_events[(current_events['time_s'] < t_start) | (current_events['time_s'] > t_end)].reset_index(drop=True)
        st.rerun()

    if not S.events.get(S.active, pd.DataFrame()).empty:
        all_ev = S.events[S.active].copy()
        win_ev = all_ev[(all_ev['time_s'] >= t_start) & (all_ev['time_s'] <= t_end)].copy()
        total_acc = int((all_ev['accepted'] == True).sum()) if 'accepted' in all_ev.columns else 0
        st.caption(f'{len(win_ev)} events in window | {total_acc} total accepted across file')
        edited = st.data_editor(
            win_ev, num_rows='dynamic', use_container_width=True, key=f'ev_table_{S.active}',
            column_config={
                'accepted': st.column_config.CheckboxColumn('Accept'),
                'time_s': st.column_config.NumberColumn('Time (s)', format='%.4f'),
                'amplitude_pA': st.column_config.NumberColumn('Amplitude (pA)', format='%.2f'),
                'prominence': st.column_config.NumberColumn('Prominence', format='%.2f'),
                'iei_s': st.column_config.NumberColumn('IEI (s)', format='%.4f'),
            }
        )
        outside = all_ev[(all_ev['time_s'] < t_start) | (all_ev['time_s'] > t_end)]
        S.events[S.active] = pd.concat([outside, edited], ignore_index=True).sort_values('time_s').reset_index(drop=True)

    full_ev = S.events.get(S.active, pd.DataFrame())
    if not full_ev.empty:
        dur = max(0.001, t_end - t_start)
        sm = summary_from_events(full_ev[(full_ev['time_s'] >= t_start) & (full_ev['time_s'] <= t_end)], dur)
        m1, m2, m3, m4, m5 = st.columns(5)
        m1.metric('Events (accepted)', sm['n_events'])
        m2.metric('Frequency (Hz)', f"{sm['freq_hz']:.4f}" if pd.notna(sm['freq_hz']) else '—')
        m3.metric('Mean |Amp| (pA)', f"{sm['amp_mean_pA']:.2f}" if pd.notna(sm['amp_mean_pA']) else '—')
        m4.metric('Median |Amp| (pA)', f"{sm['amp_median_pA']:.2f}" if pd.notna(sm['amp_median_pA']) else '—')
        m5.metric('Mean IEI (s)', f"{sm['iei_mean_s']:.4f}" if pd.notna(sm['iei_mean_s']) else '—')

    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown("<h2>Cell labels</h2>", unsafe_allow_html=True)
    lb1, lb2, lb3, lb4 = st.columns([2, 2, 1.5, 1.5])
    saved_rec = next((r for r in S.records if r.get('file_name') == S.active), {})
    with lb1:
        cell_id = st.text_input('Cell ID', value=saved_rec.get('cell_id', Path(S.active).stem), key=f'cell_id_{S.active}')
    with lb2:
        individual = st.text_input('Individual', value=saved_rec.get('individual', ''), key=f'individual_{S.active}')
    with lb3:
        group_options = ['naive', 'ovx', 'control', 'treatment', 'other']
        saved_group = saved_rec.get('group', 'naive')
        group = st.selectbox('Group', group_options, index=group_options.index(saved_group) if saved_group in group_options else 0, key=f'group_{S.active}')
    with lb4:
        status_options = ['accepted', 'needs_check', 'rejected']
        saved_status = saved_rec.get('status', 'accepted')
        status = st.selectbox('Status', status_options, index=status_options.index(saved_status) if saved_status in status_options else 0, key=f'status_{S.active}')
    if st.button('✓ Save this file to dataset', type='primary', key=f'save_{S.active}'):
        all_ev = S.events.get(S.active, pd.DataFrame())
        dur = max(0.001, t_end - t_start)
        sm = summary_from_events(all_ev[(all_ev['time_s'] >= t_start) & (all_ev['time_s'] <= t_end)] if not all_ev.empty else all_ev, dur)
        rec = {
            'file_name': S.active, 'cell_id': cell_id, 'individual': individual, 'group': group,
            'status': status, 'sweep': sweep, 'window_start_s': t_start, 'window_end_s': t_end, 'window_dur_s': dur,
            **sm, 'lp_hz': lp_hz, 'direction': direction, 'sample_rate_hz': meta['sample_rate_hz'],
        }
        S.records = [r for r in S.records if r.get('file_name') != S.active]
        S.records.append(rec)
        st.success(f'Saved {cell_id} → dataset ({len(S.records)} files total).')

if S.records:
    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown("<h2><span class='step-badge'>6</span>Summary & export</h2>", unsafe_allow_html=True)
    df_rec = pd.DataFrame(S.records).sort_values(['group', 'individual', 'cell_id'])
    st.dataframe(df_rec, use_container_width=True)
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
            fdata = S.files.get(fname)
            if not fdata:
                continue
            df_all2 = fdata['df']
            sett = S.settings.get(fname, {})
            sw = sett.get('sweep', df_all2['sweep'].iloc[0])
            ts = sett.get('t_start', float(df_all2['time_s'].min()))
            te = sett.get('t_end', float(df_all2['time_s'].max()))
            sub_f = df_all2[(df_all2['sweep'] == sw) & (df_all2['time_s'] >= ts) & (df_all2['time_s'] <= te)].copy()
            lp = sett.get('lp_hz', 0)
            if lp > 0 and fdata['meta']['sample_rate_hz'] > 0 and len(sub_f) > 20:
                sub_f['signal'] = butter_lowpass(sub_f['signal'].to_numpy(), lp, fdata['meta']['sample_rate_hz'])
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
            zf.writestr('review_state.json', json.dumps(S.records, indent=2))
        st.download_button('⬇ Download all exports (ZIP)', data=zbuf.getvalue(), file_name='syncapture_exports.zip', mime='application/zip', type='primary')
        exp1, exp2, exp3 = st.columns(3)
        with exp1:
            st.download_button('Prism: Amplitude CSV', prism_amp.to_csv(index=False).encode(), file_name='Prism_amplitude_pA.csv', mime='text/csv')
        with exp2:
            st.download_button('Prism: Frequency CSV', prism_freq.to_csv(index=False).encode(), file_name='Prism_frequency_Hz.csv', mime='text/csv')
        with exp3:
            st.download_button('Per-cell summary CSV', df_rec.to_csv(index=False).encode(), file_name='per_cell_summary.csv', mime='text/csv')

st.markdown("<hr>", unsafe_allow_html=True)
if not HAS_PYABF:
    st.warning('⚠ pyabf not installed. Install it to load ABF files: `pip install pyabf`')
st.markdown("<p style='font-size:0.75rem;color:#9ca3af;text-align:center'>SynCapture · Whole-cell patch-clamp synaptic event analysis · Run: <code>streamlit run syncapture.py</code></p>", unsafe_allow_html=True)
