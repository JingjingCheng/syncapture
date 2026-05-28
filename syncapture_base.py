#!/usr/bin/env python3
"""
SynCapture — Synaptic Event Analysis Tool
Run: streamlit run syncapture.py
Dependencies: pip install streamlit pyabf scipy pandas matplotlib numpy plotly
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
.main .block-container { padding: 0.5rem 0 1rem 0 !important; max-width: 100% !important; }
.block-container { padding-left: 0 !important; padding-right: 0 !important; }
section[data-testid="stSidebar"] { min-width: 310px !important; max-width: 340px !important; }
section[data-testid="stSidebar"] [data-testid="stVerticalBlockBorderWrapper"] { padding: 0; }
/* Make the Plotly chart fill full width and available height */
[data-testid="stPlotlyChart"] { width: 100% !important; min-height: calc(100vh - 260px); }
[data-testid="stPlotlyChart"] > div { width: 100% !important; height: 100% !important; }
[data-testid="stPlotlyChart"] iframe { width: 100% !important; height: 100% !important; }
div[data-testid="stMetric"] { background:var(--surface); border:1px solid var(--border); border-radius:8px; padding:0.5rem 0.7rem; }
.stButton > button { border-radius:6px; font-size:0.82rem; font-weight:500; }
.stButton > button[kind="primary"] { background:var(--accent); border:none; }
.stButton > button:hover { opacity:0.88; }
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

def make_trace_figure_plotly(sub, events_df, settings, file_name):
    """Interactive Plotly figure with drag-zoom, box-select, scroll-zoom and double-click reset."""
    direction = settings.get('direction', 'inward (EPSC)')
    marker_color = '#1a6b55' if 'EPSC' in direction else '#b91c1c'
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
                fig.add_trace(go.Scatter(
                    x=acc['time_s'], y=acc['amplitude_pA'] + baseline,
                    mode='markers', marker=dict(color=marker_color, size=7),
                    name=f'{len(acc)} events',
                    hovertemplate='Time: %{x:.4f}s<br>Amp: %{y:.2f}pA<extra></extra>',
                ))
            if not rej.empty:
                fig.add_trace(go.Scatter(
                    x=rej['time_s'], y=rej['amplitude_pA'] + baseline,
                    mode='markers', marker=dict(color='#9ca3af', size=5, symbol='x'),
                    name='rejected',
                    hovertemplate='Time: %{x:.4f}s<br>Amp: %{y:.2f}pA<extra></extra>',
                ))
    fig.update_layout(
        title=dict(text=file_name, font=dict(size=13, color='#1a1a1a')),
        xaxis=dict(
            title=dict(text='Time (s)', font=dict(size=11, color='#6b7280')),
            tickfont=dict(size=10, color='#6b7280'),
            showgrid=True, gridcolor='rgba(0,0,0,0.05)', zeroline=False,
        ),
        yaxis=dict(
            title=dict(text='Current (pA)', font=dict(size=11, color='#6b7280')),
            tickfont=dict(size=10, color='#6b7280'),
            showgrid=True, gridcolor='rgba(0,0,0,0.05)', zeroline=False,
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
    )
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
    <div style='display:flex;align-items:center;gap:10px;padding:2px 0 6px 0'>
        <svg width="30" height="30" viewBox="0 0 36 36" fill="none" xmlns="http://www.w3.org/2000/svg">
          <rect width="36" height="36" rx="8" fill="#1a6b55"/>
          <polyline points="4,18 10,18 14,8 18,28 22,14 26,18 32,18" stroke="white" stroke-width="2.2" stroke-linejoin="round" stroke-linecap="round" fill="none"/>
        </svg>
        <div>
            <div style='font-size:15px;font-weight:700;color:#111827;line-height:1.2'>SynCapture</div>
            <div style='font-size:10px;color:#9ca3af'>Synaptic Event Analysis</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    uploaded = st.file_uploader('Upload ABF files', type=['abf'], accept_multiple_files=True)
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
        st.divider()
        selectable = [n for n in S.file_order if n not in S.skipped]
        if selectable:
            default_ix = selectable.index(S.active) if S.active in selectable else 0
            S.active = st.selectbox('📂 File', selectable, index=default_ix, key='file_select')
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

        st.divider()
        st.markdown("**Sweep & Window**")
        sweep = st.selectbox('Sweep', sweeps_available, index=sweeps_available.index(default_sweep) if default_sweep in sweeps_available else 0, key=f'sweep_{S.active}')
        sweep_df = df_all[df_all['sweep'] == sweep].copy()
        t_min, t_max = float(sweep_df['time_s'].min()), float(sweep_df['time_s'].max())

        sc1, sc2 = st.columns(2)
        with sc1:
            t_start = st.number_input('Start (s)', min_value=0.0, max_value=t_max, value=float(prev.get('t_start', t_min)), step=0.1, key=f't_start_{S.active}')
        with sc2:
            t_end = st.number_input('End (s)', min_value=0.0, max_value=t_max, value=float(prev.get('t_end', t_max)), step=0.1, key=f't_end_{S.active}')
        lp_hz = st.number_input('Low-pass (Hz)', min_value=0.0, value=float(prev.get('lp_hz', 1000.0)), step=50.0, help='0 = off', key=f'lp_hz_{S.active}')

        st.divider()
        st.markdown("**Detection**")
        direction = st.selectbox('Direction', ['inward (EPSC)', 'outward (IPSC)'], index=['inward (EPSC)', 'outward (IPSC)'].index(prev.get('direction', 'inward (EPSC)')) if prev.get('direction', 'inward (EPSC)') in ['inward (EPSC)', 'outward (IPSC)'] else 0, key=f'direction_{S.active}')
        baseline_pct = st.slider('Baseline %', 5, 50, int(prev.get('baseline_pct', 20)), 5, key=f'bl_pct_{S.active}')

        pc1, pc2 = st.columns(2)
        with pc1:
            prominence = st.number_input('Prom. (pA)', min_value=0.5, value=float(prev.get('prominence', 8.0)), step=0.5, key=f'prom_{S.active}')
        with pc2:
            distance_ms = st.number_input('Min IEI (ms)', min_value=0.1, value=float(prev.get('distance_ms', 5.0)), step=0.5, key=f'dist_{S.active}')

        bc1, bc2 = st.columns(2)
        with bc1:
            _run_detect = st.button('⚡ Detect', type='primary', key=f'detect_{S.active}', use_container_width=True)
        with bc2:
            _clear_win = st.button('🗑 Clear', key=f'clear_win_{S.active}', use_container_width=True)

        S.settings[S.active] = {'direction': direction, 'baseline_pct': baseline_pct, 'prominence': prominence, 'distance_ms': distance_ms, 'sweep': sweep, 't_start': t_start, 't_end': t_end, 'lp_hz': lp_hz}

        st.divider()
        with st.expander('📋 Cell Labels', expanded=False):
            saved_rec = next((r for r in S.records if r.get('file_name') == S.active), {})
            cell_id = st.text_input('Cell ID', value=saved_rec.get('cell_id', Path(S.active).stem), key=f'cell_id_{S.active}')
            individual = st.text_input('Individual', value=saved_rec.get('individual', ''), key=f'individual_{S.active}')
            lc1, lc2 = st.columns(2)
            with lc1:
                group_options = ['naive', 'ovx', 'control', 'treatment', 'other']
                saved_group = saved_rec.get('group', 'naive')
                group = st.selectbox('Group', group_options, index=group_options.index(saved_group) if saved_group in group_options else 0, key=f'group_{S.active}')
            with lc2:
                status_options = ['accepted', 'needs_check', 'rejected']
                saved_status = saved_rec.get('status', 'accepted')
                status = st.selectbox('Status', status_options, index=status_options.index(saved_status) if saved_status in status_options else 0, key=f'status_{S.active}')
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
        sub['signal'] = butter_lowpass(sub['signal'].to_numpy(), lp_hz, meta['sample_rate_hz'])

    current_events = S.events.get(S.active, pd.DataFrame(columns=['time_s','amplitude_pA','prominence','iei_s','accepted']))

    # ---- handle sidebar actions ----
    if _run_detect and not sub.empty:
        new_ev = detect_synaptic_events(sub['signal'].to_numpy(), sub['time_s'].to_numpy(), direction, prominence, distance_ms, baseline_pct)
        outside = current_events[(current_events['time_s'] < t_start) | (current_events['time_s'] > t_end)] if not current_events.empty else pd.DataFrame(columns=new_ev.columns)
        S.events[S.active] = pd.concat([outside, new_ev], ignore_index=True).sort_values('time_s').reset_index(drop=True)
        st.rerun()
    if _clear_win:
        if not current_events.empty:
            S.events[S.active] = current_events[(current_events['time_s'] < t_start) | (current_events['time_s'] > t_end)].reset_index(drop=True)
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
        S.records.append(rec)
        st.toast(f'✓ Saved {cell_id} → dataset ({len(S.records)} files)')

    # ---- interactive trace chart ----
    win_events = current_events[(current_events['time_s'] >= t_start) & (current_events['time_s'] <= t_end)] if not current_events.empty else current_events
    fig_plotly = make_trace_figure_plotly(sub, win_events if not win_events.empty else None, S.settings[S.active], S.active)
    st.plotly_chart(fig_plotly, use_container_width=True, config={'scrollZoom': True, 'displayModeBar': True, 'modeBarButtonsToAdd': ['drawrect', 'eraseshape']})

    # ---- metrics row ----
    full_ev = S.events.get(S.active, pd.DataFrame())
    if not full_ev.empty:
        dur = max(0.001, t_end - t_start)
        sm = summary_from_events(full_ev[(full_ev['time_s'] >= t_start) & (full_ev['time_s'] <= t_end)], dur)
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
        st.caption(f'{len(win_ev)} events in window · {total_acc} accepted total')
        edited = st.data_editor(
            win_ev, num_rows='dynamic', use_container_width=True, key=f'ev_table_{S.active}',
            column_config={
                'accepted': st.column_config.CheckboxColumn('Accept'),
                'time_s': st.column_config.NumberColumn('Time (s)', format='%.4f'),
                'amplitude_pA': st.column_config.NumberColumn('Amplitude (pA)', format='%.2f'),
                'prominence': st.column_config.NumberColumn('Prominence', format='%.2f'),
                'iei_s': st.column_config.NumberColumn('IEI (s)', format='%.4f'),
            },
            height=260,
        )
        outside = all_ev[(all_ev['time_s'] < t_start) | (all_ev['time_s'] > t_end)]
        S.events[S.active] = pd.concat([outside, edited], ignore_index=True).sort_values('time_s').reset_index(drop=True)

# ───────────────────────────────────────────
#  EXPORT SECTION
# ───────────────────────────────────────────
if S.records:
    st.divider()
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
                    sub_f['signal'] = butter_lowpass(sub_f['signal'].to_numpy(), lp, fdata_exp['meta']['sample_rate_hz'])
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

