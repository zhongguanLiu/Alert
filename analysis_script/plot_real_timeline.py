#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Plot the real_run_001 timeline figure."""

import sys
import json
import math
import os
import pathlib
import numpy as np
import matplotlib
INTERACTIVE = '--show' in sys.argv
SAVE = '--save' in sys.argv or not INTERACTIVE
if not INTERACTIVE:
    matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

_SCRIPT_DIR = pathlib.Path(__file__).resolve().parent
_PARENT_DIR = _SCRIPT_DIR.parent
if str(_PARENT_DIR) not in sys.path:
    sys.path.insert(0, str(_PARENT_DIR))

import analysis_script.common as common

# ── Paths ──────────────────────────────────────────────────────────────────────
DATA_DIR = str(common.default_real_run_dir("real_run_001"))
OUT_DIR = str(common.real_figure_dir())
OUT_FILE, OUT_FILE_PNG = [str(path) for path in common.real_timeline_output_paths("real_run_001")]

# ── Helpers ────────────────────────────────────────────────────────────────────

def get_stamp(entry):
    h = entry.get('header', {})
    s = h.get('stamp', {})
    if isinstance(s, dict):
        return s.get('sec', s.get('secs', 0.0))
    return float(s)


def ra_stamp(e):
    ra = e.get('recorded_at', {})
    if isinstance(ra, dict) and ra.get('sec', 0) > 1e9:
        return ra['sec']
    return get_stamp(e)


# ── Data loading ───────────────────────────────────────────────────────────────

def load_clusters(t0):
    raw = []
    with open(os.path.join(DATA_DIR, 'clusters.jsonl')) as f:
        for line in f:
            raw.append(json.loads(line))
    raw.sort(key=get_stamp)

    frame_t, frame_n, frame_max, frame_mean = [], [], [], []
    sc_t, sc_d, sc_sup = [], [], []

    for frame in raw:
        t = get_stamp(frame) - t0
        cls = frame.get('clusters', [])
        disps = [math.sqrt(sum(x**2 for x in c['disp_mean'])) * 1000 for c in cls]
        sups  = [c.get('support_count', 1) for c in cls]
        frame_t.append(t)
        frame_n.append(len(cls))
        frame_max.append(max(disps) if disps else 0.0)
        frame_mean.append(sum(d * s for d, s in zip(disps, sups)) / sum(sups)
                          if sups else 0.0)
        for c, d in zip(cls, disps):
            sc_t.append(t)
            sc_d.append(d)
            sc_sup.append(c.get('support_count', 1))

    return frame_t, frame_n, frame_max, frame_mean, sc_t, sc_d, sc_sup


def load_reacquired(t0_ra):
    raw = []
    with open(os.path.join(DATA_DIR, 'anchor_states.jsonl')) as f:
        for line in f:
            raw.append(json.loads(line))
    raw.sort(key=ra_stamp)

    ra_t, ra_max, ra_count = [], [], []
    for frame in raw:
        t = ra_stamp(frame) - t0_ra
        sig_ra = [a for a in frame.get('anchors', [])
                  if a.get('reacquired') and a.get('significant')]
        if sig_ra:
            ra_t.append(t)
            ra_max.append(max(a['disp_norm'] * 1000 for a in sig_ra))
            ra_count.append(len(sig_ra))

    return ra_t, ra_max, ra_count


def build_ra_series(ra_t, ra_max, t_max=55.5, dt=0.5):
    """
    Project (ra_t, ra_max) onto a regular grid. Cells with no nearby measurement
    stay NaN so fill_between does not bridge quiet intervals (gap > 2 bins).
    """
    t_grid = np.arange(0.0, t_max + dt, dt)
    grid = np.full(len(t_grid), np.nan)
    for t_pt, d_pt in zip(ra_t, ra_max):
        idx = int(round(t_pt / dt))
        if 0 <= idx < len(grid):
            grid[idx] = d_pt if np.isnan(grid[idx]) else max(grid[idx], d_pt)
    # extend each filled cell ±1 bin to close the bar visually
    filled = np.where(~np.isnan(grid))[0]
    result = grid.copy()
    for i in filled:
        for j in [i - 1, i + 1]:
            if 0 <= j < len(result) and np.isnan(result[j]):
                result[j] = 0.0
    return t_grid, result


def load_regions(t0):
    raw = []
    with open(os.path.join(DATA_DIR, 'persistent_risk_regions.jsonl')) as f:
        for line in f:
            raw.append(json.loads(line))
    raw.sort(key=get_stamp)
    return ([get_stamp(r) - t0 for r in raw],
            [len(r.get('regions', [])) for r in raw])


def load_structure_motions(t0):
    raw = []
    with open(os.path.join(DATA_DIR, 'structure_motions.jsonl')) as f:
        for line in f:
            raw.append(json.loads(line))
    sm = []
    for m in raw:
        if m.get('motions'):
            t_m = get_stamp(m) - t0
            for mot in m['motions']:
                mv = mot['motion']
                mag = math.sqrt(mv['x']**2 + mv['y']**2 + mv['z']**2) * 1000
                sm.append({'t': t_m, 'mag': mag, 'conf': mot['confidence']})
    return sm


# ── Phase definitions (real_run_001) ──────────────────────────────────────────

PHASES = [
    # (t_start, t_end, facecolor, label, alpha)
    (0.0,  3.0,  '#d0d0d0', 'Init',        0.45),
    (3.0,  27.5, '#fff8e1', 'Pre-failure', 0.55),
    (27.5, 48.0, '#ffcdd2', 'Cascade',     0.45),
    (48.0, 51.5, '#f3e5f5', 'Quiet',       0.40),
    (51.5, 55.5, '#e3f2fd', '2nd',         0.55),
]

C = {
    'max_fill':  '#ef9a9a',
    'max_line':  '#c62828',
    'scatter':   '#e53935',
    'n_cl':      '#1565c0',
    'reg':       '#e65100',
    'ra_line':   '#7b1fa2',
    'ra_fill':   '#ce93d8',
    'sm':        '#00695c',
    'onset_v':   '#c62828',
    'w2_v':      '#0277bd',
}


def shade_phases(ax, t_max=55.5):
    for s, e, c, label, a in PHASES:
        ax.axvspan(s, min(e, t_max), color=c, alpha=a, zorder=0, lw=0)


def draw_event_vlines(ax, sm, onset=True, w2=True, sm_line=True):
    if onset:
        ax.axvline(27.5, color=C['onset_v'], lw=1.0, ls=':', zorder=6)
    if w2:
        ax.axvline(51.5, color=C['w2_v'],  lw=1.0, ls=':', zorder=6)
    if sm_line:
        for ev in sm:
            ax.axvline(ev['t'], color=C['sm'], lw=0.9, ls='--', alpha=0.65, zorder=5)


def panel_label(ax, letter, x=0.012, y=0.93):
    ax.text(x, y, f'({letter})', transform=ax.transAxes,
            fontsize=9, fontweight='bold', va='top')


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    # Determine t0 from first cluster header stamp; t0_ra from recorded_at wall clock
    raw0 = []
    with open(os.path.join(DATA_DIR, 'clusters.jsonl')) as f:
        for line in f:
            raw0.append(json.loads(line))
    raw0.sort(key=get_stamp)
    t0 = get_stamp(raw0[0])

    raw0_by_wall = sorted(raw0, key=ra_stamp)
    t0_ra = raw0_by_wall[0].get('recorded_at', {}).get('sec', t0)

    # Load data
    (frame_t, frame_n, frame_max, frame_mean,
     sc_t, sc_d, sc_sup) = load_clusters(t0)
    reg_t, reg_n = load_regions(t0)
    sm = load_structure_motions(t0)
    ra_t, ra_max, ra_count = load_reacquired(t0_ra)

    t_max = 55.5

    # Build REACQ series — show all events (including early pre-failure signal)
    ra_tg, ra_mg = build_ra_series(ra_t, ra_max, t_max=t_max)

    # ── Figure setup ───────────────────────────────────────────────────────────
    plt.rcParams.update({
        'font.family':      'sans-serif',
        'font.size':        8,
        'axes.linewidth':   0.7,
        'xtick.major.width': 0.6,
        'ytick.major.width': 0.6,
        'xtick.minor.width': 0.4,
        'xtick.major.size':  3.5,
        'xtick.minor.size':  2.0,
        'ytick.major.size':  3.5,
        'lines.linewidth':   1.2,
        'legend.framealpha': 0.85,
        'legend.fontsize':   7.5,
        'legend.borderpad':  0.4,
        'legend.handlelength': 1.6,
    })

    fig, axes = plt.subplots(
        2, 1, figsize=(7.16, 4.4),
        gridspec_kw={'height_ratios': [1.8, 1.2]},
        sharex=True
    )
    fig.subplots_adjust(hspace=0.06, left=0.09, right=0.76,
                        top=0.97, bottom=0.10)

    # ══ Panel (a): Displacement ════════════════════════════════════════════════
    ax1 = axes[0]
    shade_phases(ax1)

    # Cluster layer: filled max + weighted-mean
    ax1.fill_between(frame_t, frame_max, step='mid',
                     color=C['max_fill'], alpha=0.45, zorder=2, lw=0)
    ax1.step(frame_t, frame_mean, where='mid',
             color='#ef9a9a', lw=0.9, alpha=0.7, zorder=3)
    ax1.step(frame_t, frame_max, where='mid',
             color=C['max_line'], lw=1.8, zorder=4, label='Max cluster disp.')

    # REACQ filled band (all events, including early pre-failure signal)
    ax1.fill_between(ra_tg, ra_mg, step='mid',
                     color=C['ra_fill'], alpha=0.40, zorder=2, lw=0)
    ax1.step(ra_tg, ra_mg, where='mid',
             color=C['ra_line'], lw=1.5, alpha=0.92, zorder=4)

    draw_event_vlines(ax1, sm)

    ax1.set_ylabel('Displacement (mm)', fontsize=8)
    ax1.set_ylim(-4, 150)
    ax1.yaxis.set_major_locator(mticker.MultipleLocator(25))
    ax1.yaxis.set_minor_locator(mticker.MultipleLocator(12.5))

    # Phase labels removed — background color alone indicates each phase

    # Manual rotated legend: two vertical columns in the reserved right-side area
    ax1_bbox = ax1.get_position()
    legend_y = ax1_bbox.y0 + ax1_bbox.height * 0.54
    legend_region_width = 1.0 - ax1_bbox.x1
    col_x1 = ax1_bbox.x1 + legend_region_width * 0.22
    col_x2 = ax1_bbox.x1 + legend_region_width * 0.52

    for x_c, fc, tc, label in [
        (col_x1, C['max_fill'], C['max_line'], 'MAX clusters'),
        (col_x2, C['ra_fill'],  C['ra_line'],  'MAX reacquired'),
    ]:
        fig.text(x_c, legend_y, label,
                 fontsize=6.2, rotation=90,
                 ha='center', va='center',
                 color=tc, transform=fig.transFigure, clip_on=False,
                 bbox=dict(boxstyle='round,pad=0.28', fc=fc, alpha=0.78,
                           ec='gray', lw=0.5))
    panel_label(ax1, 'a')

    # ══ Panel (b): Active cluster count + confirmed regions ════════════════════
    ax2 = axes[1]
    shade_phases(ax2)
    draw_event_vlines(ax2, sm)

    ax2.bar(frame_t, frame_n, width=0.45,
            color=C['n_cl'], alpha=0.70, zorder=3, label='Active clusters')
    ax2.set_ylabel('Active clusters', color=C['n_cl'], fontsize=8.5)
    ax2.tick_params(axis='y', colors=C['n_cl'], labelsize=7.5)
    ax2.set_ylim(0, 20)
    ax2.yaxis.set_major_locator(mticker.MultipleLocator(5))

    ax2r = ax2.twinx()
    reg_arr     = np.array(reg_n)
    reg_t_arr   = np.array(reg_t)
    baseline_val = float(np.mean(reg_arr[reg_t_arr < 3.0])) if any(reg_t_arr < 3.0) else 5.0
    reg_net = np.maximum(reg_arr - baseline_val, 0.0)
    ax2r.plot(reg_t, reg_net, color=C['reg'], lw=1.6, alpha=0.90,
              zorder=4, label='Confirmed regions')
    ax2r.set_ylabel('Confirmed regions', color=C['reg'], fontsize=7.5)
    ax2r.tick_params(axis='y', colors=C['reg'], labelsize=7)
    ax2r.set_ylim(0, 32)
    ax2r.yaxis.set_major_locator(mticker.MultipleLocator(10))

    panel_label(ax2, 'b')

    # ── Common x-axis ──────────────────────────────────────────────────────────
    for ax in [ax1, ax2]:
        ax.tick_params(axis='x', labelsize=7.5)
        ax.xaxis.set_major_locator(mticker.MultipleLocator(5))
        ax.xaxis.set_minor_locator(mticker.MultipleLocator(1))
        ax.grid(axis='x', which='major', lw=0.35, alpha=0.4, ls=':',  zorder=1)
        ax.grid(axis='y', which='major', lw=0.25, alpha=0.30, ls=':', zorder=1)
        ax.set_xlim(0, t_max)
    ax2.set_xlabel('Time (s)', fontsize=8)

    # ── Interactive / save ─────────────────────────────────────────────────────
    if INTERACTIVE:
        print("Interactive mode: close window to save.")
        plt.show()
        if SAVE:
            fig.savefig(OUT_FILE, bbox_inches='tight', dpi=200)
            fig.savefig(OUT_FILE_PNG, bbox_inches='tight', dpi=300)
            print(f"Saved: {OUT_FILE}")
            print(f"Saved: {OUT_FILE_PNG}")
    else:
        fig.savefig(OUT_FILE, bbox_inches='tight', dpi=200)
        fig.savefig(OUT_FILE_PNG, bbox_inches='tight', dpi=300)
        print(f"Saved: {OUT_FILE}")
        print(f"Saved: {OUT_FILE_PNG}")


if __name__ == '__main__':
    main()
