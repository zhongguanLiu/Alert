#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Generate the system overview figure."""
import os
import pathlib
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle, Circle, Polygon
from matplotlib.lines import Line2D

_SCRIPT_DIR = pathlib.Path(__file__).resolve().parent
_PARENT_DIR = _SCRIPT_DIR.parent
if str(_PARENT_DIR) not in sys.path:
    sys.path.insert(0, str(_PARENT_DIR))

import analysis_script.common as common

# ── Palette ────────────────────────────────────────────────────────────────────
C = dict(
    navy    = '#1b3a5c',
    blue_m  = '#2e6da4',
    blue_l  = '#dce9f5',
    red_d   = '#c62828',
    red_l   = '#ffcdd2',
    purple  = '#7b1fa2',
    purp_l  = '#ede0f5',
    orange  = '#e65100',
    orang_l = '#fff3e0',
    green_d = '#2e7d32',
    green_l = '#e8f5e9',
    teal_d  = '#00695c',
    teal_l  = '#e0f2f1',
    amber   = '#f59f00',
    amber_l = '#fff9e0',
    gray_l  = '#f0f2f5',
    gray_m  = '#9e9e9e',
    bg      = '#f4f6f9',
    white   = '#ffffff',
)

# ── Canvas ─────────────────────────────────────────────────────────────────────
FW, FH = 17, 9.5
fig = plt.figure(figsize=(FW, FH), dpi=200)
ax  = fig.add_axes([0, 0, 1, 1])
ax.set_xlim(0, FW); ax.set_ylim(0, FH); ax.axis('off')
fig.patch.set_facecolor(C['bg'])

# ── Low-level helpers ──────────────────────────────────────────────────────────
def rbox(x, y, w, h, fc, ec, lw=1.5, r=0.18, alpha=1.0, zorder=3):
    p = FancyBboxPatch((x, y), w, h,
                       boxstyle=f'round,pad=0,rounding_size={r}',
                       fc=fc, ec=ec, lw=lw, alpha=alpha, zorder=zorder)
    ax.add_patch(p)
    return p

def module_box(x, y, w, h, title, hdr_c, body_c, fontsize=8.5):
    """Dark header + light body module box."""
    rbox(x, y, w, h, fc=body_c, ec=hdr_c, lw=2.0, r=0.20)
    rbox(x, y+h-0.56, w, 0.56, fc=hdr_c, ec=hdr_c, lw=0, r=0.20)
    rbox(x, y+h-0.56, w, 0.30, fc=body_c, ec='none', lw=0, r=0)   # cover seam
    ax.text(x+w/2, y+h-0.28, title, ha='center', va='center',
            fontsize=fontsize, fontweight='bold', color='white', zorder=6)

def arr(x0, y0, x1, y1, col=C['navy'], lw=2.0, hw=0.22, hl=0.26, style='->'):
    ax.annotate('', xy=(x1, y1), xytext=(x0, y0),
                arrowprops=dict(
                    arrowstyle=f'{style},head_width={hw},head_length={hl}',
                    color=col, lw=lw, connectionstyle='arc3,rad=0.0'),
                zorder=7)

def txt(x, y, s, fs=7.5, col=C['navy'], ha='center', va='center', bold=False,
        italic=False, zorder=8, **kw):
    fw = 'bold' if bold else 'normal'
    fs2 = 'italic' if italic else 'normal'
    ax.text(x, y, s, ha=ha, va=va, fontsize=fs, color=col,
            fontweight=fw, fontstyle=fs2, zorder=zorder, **kw)

def hrule(x0, x1, y, col=C['gray_m'], lw=0.8, ls='--'):
    ax.plot([x0, x1], [y, y], color=col, lw=lw, ls=ls, zorder=4)


# ══════════════════════════════════════════════════════════════════════════════
# BACKGROUND STRIPS
# ══════════════════════════════════════════════════════════════════════════════
# Bottom: input strip
rbox(0.12, 0.12, FW-0.24, 2.10, fc='#e3eaf3', ec='#a0b4c8', lw=1.0, r=0.25)
txt(0.55, 1.17, 'SENSING &\nLOCALIZATION', fs=7.5, col=C['navy'], bold=True,
    rotation=90)

# Top: title band
rbox(0.12, FH-1.02, FW-0.24, 0.90, fc=C['navy'], ec=C['navy'], lw=0, r=0.20)
txt(FW/2, FH-0.57,
    'ALERT — Adaptive LiDAR Early-Warning for Structural Response Tracking',
    fs=11, col='white', bold=True)

# ══════════════════════════════════════════════════════════════════════════════
# INPUT / SENSING ZONE (x=0.8..3.3, y strip)
# ══════════════════════════════════════════════════════════════════════════════
# Robot silhouette (stylised rectangles)
rx, ry = 0.85, 0.32
# Chassis
rbox(rx, ry+0.20, 1.00, 0.52, fc='#546e7a', ec='#37474f', lw=1.2, r=0.08)
# LiDAR dome on top
rbox(rx+0.28, ry+0.72, 0.42, 0.20, fc='#f59f00', ec='#e65100', lw=1.0, r=0.08)
# Head
rbox(rx+0.65, ry+0.30, 0.45, 0.38, fc='#546e7a', ec='#37474f', lw=1.0, r=0.07)
# 4 legs
for lxo in [0.10, 0.28, 0.60, 0.82]:
    ax.plot([rx+lxo, rx+lxo-0.04], [ry+0.20, ry+0.02],
            color='#37474f', lw=2.2, solid_capstyle='round', zorder=4)
# LiDAR rays
for ang, rc in [(-50, C['blue_m']), (-20, C['teal_d']),
                (20,  C['red_d']),  (55,  C['orange'])]:
    ex = 0.52*np.cos(np.radians(ang)); ey = 0.52*np.sin(np.radians(ang))
    ax.annotate('', xy=(rx+0.49+ex, ry+0.82+ey), xytext=(rx+0.49, ry+0.82),
                arrowprops=dict(
                    arrowstyle='->,head_width=0.06,head_length=0.09',
                    color=rc, lw=0.9),
                zorder=5)
txt(rx+0.65, ry-0.12, 'LiDAR-Inertial\nOdometry (FAST-LIO2)',
    fs=6.8, italic=True, col=C['navy'])

# Mini point cloud
np.random.seed(7)
px = rx+1.35; py = ry
pts = np.column_stack([px + np.random.randn(100)*0.38,
                       py + 0.25 + np.random.rand(100)*1.35])
cvals = (pts[:,1]-py) / 1.6
ax.scatter(pts[:,0], pts[:,1], c=plt.cm.plasma(cvals), s=3.5,
           alpha=0.75, zorder=4)
txt(px+0.38, py+0.05, r'$\mathbf{P}_t$', fs=8, col=C['navy'])
txt(px+0.38, py-0.16, r'$\mathbf{T}_{WB},\;\mathbf{\Sigma}_\xi$', fs=8, col=C['navy'])

# Arrow: input → Module A
arr(px+0.76, 1.17, 3.45, 1.17, col=C['navy'])

# ══════════════════════════════════════════════════════════════════════════════
# MODULE A — OBSERVATION MODEL
# ══════════════════════════════════════════════════════════════════════════════
AX, AY, AW, AH = 3.45, 2.30, 3.50, 5.80
module_box(AX, AY, AW, AH, 'A.  UNCERTAINTY-AWARE OBSERVATION', C['navy'], C['blue_l'])

# ── Reference init sub-box (one-time) ──────────────────────────────────────
rbox(AX+0.18, AY+0.10, AW-0.36, 0.92,
     fc=C['amber_l'], ec=C['amber'], lw=1.3, r=0.12)
txt(AX+AW/2, AY+0.70, 'Reference Initialization  (one-time)',
    fs=7.0, col=C['amber'], bold=True)
txt(AX+AW/2, AY+0.38, r'$N_\mathrm{init}$ frames $\rightarrow$ frozen anchors $\{A_i\}$',
    fs=7.5, col='#5d4037')
arr(AX+AW/2, AY+1.02, AX+AW/2, AY+1.40, col=C['amber'], lw=1.5, hw=0.14, hl=0.18)

# ── Anchor type icons ──────────────────────────────────────────────────────
hrule(AX+0.18, AX+AW-0.18, AY+1.42)
txt(AX+AW/2, AY+1.68, 'Anchor Type Classification',
    fs=7.5, col=C['navy'], bold=True)
icon_y = AY+1.82
for k, (label_t, col_t, sym) in enumerate([
        ('PLANE\n1-DOF', C['blue_m'],  '▬'),
        ('EDGE\n2-DOF',  C['teal_d'],  '◤'),
        ('BAND\n2-DOF',  C['purple'],  '⬡'),
]):
    bx = AX+0.20 + k*1.12
    rbox(bx, icon_y, 0.95, 0.92, fc=C['blue_l'], ec=col_t, lw=0.9, r=0.10)
    txt(bx+0.475, icon_y+0.67, sym, fs=14, col=col_t)
    txt(bx+0.475, icon_y+0.26, label_t, fs=5.8, col=col_t, bold=True)

# ── Pose uncertainty ───────────────────────────────────────────────────────
hrule(AX+0.18, AX+AW-0.18, AY+2.86)
txt(AX+AW/2, AY+3.12, 'Pose Uncertainty Propagation',
    fs=7.5, col=C['navy'], bold=True)
txt(AX+AW/2, AY+3.52,
    r'$\mathbf{\Sigma}_\mathbf{x}=\mathbf{J}_p\,\tilde{\mathbf{\Sigma}}_\xi\,\mathbf{J}_p^\top+\sigma_p^2\mathbf{I}$',
    fs=8.5, col=C['blue_m'])

# ── Scalar measurement ─────────────────────────────────────────────────────
hrule(AX+0.18, AX+AW-0.18, AY+3.92)
txt(AX+AW/2, AY+4.18, 'Scalar Measurement Construction',
    fs=7.5, col=C['navy'], bold=True)
txt(AX+AW/2, AY+4.58,
    r'$z_n = \mathbf{n}^\top(\bar{\mathbf{p}}_\mathrm{curr}-\mathbf{c}_i^\mathrm{ref})$',
    fs=8.5, col=C['blue_m'])
txt(AX+AW/2, AY+4.95, r'$r_n = \mathbf{n}^\top(\mathbf{\Sigma}_\mathrm{cen}+\mathbf{\Sigma}_i^\mathrm{ref})\mathbf{n}+\sigma_n^2$',
    fs=7.8, col=C['blue_m'])

# ── A → B arrow ─────────────────────────────────────────────────────────────
arr(AX+AW, AY+3.5, 7.2, AY+3.5, col=C['navy'])
txt((AX+AW+7.2)/2, AY+3.82, r'$(z_k,\;r_k)$', fs=8, col=C['navy'])

# ══════════════════════════════════════════════════════════════════════════════
# MODULE B — IMM-IF ESTIMATOR
# ══════════════════════════════════════════════════════════════════════════════
BX, BY, BW, BH = 7.20, 2.30, 3.30, 5.80
module_box(BX, BY, BW, BH, 'B.  IMM-IF ESTIMATOR', C['navy'], C['teal_l'])

# ── Dual model boxes ────────────────────────────────────────────────────────
my = BY+BH-2.30
for k, (fc, ec, mname, mdesc) in enumerate([
        (C['teal_l'],  C['teal_d'],  r'$M_0$  Quiescent',  r'$\rho\,\mathbf{v}$ damping'),
        (C['blue_l'],  C['blue_m'],  r'$M_1$  Active',     'free motion'),
]):
    bx = BX+0.18 + k*1.66
    rbox(bx, my, 1.44, 1.22, fc=fc, ec=ec, lw=1.4, r=0.12)
    txt(bx+0.72, my+0.95, mname, fs=7.5, col=ec, bold=True)
    txt(bx+0.72, my+0.62, r'$\mathbf{F}_0$' if k==0 else r'$\mathbf{F}_1$',
        fs=11, col=ec)
    txt(bx+0.72, my+0.24, mdesc, fs=6.5, col=ec)

# Mixing double arrow
ax.annotate('', xy=(BX+1.62, my+0.61), xytext=(BX+1.28, my+0.61),
            arrowprops=dict(
                arrowstyle='<->,head_width=0.13,head_length=0.12',
                color=C['navy'], lw=1.4),
            zorder=6)
txt(BX+1.45, my+0.82, 'mix', fs=6.5, col=C['navy'], italic=True)

# ── Type constraint ─────────────────────────────────────────────────────────
hrule(BX+0.18, BX+BW-0.18, BY+BH-2.44)
arr(BX+BW/2, BY+BH-2.44, BX+BW/2, BY+BH-2.90, col=C['navy'], lw=1.5, hw=0.14, hl=0.16)
rbox(BX+0.18, BY+BH-3.85, BW-0.36, 0.88,
     fc=C['orang_l'], ec=C['orange'], lw=1.3, r=0.12)
txt(BX+BW/2, BY+BH-3.30, r'Type Constraint  $\lambda_s\mathbf{P}_\perp$',
    fs=7.5, col=C['orange'], bold=True)
txt(BX+BW/2, BY+BH-3.65,
    r'$\mathbf{\Lambda}^+_j=\mathbf{\Lambda}^-_j+\mathbf{\Lambda}_\mathrm{obs}+\lambda_s\mathbf{P}_\perp$',
    fs=7.8, col='#5d4037')

# ── Posterior out ────────────────────────────────────────────────────────────
hrule(BX+0.18, BX+BW-0.18, BY+BH-4.0)
arr(BX+BW/2, BY+BH-4.0, BX+BW/2, BY+BH-4.38, col=C['navy'], lw=1.5, hw=0.14, hl=0.16)
txt(BX+BW/2, BY+BH-4.60, 'Posterior Displacement',
    fs=7.5, col=C['navy'], bold=True)
txt(BX+BW/2, BY+BH-4.92,
    r'$(\mathbf{u}_i,\;\mathbf{\Sigma}_{u_i})$  per anchor',
    fs=8.5, col=C['teal_d'])

# ── Reacquisition pathway (special bottom sub-box) ──────────────────────────
hrule(BX+0.18, BX+BW-0.18, BY+BH-5.28)
rbox(BX+0.18, BY+0.10, BW-0.36, 1.00,
     fc=C['purp_l'], ec=C['purple'], lw=1.4, r=0.12)
txt(BX+BW/2, BY+0.77, 'Reacquisition Mode',
    fs=7.5, col=C['purple'], bold=True)
txt(BX+BW/2, BY+0.40,
    r'$d_\mathrm{reacq}=\|\bar{\mathbf{p}}_\mathrm{curr}-\mathbf{c}_i^\mathrm{ref}\|$',
    fs=7.8, col=C['purple'])
arr(BX+BW/2, BY+1.10, BX+BW/2, BY+1.48, col=C['purple'], lw=1.5, hw=0.13, hl=0.16)
txt(BX+BW/2, BY+1.70, 'direct full displacement',
    fs=6.5, col=C['purple'], italic=True)

# ── B → C arrow ─────────────────────────────────────────────────────────────
arr(BX+BW, BY+BH-4.6, 10.7, BY+BH-4.6, col=C['navy'])
txt((BX+BW+10.7)/2, BY+BH-4.32, r'$(\mathbf{u}_i,\;\mathbf{\Sigma}_{u_i})$',
    fs=8, col=C['navy'])

# ══════════════════════════════════════════════════════════════════════════════
# MODULE C — MULTI-SCALE EVIDENCE FUSION
# ══════════════════════════════════════════════════════════════════════════════
CX, CY, CW, CH = 10.70, 2.30, 3.40, 5.80
module_box(CX, CY, CW, CH, 'C.  EVIDENCE FUSION', C['navy'], C['red_l'])

# ── BC-CUSUM mini chart ──────────────────────────────────────────────────────
cpx, cpy, cpw, cph = CX+0.22, CY+CH-2.62, CW-0.44, 1.72
rbox(cpx, cpy, cpw, cph, fc='white', ec=C['gray_m'], lw=0.8, r=0.07)

t = np.linspace(0, 1, 100)
c_val = np.zeros(100)
for k in range(1, 100):
    s = -0.5 if k < 38 else +1.0
    c_val[k] = min(9, max(0, 0.94*c_val[k-1] + s))
c_scaled = cpy + 0.12 + (c_val/9.0)*(cph-0.26)
t_scaled = cpx + 0.06 + t*(cpw-0.12)
ax.plot(t_scaled, c_scaled, color=C['red_d'], lw=1.5, zorder=5)

# Threshold line h
hy = cpy + 0.12 + (6.5/9.0)*(cph-0.26)
ax.plot([cpx+0.06, cpx+cpw-0.06], [hy, hy],
        color=C['orange'], lw=1.1, ls='--', zorder=5)
txt(cpx+cpw-0.10, hy+0.09, '$h$', fs=7.5, col=C['orange'])

# Alarm shaded region
alarm_start = cpx + 0.06 + 0.38*(cpw-0.12)
alarm_rect = Rectangle((alarm_start, cpy+0.02), cpx+cpw-0.06-alarm_start, cph-0.04,
                        fc=C['red_d'], alpha=0.10, zorder=4)
ax.add_patch(alarm_rect)
txt(alarm_start + 0.18, cpy+0.28, '▶ ALARM', fs=6.5, col=C['red_d'], bold=True)

txt(cpx+cpw/2, cpy+cph-0.14, 'BC-CUSUM  (temporal accumulation)',
    fs=7.0, col=C['red_d'], bold=True)
txt(cpx+0.22, cpy+0.14, r'$C_{i,t}=\min(C_\mathrm{max},\,\max(0,\lambda_c C_{i,t-1}+s_{i,t}))$',
    fs=6.8, col=C['navy'], ha='left')

# ── Directional coherence ────────────────────────────────────────────────────
hrule(CX+0.18, CX+CW-0.18, CY+CH-2.74)
txt(CX+CW/2, CY+CH-3.00, 'Directional Consistency',
    fs=7.5, col=C['navy'], bold=True)
txt(CX+CW/2, CY+CH-3.36,
    r'$\mathbf{S}_{i,t}=\lambda^{\Delta t}\mathbf{S}_{i,t-1}+w_t\,\mathbf{u}_{i,t}$',
    fs=7.8, col=C['red_d'])
txt(CX+CW/2, CY+CH-3.70, r'coherence ratio: $\|\mathbf{S}\|/Q \to 0$ for noise',
    fs=6.8, col=C['navy'], italic=True)

# ── Spatial graph ────────────────────────────────────────────────────────────
hrule(CX+0.18, CX+CW-0.18, CY+CH-3.88)
txt(CX+CW/2, CY+CH-4.14, 'Spatial Graph Coherence',
    fs=7.5, col=C['navy'], bold=True)

# Mini adjacency graph
gnx = CX + 0.30; gny = CY+CH-5.36
nodes = [(gnx+0.32,gny+0.55), (gnx+0.90,gny+0.70), (gnx+0.62,gny+0.22),
         (gnx+1.42,gny+0.48), (gnx+1.82,gny+0.65), (gnx+2.18,gny+0.32)]
ncols = [C['red_d']]*3 + [C['gray_m']]*3
edges = [(0,1),(0,2),(1,2),(1,3),(3,4),(3,5)]
for i,j in edges:
    ec2 = C['red_d'] if (ncols[i]==C['red_d'] and ncols[j]==C['red_d']) else C['gray_m']
    ax.plot([nodes[i][0],nodes[j][0]], [nodes[i][1],nodes[j][1]],
            color=ec2, lw=1.6, alpha=0.8, zorder=4)
for (nx2,ny2), nc in zip(nodes, ncols):
    ax.add_patch(Circle((nx2,ny2), 0.105, fc=nc, ec='white', lw=0.8, zorder=5))
txt(gnx+1.30, gny-0.08, 'anchor adjacency graph',
    fs=6.5, col=C['navy'], italic=True)

hrule(CX+0.18, CX+CW-0.18, CY+CH-5.48)
txt(CX+CW/2, CY+CH-5.72, 'Local Contrast Score',
    fs=7.5, col=C['navy'], bold=True)
txt(CX+CW/2, CY+CH-6.02, r'displacement vs. neighbourhood $\sigma$',
    fs=6.8, col=C['red_d'], italic=True)

# ── C → D arrow ─────────────────────────────────────────────────────────────
arr(CX+CW, CY+BH-4.6, 14.35, CY+BH-4.6, col=C['navy'])
txt((CX+CW+14.35)/2, CY+BH-4.32, 'evidence state', fs=7.5, col=C['navy'])

# ══════════════════════════════════════════════════════════════════════════════
# MODULE D — RISK OUTPUT  (right column, three stacked output boxes)
# ══════════════════════════════════════════════════════════════════════════════
DX = 14.35
txt(DX+1.10, CY+BH-0.20, 'D.  RISK OUTPUT', fs=9.0, col=C['navy'], bold=True)

# Drift compensation (top box)
rbox(DX, CY+BH-1.18, 2.20, 0.85, fc=C['amber_l'], ec=C['amber'], lw=1.5, r=0.14)
txt(DX+1.10, CY+BH-0.66, 'Drift Compensation', fs=7.2, col=C['amber'], bold=True)
txt(DX+1.10, CY+BH-0.96, r'IRLS bias $\hat{\mathbf{b}}$  (feedback)',
    fs=7.0, col='#5d4037')

# Three output signal boxes
out_specs = [
    (C['red_d'],   C['red_l'],   'Motion Clusters',
     r'$\bar{\mathbf{u}}_C$, mag, count'),
    (C['purple'],  C['purp_l'],  'Risk Regions',
     'spatial extent + confidence'),
    (C['teal_d'],  C['teal_l'],  'Structure Motion',
     'magnitude + timestamp'),
]
oy = CY + BH - 1.38
for fc_o, fl_o, title_o, sub_o in out_specs:
    oy -= 1.48
    rbox(DX, oy, 2.20, 1.32, fc=fl_o, ec=fc_o, lw=1.8, r=0.15)
    txt(DX+1.10, oy+0.98, title_o, fs=7.5, col=fc_o, bold=True)
    txt(DX+1.10, oy+0.56, sub_o,   fs=6.8, col=fc_o)
    # Small icon
    ax.add_patch(Circle((DX+0.22, oy+0.76), 0.14, fc=fc_o, ec='white',
                          lw=0.8, zorder=5))

# Arrow: C-bottom REACQ bypass to D risk regions
arr(CX+CW, CY+0.88, DX, CY+0.88, col=C['purple'], lw=1.5, hw=0.14, hl=0.18)
txt((CX+CW+DX)/2, CY+0.60, 'reacq. channel', fs=6.5, col=C['purple'], italic=True)

# ══════════════════════════════════════════════════════════════════════════════
# DRIFT FEEDBACK ARROW  (D → A, below the input strip)
# ══════════════════════════════════════════════════════════════════════════════
# Path: D drift box bottom → down to y=0.32 → left to AX+0.5 → up into Module A
fb_x0 = DX + 1.10; fb_y0 = CY + BH - 1.18        # bottom of drift box
fb_y_low = 0.30

ax.plot([fb_x0, fb_x0], [fb_y0, fb_y_low],
        color=C['amber'], lw=1.6, ls='--', zorder=6)
ax.plot([AX+0.50, fb_x0], [fb_y_low, fb_y_low],
        color=C['amber'], lw=1.6, ls='--', zorder=6)
arr(AX+0.50, fb_y_low, AX+0.50, AY+0.10+0.05,
    col=C['amber'], lw=1.6, hw=0.13, hl=0.16)
txt((AX+0.50+fb_x0)/2, fb_y_low+0.14,
    r'background drift $\hat{\mathbf{b}}$ fed back to observation stage',
    fs=7.0, col=C['amber'], italic=True)

# ══════════════════════════════════════════════════════════════════════════════
# INPUT STRIP: label the three processing columns
# ══════════════════════════════════════════════════════════════════════════════
for xc, lb in [(AX+AW/2, 'Observation\n& Anchoring'),
               (BX+BW/2, 'Per-anchor\nState Estimation'),
               (CX+CW/2, 'Evidence\nVerification')]:
    txt(xc, 0.92, lb, fs=7, col=C['navy'], italic=True, alpha=0.65)

# ══════════════════════════════════════════════════════════════════════════════
# REACQUISITION label between B and C at bottom
# ══════════════════════════════════════════════════════════════════════════════
txt((BX+BW+CX)/2, BY+0.55, '← bypass IMM →', fs=6.5, col=C['purple'], italic=True)

# ══════════════════════════════════════════════════════════════════════════════
# SAVE
# ══════════════════════════════════════════════════════════════════════════════
def main():
    out = common.result_root() / 'summary' / 'figures' / 'system_overview_draft.png'
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=200, bbox_inches='tight', facecolor=C['bg'])
    print(f'Saved: {out}')


if __name__ == '__main__':
    main()
