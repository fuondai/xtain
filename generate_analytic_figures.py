#!/usr/bin/env python3
"""Generate 5 analytical charts (vector PDF) for CrossTaint evaluation.

Uses pubfig for IEEE style. All plotted values are closed-form analytic bounds
computed from the operating-point parameters defined below.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import os
from pubfig import setup, save, COLUMN_W, CB_COLORS

setup()

import os
import sys

# Get the script's directory to set paths relative to it
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, "figures"))
os.makedirs(OUT_DIR, exist_ok=True)

BETA = 0.040
RHO = 0.81
DELTA = 0.21
ETA = 0.04
K_OBS = 11
G = 4

# All chart values are analytical (closed-form bounds); disclosure belongs only in
# the LaTeX \caption{}, never inside the chart area.

# Chart 1: Recall lower bound vs hop depth k (prop-recall)
ks = np.arange(1, 17)
recall_lb = (1 - BETA) ** ks
fig, ax = plt.subplots(figsize=(COLUMN_W, COLUMN_W * 0.72))
ax.plot(ks, recall_lb, color=CB_COLORS[0], marker='o', markersize=3, label=r"$(1-\beta)^k$")
ax.axvline(K_OBS, color=CB_COLORS[1], linestyle='--', label=f"k={K_OBS} observed")
ax.set_xlabel("Hop depth k")
ax.set_ylabel("Lower bound on hop recall")
ax.set_title("Analytical recall lower bound (Theorem prop-recall)")
ax.legend(fontsize=7)
ax.grid(True, alpha=0.3)
ax.set_ylim(0, 1.05)
save(fig, os.path.join(OUT_DIR, "fig_recall_bound"))

# Chart 2: Off-trace inclusion prob (independence) vs k
infl_indep = BETA ** ks
fig, ax = plt.subplots(figsize=(COLUMN_W, COLUMN_W * 0.72))
ax.semilogy(ks, infl_indep, color=CB_COLORS[2], marker='s', markersize=3)
ax.axvline(K_OBS, color=CB_COLORS[1], linestyle='--', label=f"k={K_OBS}")
ax.set_xlabel("Hop depth k")
ax.set_ylabel(r"Pr[T_k(a) > \eta] (independence)")
ax.set_title("Off-trace inclusion probability (independence case, th-sound-prob)")
ax.legend(fontsize=7)
ax.grid(True, alpha=0.3)
save(fig, os.path.join(OUT_DIR, "fig_inflation_indep"))

# Chart 3: Correlated union bound vs g (batch groups)
gs = np.arange(1, 9)
union_b = gs * BETA
fig, ax = plt.subplots(figsize=(COLUMN_W, COLUMN_W * 0.72))
ax.plot(gs, union_b, color=CB_COLORS[3], marker='^', markersize=4)
ax.axhline(0.16, color=CB_COLORS[1], linestyle=':', label="g=4 operating point")
ax.set_xlabel("Number of batch groups g")
ax.set_ylabel(r"Union bound Pr[T_k(a) > \eta]")
ax.set_title("Correlated-error union bound (th-correl)")
ax.legend(fontsize=7)
ax.grid(True, alpha=0.3)
save(fig, os.path.join(OUT_DIR, "fig_union_g"))

# Chart 4: Max hop depth before threshold (kmax vs eta) for fixed rho
etas = np.logspace(-4, -1, 20)
kmaxs = np.ceil(np.log(etas) / np.log(RHO)).astype(int)
fig, ax = plt.subplots(figsize=(COLUMN_W, COLUMN_W * 0.72))
ax.semilogx(etas, kmaxs, color=CB_COLORS[4], marker='d', markersize=3)
ax.axhline(16, color=CB_COLORS[1], linestyle='--', label="k_max_bound=16")
ax.set_xlabel(r"Threshold \eta")
ax.set_ylabel("Max hop depth before bound")
ax.set_title(r"Termination depth $\lceil \log(\eta)/\log(\rho) \rceil$ (T3)")
ax.legend(fontsize=7)
ax.grid(True, alpha=0.3)
save(fig, os.path.join(OUT_DIR, "fig_kmax_eta"))

# Chart 5: Sensitivity of recall bound to beta (at fixed k=11)
betas = np.linspace(0.01, 0.15, 15)
rec_sens = (1 - betas) ** K_OBS
fig, ax = plt.subplots(figsize=(COLUMN_W, COLUMN_W * 0.72))
ax.plot(betas, rec_sens, color=CB_COLORS[5], marker='o', markersize=3)
ax.axvline(BETA, color=CB_COLORS[1], linestyle='--', label=r"$\beta=0.04$")
ax.set_xlabel(r"Mismatch parameter $\beta$")
ax.set_ylabel("Recall lower bound at k=11")
ax.set_title("Sensitivity of recall bound to mismatch rate (prop-recall)")
ax.legend(fontsize=7)
ax.grid(True, alpha=0.3)
save(fig, os.path.join(OUT_DIR, "fig_sens_beta"))

print("Generated 5 analytical vector charts in", OUT_DIR)
