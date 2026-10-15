#!/usr/bin/env python
"""Generate research-meeting-quality plots + a summary table from the 25-ep
full benchmark CSV (/tmp/fullbench25/results.csv).

Outputs to results_pap/bench25/:
  - fig1_per_object_grouped.png   grouped bars, ra vs la full-success per object
  - fig2_heatmap.png              objects x {ra/la grasp/full} heatmap
  - fig3_ranking.png              combined per-object ranking vs 90% target
  - fig4_summary_kpi.png          headline KPI panel
  - summary_table.md / .csv       the numbers

Run:  /home/robot/anaconda3/envs/env_isaaclab/bin/python example_grasp/plot_bench25.py
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager  # noqa

CSV = os.environ.get("BENCH_CSV", "/tmp/fullbench25/results.csv")
OUT = os.path.join(os.path.dirname(__file__), "..", "results_pap", "bench25")
OUT = os.path.abspath(OUT)
os.makedirs(OUT, exist_ok=True)

# ---- modern research aesthetic -------------------------------------------------
RA_C = "#2563EB"   # blue
LA_C = "#F59E0B"   # amber
GOOD = "#16A34A"   # green
BAD  = "#DC2626"   # red
INK  = "#1E293B"
GRID = "#E2E8F0"
plt.rcParams.update({
    "figure.dpi": 140, "savefig.dpi": 220,
    "font.family": "DejaVu Sans", "font.size": 12,
    "axes.edgecolor": "#94A3B8", "axes.linewidth": 1.0,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.9,
    "axes.axisbelow": True, "axes.spines.top": False, "axes.spines.right": False,
    "text.color": INK, "axes.labelcolor": INK, "xtick.color": INK, "ytick.color": INK,
})

def pct(s, t):
    return 100.0 * s / t if t else 0.0

# ---- load ----------------------------------------------------------------------
df = pd.read_csv(CSV)
for c in ["grasp_succ", "grasp_tot", "full_succ", "full_tot"]:
    df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0).astype(int)
df["full_pct"] = df.apply(lambda r: pct(r.full_succ, r.full_tot), axis=1)
df["grasp_pct"] = df.apply(lambda r: pct(r.grasp_succ, r.grasp_tot), axis=1)

piv_full = df.pivot_table(index="object", columns="arm", values="full_pct")
piv_grasp = df.pivot_table(index="object", columns="arm", values="grasp_pct")
for p in (piv_full, piv_grasp):
    for a in ("ra", "la"):
        if a not in p.columns:
            p[a] = np.nan
piv_full["combined"] = piv_full[["ra", "la"]].mean(axis=1)
order = piv_full.sort_values("combined", ascending=False).index.tolist()
piv_full = piv_full.loc[order]
piv_grasp = piv_grasp.loc[order]

# overall numbers
tot_full = df.full_succ.sum(); tot_n = df.full_tot.sum()
overall = pct(tot_full, tot_n)
ra_overall = pct(df[df.arm=="ra"].full_succ.sum(), df[df.arm=="ra"].full_tot.sum())
la_overall = pct(df[df.arm=="la"].full_succ.sum(), df[df.arm=="la"].full_tot.sum())
n_ge90 = int((piv_full["combined"] >= 90).sum())
n_obj = len(order)

# ---- summary table -------------------------------------------------------------
tbl = piv_full.copy()
tbl["ra_grasp"] = piv_grasp["ra"]; tbl["la_grasp"] = piv_grasp["la"]
tbl = tbl[["ra", "la", "combined", "ra_grasp", "la_grasp"]].round(1)
tbl.columns = ["ra_full%", "la_full%", "combined%", "ra_grasp%", "la_grasp%"]
tbl.to_csv(os.path.join(OUT, "summary_table.csv"))
with open(os.path.join(OUT, "summary_table.md"), "w") as fh:
    fh.write(f"# 25-episode benchmark — {n_obj} objects, both arms ({tot_n} episodes)\n\n")
    fh.write(f"**Overall full success: {overall:.1f}%**  |  ra {ra_overall:.1f}%  |  la {la_overall:.1f}%  |  "
             f"{n_ge90}/{n_obj} objects ≥ 90% combined\n\n")
    # manual markdown table (no tabulate dependency)
    hdr = ["object"] + list(tbl.columns)
    fh.write("| " + " | ".join(hdr) + " |\n")
    fh.write("|" + "|".join(["---"] * len(hdr)) + "|\n")
    for obj, row in tbl.iterrows():
        cells = [obj] + [("" if pd.isna(v) else f"{v:.1f}") for v in row.values]
        fh.write("| " + " | ".join(cells) + " |\n")
print(f"overall={overall:.1f}% ra={ra_overall:.1f}% la={la_overall:.1f}% ge90={n_ge90}/{n_obj}")

def barlabels(ax, bars, vals, horiz=False, fs=9):
    for b, v in zip(bars, vals):
        if np.isnan(v):
            continue
        if horiz:
            ax.text(b.get_width()+1.2, b.get_y()+b.get_height()/2, f"{v:.0f}",
                    va="center", ha="left", fontsize=fs, color=INK)
        else:
            ax.text(b.get_x()+b.get_width()/2, b.get_height()+1.2, f"{v:.0f}",
                    va="bottom", ha="center", fontsize=fs, color=INK)

# ---- FIG 1: grouped horizontal bars, ra vs la full success ---------------------
y = np.arange(n_obj)[::-1]; h = 0.38
fig, ax = plt.subplots(figsize=(11, 9))
b1 = ax.barh(y+h/2, piv_full["ra"].values, height=h, color=RA_C, label="Right arm (ra)")
b2 = ax.barh(y-h/2, piv_full["la"].values, height=h, color=LA_C, label="Left arm (la)")
barlabels(ax, b1, piv_full["ra"].values, horiz=True)
barlabels(ax, b2, piv_full["la"].values, horiz=True)
ax.axvline(90, color=GOOD, ls="--", lw=1.6, alpha=0.8)
ax.text(90, n_obj-0.3, " 90% target", color=GOOD, fontsize=10, va="top")
ax.set_yticks(y); ax.set_yticklabels(order)
ax.set_xlim(0, 108); ax.set_xlabel("Full pick-and-place success (%)")
ax.set_title("BODex grasp success by object and arm  ·  25 episodes each",
             fontsize=15, fontweight="bold", loc="left", pad=12)
ax.legend(loc="lower right", frameon=False)
fig.tight_layout(); fig.savefig(os.path.join(OUT, "fig1_per_object_grouped.png")); plt.close(fig)

# ---- FIG 2: heatmap ------------------------------------------------------------
mat = np.column_stack([piv_grasp["ra"], piv_full["ra"], piv_grasp["la"], piv_full["la"]])
cols = ["ra\ngrasp", "ra\nfull", "la\ngrasp", "la\nfull"]
fig, ax = plt.subplots(figsize=(7.5, 9))
im = ax.imshow(mat, cmap="RdYlGn", vmin=0, vmax=100, aspect="auto")
ax.set_xticks(range(4)); ax.set_xticklabels(cols, fontsize=11)
ax.set_yticks(range(n_obj)); ax.set_yticklabels(order)
for i in range(n_obj):
    for j in range(4):
        v = mat[i, j]
        if not np.isnan(v):
            ax.text(j, i, f"{v:.0f}", ha="center", va="center",
                    fontsize=9.5, color="#111" if 25 < v < 80 else "white", fontweight="bold")
ax.set_title("Success-rate heatmap (%)", fontsize=15, fontweight="bold", loc="left", pad=12)
cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04); cb.set_label("success %")
ax.set_xticks(np.arange(-.5, 4, 1), minor=True); ax.set_yticks(np.arange(-.5, n_obj, 1), minor=True)
ax.grid(which="minor", color="white", linewidth=2); ax.tick_params(which="minor", length=0)
fig.tight_layout(); fig.savefig(os.path.join(OUT, "fig2_heatmap.png")); plt.close(fig)

# ---- FIG 3: combined ranking vs 90% target -------------------------------------
comb = piv_full["combined"].values
colors = [GOOD if v >= 90 else (LA_C if v >= 75 else BAD) for v in comb]
fig, ax = plt.subplots(figsize=(11, 9))
bars = ax.barh(np.arange(n_obj)[::-1], comb, color=colors)
barlabels(ax, bars, comb, horiz=True)
ax.axvline(90, color=GOOD, ls="--", lw=1.6); ax.text(90, n_obj-0.3, " 90%", color=GOOD, va="top")
ax.set_yticks(np.arange(n_obj)[::-1]); ax.set_yticklabels(order)
ax.set_xlim(0, 108); ax.set_xlabel("Combined full success (ra+la average, %)")
ax.set_title(f"Per-object grasp reliability  ·  {n_ge90}/{n_obj} objects ≥ 90%",
             fontsize=15, fontweight="bold", loc="left", pad=12)
import matplotlib.patches as mp
leg = [mp.Patch(color=GOOD, label="≥ 90%"), mp.Patch(color=LA_C, label="75–90%"),
       mp.Patch(color=BAD, label="< 75%")]
ax.legend(handles=leg, loc="lower right", frameon=False)
fig.tight_layout(); fig.savefig(os.path.join(OUT, "fig3_ranking.png")); plt.close(fig)

# ---- FIG 4: KPI panel ----------------------------------------------------------
fig, ax = plt.subplots(figsize=(11, 3.0)); ax.axis("off")
kpis = [("Overall success", f"{overall:.1f}%", RA_C),
        ("Right arm (ra)", f"{ra_overall:.1f}%", RA_C),
        ("Left arm (la)", f"{la_overall:.1f}%", LA_C),
        (f"Objects ≥ 90%", f"{n_ge90}/{n_obj}", GOOD),
        ("Episodes", f"{int(tot_n)}", INK)]
for i, (lab, val, c) in enumerate(kpis):
    x = i / len(kpis) + 0.5/len(kpis)
    ax.text(x, 0.62, val, ha="center", va="center", fontsize=30, fontweight="bold", color=c, transform=ax.transAxes)
    ax.text(x, 0.20, lab, ha="center", va="center", fontsize=12, color="#475569", transform=ax.transAxes)
ax.set_title("BODex dexterous grasping — benchmark summary (25 ep/cell, both arms)",
             fontsize=15, fontweight="bold", loc="left")
fig.tight_layout(); fig.savefig(os.path.join(OUT, "fig4_summary_kpi.png")); plt.close(fig)

print("wrote plots + tables to", OUT)
for f in sorted(os.listdir(OUT)):
    print("  ", f)
