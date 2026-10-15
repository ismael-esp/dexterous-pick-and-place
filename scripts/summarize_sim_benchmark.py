#!/usr/bin/env python3
"""Summarise a scripts/run_sim_benchmark.sh output dir into CSVs, a Markdown table and plots.

    python scripts/summarize_sim_benchmark.py runs/sim_benchmark [--out docs/results/sim]
"""
import argparse
import glob
import os
import re

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ORDER = ["green_cube", "red_cube", "pikachu", "whale", "snorlax", "tiger",
         "milk", "mustard", "green_can", "tea_peach"]
ARM_NAME = {"ra": "right arm (UR3)", "la": "left arm (UR3e)"}
ARM_COLOR = {"ra": "#2563eb", "la": "#ea580c"}

R_LOAD = re.compile(r"trajectory loaded T=(\d+).*?approach=(\w+) obj=\(([-\d.]+),([-\d.]+),([-\d.]+)\)"
                    r" place=\(([-\d.]+),([-\d.]+),([-\d.]+)\)")
R_SYN = re.compile(r"bodex_cli=([\d.]+)s")
R_EVAL = {
    "lifted_by": re.compile(r"lifted_by\s+=\s+([-\d.]+) m\s+grasp_success=(\w+)"),
    "xy": re.compile(r"final_xy_err\s+=\s+([-\d.]+)"),
    "z": re.compile(r"final_z_err\s+=\s+([-\d.]+)"),
    "place": re.compile(r"place_success\s+=\s+(\w+)"),
    "attempts": re.compile(r"attempts\s+=\s+(\d+)"),
    "overall": re.compile(r"OVERALL SUCCESS\s+=\s+(\w+)"),
}


def parse_log(path, arm, obj):
    txt = open(path, errors="replace").read()
    blocks = re.split(r"=== Episode (\d+)/\d+ ===", txt)[1:]
    eps = []
    for i in range(0, len(blocks), 2):
        n, b = int(blocks[i]), blocks[i + 1]
        r = dict(arm=arm, object=obj, episode=n)
        m = R_LOAD.search(b)
        if m:
            r.update(traj_len=int(m[1]), approach=m[2], obj_x=float(m[3]), obj_y=float(m[4]),
                     obj_z=float(m[5]), place_x=float(m[6]), place_y=float(m[7]))
        m = R_SYN.search(b)
        r["synth_s"] = float(m[1]) if m else np.nan
        m = R_EVAL["lifted_by"].search(b)
        r["lifted_by_m"] = float(m[1]) if m else np.nan
        r["grasp_success"] = bool(m and m[2] == "True")
        for k in ("xy", "z"):
            m = R_EVAL[k].search(b)
            r[f"place_{k}_err_m"] = float(m[1]) if m else np.nan
        m = R_EVAL["place"].search(b)
        r["place_success"] = bool(m and m[1] == "True")
        m = R_EVAL["attempts"].search(b)
        r["attempts"] = int(m[1]) if m else np.nan
        m = R_EVAL["overall"].search(b)
        r["success"] = bool(m and m[1] == "True")
        r["evaluated"] = m is not None
        eps.append(r)
    return eps


def main():
    p = argparse.ArgumentParser()
    p.add_argument("run_dir")
    p.add_argument("--out", default=None)
    a = p.parse_args()
    out = a.out or os.path.join(a.run_dir, "summary")
    os.makedirs(out, exist_ok=True)

    eps = []
    for f in sorted(glob.glob(os.path.join(a.run_dir, "logs", "*_*.log"))):
        name = os.path.basename(f)[:-4]
        if name.endswith("_vid") or name == "worker":
            continue
        arm, obj = name.split("_", 1)
        eps += parse_log(f, arm, obj)
    d = pd.DataFrame(eps)
    d.to_csv(os.path.join(out, "sim_episodes.csv"), index=False, float_format="%.4f")

    g = d.groupby(["arm", "object"])
    s = pd.DataFrame(dict(
        episodes=g.size(), grasp=g.grasp_success.sum(), place=g.place_success.sum(),
        success=g.success.sum(), approach=g.approach.agg(lambda x: x.dropna().mode().iloc[0] if x.notna().any() else ""),
        synth_s=g.synth_s.mean(), place_xy_err_cm=g.place_xy_err_m.mean() * 100,
        lifted_cm=g.lifted_by_m.mean() * 100)).reset_index()
    s["rate"] = s.success / s.episodes
    s.to_csv(os.path.join(out, "sim_summary.csv"), index=False, float_format="%.3f")

    arms = [x for x in ("ra", "la") if x in set(s.arm)]
    objs = [o for o in ORDER if o in set(s.object)]
    lines = ["| object | approach | " + " | ".join(f"{ARM_NAME[x]}" for x in arms) + " |",
             "|---|---|" + "---|" * len(arms)]
    for o in objs:
        cells = []
        for x in arms:
            r = s[(s.arm == x) & (s.object == o)]
            cells.append(f"{int(r.success.iloc[0])}/{int(r.episodes.iloc[0])}" if len(r) else "-")
        appr = d[d.object == o].approach.dropna().mode().iloc[0]
        lines.append(f"| `{o}` | {appr} | " + " | ".join(cells) + " |")
    tot = [f"**{int(s[s.arm == x].success.sum())}/{int(s[s.arm == x].episodes.sum())} "
           f"({100 * s[s.arm == x].success.sum() / s[s.arm == x].episodes.sum():.0f}%)**" for x in arms]
    lines.append("| **all** | | " + " | ".join(tot) + " |")
    md = "\n".join(lines)
    open(os.path.join(out, "sim_table.md"), "w").write(md + "\n")
    print(md)
    print(f"\noverall {int(d.success.sum())}/{len(d)} = {100 * d.success.mean():.1f}%  "
          f"grasp {100 * d.grasp_success.mean():.1f}%  mean synth {d.synth_s.mean():.1f}s")

    # --- plots --------------------------------------------------------------------------
    plt.rcParams.update({"font.size": 11, "axes.spines.top": False, "axes.spines.right": False})
    fig, ax = plt.subplots(figsize=(12, 4.2), dpi=150)
    w = 0.38
    x = np.arange(len(objs))
    for k, arm in enumerate(arms):
        r = [s[(s.arm == arm) & (s.object == o)].rate.sum() * 100 for o in objs]
        b = ax.bar(x + (k - (len(arms) - 1) / 2) * w, r, w, label=ARM_NAME[arm], color=ARM_COLOR[arm])
        ax.bar_label(b, fmt="%.0f", fontsize=8, padding=2)
    ax.set_xticks(x, objs, rotation=20)
    ax.set_ylim(0, 112)
    ax.set_ylabel("full pick-and-place success [%]")
    ax.set_title("Isaac Lab: success per object and arm")
    ax.legend(frameon=False, loc="lower left")
    fig.tight_layout()
    fig.savefig(os.path.join(out, "sim_success_by_object.png"))

    fig, axs = plt.subplots(1, 2, figsize=(12, 4.2), dpi=150)
    for arm in arms:
        da = d[d.arm == arm]
        axs[0].scatter(da.obj_x[da.success], da.obj_y[da.success], s=22, color=ARM_COLOR[arm],
                       alpha=.8, label=f"{ARM_NAME[arm]} success")
        axs[0].scatter(da.obj_x[~da.success], da.obj_y[~da.success], s=60, marker="x",
                       color=ARM_COLOR[arm], label=f"{ARM_NAME[arm]} fail")
    axs[0].set(xlabel="object x [m] (world)", ylabel="object y [m] (world)",
               title="Spawn positions by outcome")
    axs[0].axis("equal")
    axs[0].legend(frameon=False, fontsize=8)
    data = [d[(d.arm == arm)].synth_s.dropna() for arm in arms]
    bp = axs[1].boxplot(data, tick_labels=[ARM_NAME[x] for x in arms], patch_artist=True, widths=.5)
    for patch, arm in zip(bp["boxes"], arms):
        patch.set_facecolor(ARM_COLOR[arm])
        patch.set_alpha(.35)
    axs[1].set(ylabel="BODex grasp + trajectory synthesis [s]", title="Synthesis time per episode")
    fig.tight_layout()
    fig.savefig(os.path.join(out, "sim_workspace_and_timing.png"))
    print("wrote", out)


if __name__ == "__main__":
    main()
