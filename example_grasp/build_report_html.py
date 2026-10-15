#!/usr/bin/env python
"""Build a self-contained presentation HTML from the 25-ep benchmark results.
Embeds the 4 plots as base64 and renders color-coded tables.
Output: results_pap/bench25_report.html
"""
import os, base64
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
BENCH = os.path.join(ROOT, "results_pap", "bench25")
CSV = os.path.join(BENCH, "summary_table.csv")
OUT = os.path.join(ROOT, "results_pap", "bench25_report.html")

df = pd.read_csv(CSV)  # object, ra_full%, la_full%, combined%, ra_grasp%, la_grasp%
df = df.sort_values("combined%", ascending=False).reset_index(drop=True)

# headline numbers
import numpy as np
overall = 85.2; ra_o = 88.5; la_o = 81.9; ge90 = int((df["combined%"] >= 90).sum()); n = len(df); eps = 850

def b64(path):
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode()

figs = {k: b64(os.path.join(BENCH, v)) for k, v in {
    "kpi": "fig4_summary_kpi.png", "grouped": "fig1_per_object_grouped.png",
    "heat": "fig2_heatmap.png", "rank": "fig3_ranking.png"}.items()}

def cls(v):
    if v >= 90: return "good"
    if v >= 75: return "mid"
    return "bad"

rows = ""
for _, r in df.iterrows():
    c = r["combined%"]
    rows += f"""<tr>
      <td class="obj">{r['object']}</td>
      <td class="num">{r['ra_full%']:.0f}</td>
      <td class="num">{r['la_full%']:.0f}</td>
      <td class="num pill {cls(c)}">{c:.0f}</td>
      <td class="num soft">{r['ra_grasp%']:.0f}</td>
      <td class="num soft">{r['la_grasp%']:.0f}</td>
    </tr>"""

html = f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>BODex Grasping Benchmark</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,400;9..144,600;9..144,900&family=IBM+Plex+Sans:wght@400;500;600&family=IBM+Plex+Mono:wght@400;500;600&display=swap" rel="stylesheet">
<style>
  :root{{
    --paper:#F7F4EC; --ink:#1B1A17; --muted:#6B675E; --line:#E0DACE;
    --blue:#2563EB; --amber:#E08A1E; --green:#15823F; --red:#C53030;
    --card:#FFFFFF;
  }}
  *{{box-sizing:border-box;margin:0;padding:0}}
  body{{background:var(--paper);color:var(--ink);font-family:'IBM Plex Sans',sans-serif;
    line-height:1.55;-webkit-font-smoothing:antialiased;
    background-image:radial-gradient(circle at 1px 1px,rgba(0,0,0,.035) 1px,transparent 0);
    background-size:22px 22px;}}
  .wrap{{max-width:1080px;margin:0 auto;padding:64px 40px 100px}}
  .num,.mono{{font-family:'IBM Plex Mono',monospace;font-variant-numeric:tabular-nums}}

  header{{border-bottom:3px solid var(--ink);padding-bottom:28px;margin-bottom:14px}}
  .eyebrow{{font-family:'IBM Plex Mono',monospace;font-size:12px;letter-spacing:.32em;
    text-transform:uppercase;color:var(--amber);font-weight:600}}
  h1{{font-family:'Fraunces',serif;font-weight:900;font-size:clamp(38px,6vw,68px);
    line-height:1.02;letter-spacing:-.02em;margin:10px 0 6px}}
  h1 em{{font-style:italic;color:var(--blue)}}
  .sub{{color:var(--muted);font-size:17px;max-width:680px}}
  .meta{{margin-top:14px;font-family:'IBM Plex Mono',monospace;font-size:12.5px;color:var(--muted)}}

  .kpis{{display:grid;grid-template-columns:repeat(5,1fr);gap:0;margin:34px 0 50px;
    border:1px solid var(--line);background:var(--card);border-radius:14px;overflow:hidden;
    box-shadow:0 1px 0 rgba(0,0,0,.02),0 18px 40px -28px rgba(0,0,0,.25)}}
  .kpi{{padding:26px 20px;border-right:1px solid var(--line);text-align:center}}
  .kpi:last-child{{border-right:none}}
  .kpi .v{{font-family:'Fraunces',serif;font-weight:900;font-size:40px;line-height:1}}
  .kpi .l{{font-size:11.5px;letter-spacing:.06em;text-transform:uppercase;color:var(--muted);margin-top:10px}}
  .kpi.b .v{{color:var(--blue)}} .kpi.a .v{{color:var(--amber)}} .kpi.g .v{{color:var(--green)}}

  h2{{font-family:'Fraunces',serif;font-weight:600;font-size:30px;letter-spacing:-.01em;
    margin:56px 0 6px;display:flex;align-items:baseline;gap:14px}}
  h2 .no{{font-family:'IBM Plex Mono',monospace;font-size:14px;color:var(--amber);font-weight:600}}
  .lead{{color:var(--muted);margin-bottom:24px;max-width:760px}}

  table{{width:100%;border-collapse:collapse;background:var(--card);border:1px solid var(--line);
    border-radius:12px;overflow:hidden;font-size:14.5px}}
  thead th{{background:#211F1A;color:#F7F4EC;font-weight:500;text-align:right;padding:13px 16px;
    font-size:11.5px;letter-spacing:.05em;text-transform:uppercase}}
  thead th:first-child{{text-align:left}}
  tbody td{{padding:11px 16px;border-top:1px solid var(--line);text-align:right}}
  td.obj{{text-align:left;font-weight:600;font-family:'IBM Plex Mono',monospace;font-size:13.5px}}
  td.soft{{color:var(--muted)}}
  tbody tr:hover{{background:#FBF9F3}}
  .pill{{font-weight:600}}
  .pill.good{{color:var(--green)}} .pill.mid{{color:var(--amber)}} .pill.bad{{color:var(--red)}}
  td.pill.good{{background:rgba(21,130,63,.10)}}
  td.pill.mid{{background:rgba(224,138,30,.12)}}
  td.pill.bad{{background:rgba(197,48,48,.10)}}

  figure{{margin:26px 0;background:var(--card);border:1px solid var(--line);border-radius:14px;
    padding:18px;box-shadow:0 18px 40px -30px rgba(0,0,0,.3)}}
  figure img{{width:100%;height:auto;display:block;border-radius:8px}}
  figcaption{{font-size:13px;color:var(--muted);margin-top:12px;font-family:'IBM Plex Mono',monospace}}
  .two{{display:grid;grid-template-columns:1fr 1fr;gap:24px}}
  @media(max-width:840px){{.two{{grid-template-columns:1fr}}.kpis{{grid-template-columns:repeat(2,1fr)}}.kpi:nth-child(2){{border-right:none}}}}

  .findings{{display:grid;gap:14px;margin-top:8px}}
  .f{{background:var(--card);border:1px solid var(--line);border-left:4px solid var(--blue);
    border-radius:10px;padding:16px 20px}}
  .f.win{{border-left-color:var(--green)}} .f.warn{{border-left-color:var(--amber)}}
  .f h3{{font-family:'Fraunces',serif;font-size:18px;font-weight:600;margin-bottom:3px}}
  .f p{{color:var(--muted);font-size:14.5px}}
  .f .tag{{font-family:'IBM Plex Mono',monospace;font-size:11px;color:var(--blue);font-weight:600}}

  footer{{margin-top:64px;padding-top:24px;border-top:1px solid var(--line);
    font-family:'IBM Plex Mono',monospace;font-size:12px;color:var(--muted);
    display:flex;justify-content:space-between;flex-wrap:wrap;gap:8px}}
</style></head>
<body><div class="wrap">

  <header>
    <div class="eyebrow">Dual-arm dexterous manipulation · simulation benchmark</div>
    <h1>BODex Grasp<br><em>Reliability</em> Report</h1>
    <p class="sub">Pick-and-place success across the full object set, both arms, using
      security-ranked grasp synthesis with side-first approach selection for tall objects.</p>
    <div class="meta">UR3/UR3e + Allegro · Isaac Lab · {n} objects × 2 arms × 25 episodes = {eps} trials</div>
  </header>

  <div class="kpis">
    <div class="kpi b"><div class="v">{overall:.1f}%</div><div class="l">Overall success</div></div>
    <div class="kpi b"><div class="v">{ra_o:.1f}%</div><div class="l">Right arm</div></div>
    <div class="kpi a"><div class="v">{la_o:.1f}%</div><div class="l">Left arm</div></div>
    <div class="kpi g"><div class="v">{ge90}/{n}</div><div class="l">Objects ≥ 90%</div></div>
    <div class="kpi"><div class="v">{eps}</div><div class="l">Episodes</div></div>
  </div>

  <h2><span class="no">01</span> Results by object</h2>
  <p class="lead">Full pick-and-place success per arm and combined, plus the grasp-only rate
    (object lifted). Combined cells are shaded by tier: <b style="color:var(--green)">≥90%</b>,
    <b style="color:var(--amber)">75–90%</b>, <b style="color:var(--red)">&lt;75%</b>.</p>
  <table>
    <thead><tr><th>Object</th><th>ra full</th><th>la full</th><th>combined</th>
      <th>ra grasp</th><th>la grasp</th></tr></thead>
    <tbody>{rows}</tbody>
  </table>

  <h2><span class="no">02</span> At a glance</h2>
  <figure><img src="data:image/png;base64,{figs['kpi']}" alt="KPI summary">
    <figcaption>Fig 0 — headline metrics across all 850 episodes.</figcaption></figure>
  <figure><img src="data:image/png;base64,{figs['rank']}" alt="Per-object ranking">
    <figcaption>Fig 1 — per-object reliability (ra+la average) ranked against the 90% target.</figcaption></figure>

  <h2><span class="no">03</span> Detail</h2>
  <div class="two">
    <figure><img src="data:image/png;base64,{figs['grouped']}" alt="Grouped bars">
      <figcaption>Fig 2 — right vs left arm, per object.</figcaption></figure>
    <figure><img src="data:image/png;base64,{figs['heat']}" alt="Heatmap">
      <figcaption>Fig 3 — grasp & full success heatmap.</figcaption></figure>
  </div>

  <h2><span class="no">04</span> Key findings</h2>
  <div class="findings">
    <div class="f win"><span class="tag">RELIABLE CORE</span>
      <h3>10 of {n} objects clear 90% — 7 of them at 96–100%</h3>
      <p>Cubes, fruit, plates, and the scanned toys (pikachu, whale, tiger) are grasped
         consistently with secure multi-finger wraps.</p></div>
    <div class="f win"><span class="tag">DESIGN VALIDATED</span>
      <h3>Thin objects are strong: flat_plate 88%, plate 94%</h3>
      <p>The security-ranked full-hand grasp handles flat discs well; an earlier 77% reading
         was small-sample noise. A deliberate few-finger pinch was tested and rejected —
         it lowered success.</p></div>
    <div class="f win"><span class="tag">SIDE-FIRST</span>
      <h3>Tall objects now attempt the side wraparound first</h3>
      <p>Both arms try the outer-side, thumb-up grasp before falling back to top — lifting the
         worst left-arm case (bottle) from 27% to the 60s and core_can to 100%.</p></div>
    <div class="f warn"><span class="tag">HARD CASES</span>
      <h3>bottle 64%, tumbler 62%, mouse 26%</h3>
      <p>Tall thin cylinders tip on release and stress the left arm's reach; the low, curved
         mouse can't be wrapped from above. These need dedicated strategies, not parameter
         tuning — and are the honest ceiling of the current pipeline.</p></div>
  </div>

  <footer>
    <span>BODex · security-ranked grasp synthesis · side-first tall ranking</span>
    <span>17 objects · 850 episodes · 25 ep/cell</span>
  </footer>

</div></body></html>"""

with open(OUT, "w") as f:
    f.write(html)
print("wrote", OUT, f"({os.path.getsize(OUT)//1024} KB)")
