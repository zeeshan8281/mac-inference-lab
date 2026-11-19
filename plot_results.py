"""Render one measured graph; never synthesize benchmark values."""
import json
import random
import statistics
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

run = Path(sys.argv[1])
summary = json.loads((run / "summary.json").read_text())
episodes = [json.loads(line) for line in (run / "episodes.jsonl").read_text().splitlines()]
index = {x["sequence"]: x["policy_mode"] for x in
         (json.loads(line) for line in (run / "test_index.jsonl").read_text().splitlines())}
profiles = ["solo_short", "solo_long", "burst_short", "burst_mixed"]
labels = ["1 request\nshort prompt", "1 request\nlong prompt", "4 requests\nshort prompts", "4 requests\nmixed prompts"]
rng = random.Random(901)
fig, ax = plt.subplots(figsize=(12.5, 7.7), dpi=160)
fig.patch.set_facecolor("#f8fafc")
ax.set_facecolor("#f8fafc")
colors = {"fixed": "#5b6ee1", "adaptive": "#008a78"}
for offset, mode in [(-.18, "fixed"), (.18, "adaptive")]:
    values, lower, upper = [], [], []
    for profile in profiles:
        sample = [e["metrics"]["goodput_per_min"] for e in episodes
                  if e["phase"] == "test" and e["profile"] == profile and index[e["sequence"]] == mode]
        value = statistics.mean(sample)
        draws = sorted(statistics.mean(rng.choices(sample, k=len(sample))) for _ in range(5000))
        values.append(value)
        lower.append(value - draws[125])
        upper.append(draws[4874] - value)
    bars = ax.bar(np.arange(4) + offset, values, .32, color=colors[mode],
                  label="Best fixed setting" if mode == "fixed" else "Adaptive selector",
                  yerr=[lower, upper], capsize=4, error_kw={"linewidth": 1.2})
    ax.bar_label(bars, labels=[f"{v:.1f}" for v in values], padding=9, fontsize=12, fontweight="bold")
ax.set_xticks(range(4), labels, fontsize=12)
ax.set_ylabel("Correct responses before deadline / minute", fontsize=12)
ax.spines[["top", "right", "left"]].set_visible(False)
ax.spines["bottom"].set_color("#cbd5e1")
ax.yaxis.grid(True, color="#dce3eb", linewidth=.7)
ax.set_axisbelow(True)
ax.set_ylim(0, ax.get_ylim()[1] * 1.18)
ax.legend(loc="upper left", frameon=False, ncol=2, fontsize=12)
fig.text(.085, .94, "Can a measured policy beat one fixed setting?", fontsize=22, weight="bold", color="#172235")
fig.text(.085, .893, "Qwen2.5-Coder 7B 4-bit · M5 Pro · 48 GiB unified memory · greedy decoding", fontsize=12, color="#536174")
p = summary["paired_result"]
change = p["relative_change_percent"]
lo, hi = p["difference_95ci"]
fig.text(.085, .10,
         f"Mean change: {change:+.1f}%  |  paired difference: {p['difference_goodput']:+.2f} responses/min"
         f" (95% CI {lo:+.2f} to {hi:+.2f})",
         fontsize=11.5, color="#172235")
requests = sum(g[m]["requests"] for g in summary["profiles"].values() for m in ["fixed", "adaptive"])
fig.text(.085, .055,
         f"{p['blocks']} paired held-out workload blocks · {requests} requests · warm resident models · finite bursts\n"
         "Calibration-only policy selection. Error bars: bootstrap 95% intervals. This is a controlled extraction pilot.",
         fontsize=10, color="#536174", linespacing=1.6)
fig.subplots_adjust(left=.085, right=.975, top=.84, bottom=.22)
path = run / "comparison.png"
fig.savefig(path, facecolor=fig.get_facecolor())
print(path.resolve())
