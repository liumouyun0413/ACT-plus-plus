"""Generate Gantt chart PNG for the ACT-plus-plus 9-week plan."""
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib import rcParams
from datetime import date, timedelta

# Chinese font
rcParams['font.sans-serif'] = ['Noto Sans CJK SC', 'WenQuanYi Zen Hei',
                                'Source Han Sans SC', 'SimHei', 'DejaVu Sans']
rcParams['axes.unicode_minus'] = False

START = date(2026, 5, 8)

# (name, section, start_offset_days, duration_days, color)
tasks = [
    ("硬件/软件/标定",       "基础建设",  0,   3, "#9FB4E8"),
    ("Phase 1 采集",         "Phase 1 数据",  3,  14, "#FFD79A"),
    ("数据转换与一致性校验", "Phase 1 数据", 17,   1, "#FFD79A"),
    ("p1 打基础",            "Phase 1 训练", 18,   2, "#B7D7A8"),
    ("p2 收敛",              "Phase 1 训练", 20,   2, "#B7D7A8"),
    ("p3 优化",              "Phase 1 训练", 22,   1, "#B7D7A8"),
    ("Phase 1 在线评估",     "Phase 1 训练", 23,   2, "#B7D7A8"),
    ("Phase 2 采集",         "Phase 2 数据", 25,  16, "#F6B6C6"),
    ("Phase 2 warm-start 训练","Phase 2 训练",41,  4, "#C7A6E0"),
    ("Phase 2 评估+Phase 1 回测","Phase 2 训练",45,2, "#C7A6E0"),
    ("Thor 部署+10 场景演示","部署验收",   47,   6, "#E8A6A6"),
]

fig, ax = plt.subplots(figsize=(14, 6.5))

for i, (name, section, off, dur, color) in enumerate(tasks):
    ax.barh(i, dur, left=off, height=0.6, color=color,
            edgecolor="#444", linewidth=0.6)
    # task label centered in bar
    ax.text(off + dur / 2, i, name, ha="center", va="center",
            fontsize=10, color="#222")

ax.set_yticks(range(len(tasks)))
ax.set_yticklabels([t[1] for t in tasks], fontsize=10)
ax.invert_yaxis()

# x-axis: date ticks every 7 days
total_days = 53
ax.set_xlim(0, total_days)
xticks = list(range(0, total_days, 7)) + [total_days]
ax.set_xticks(xticks)
ax.set_xticklabels([(START + timedelta(days=d)).strftime("%m-%d") for d in xticks],
                   fontsize=9)
ax.set_xlabel("日期", fontsize=11)

ax.set_title("ACT-plus-plus 堆叠抓取 8 周开发计划", fontsize=14, pad=14)
ax.grid(axis="x", linestyle="--", alpha=0.4)

# legend
sections = [("基础建设", "#9FB4E8"), ("Phase 1 数据", "#FFD79A"),
            ("Phase 1 训练", "#B7D7A8"), ("Phase 2 数据", "#F6B6C6"),
            ("Phase 2 训练", "#C7A6E0"), ("部署验收", "#E8A6A6")]
handles = [mpatches.Patch(color=c, label=n) for n, c in sections]
ax.legend(handles=handles, loc="lower right", fontsize=9, ncol=3)

plt.tight_layout()
out = "docs/gantt_9week.png"
import os
os.makedirs("docs", exist_ok=True)
plt.savefig(out, dpi=160, bbox_inches="tight")
print(f"saved: {out}")
