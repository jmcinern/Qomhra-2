#!/usr/bin/env python3
"""MEXA reporting: summary + validation CSVs, three figures, markdown tables.

Runs on the laptop, not LUMI: it consumes the long CSV that mexa_score.py produces on the
cluster and needs matplotlib, which the container does not carry.

Every output derives from `mexa_long.csv` (one row per model state x condition x layer x
pooling), so re-cutting the analysis never means re-running a model.

  python mexa_report.py --long output/mexa_long.csv --evals ../../Qomhra2-Paper/evals
"""
import argparse
import csv
import os
from collections import defaultdict

import numpy as np

# Validated categorical palette (dataviz reference instance, slots 1-5; validated for the
# adjacent pairlist in light mode - worst adjacent CVD dE 9.1). Colour follows the RECIPE,
# never its rank, so a recipe keeps its hue in every figure.
RECIPE_COLOUR = {
    "base": "#52514e",       # secondary ink: base is the reference, not a competitor
    "text": "#2a78d6",
    "speech": "#eb6834",
    "both": "#1baf7a",
    "aligned": "#eda100",
}
RECIPE_ORDER = ["base", "text", "speech", "both", "aligned"]
MUTED, GRID, BASELINE = "#898781", "#e1e0d9", "#c3c2b7"
INK, INK2 = "#0b0b0b", "#52514e"

# Conditions in a fixed reading order: text-only first, then the cross-modal ones.
CONDITION_ORDER = [
    "text_ga~text_en", "speech_en~text_en", "speech_ga~text_ga",
    "speech_ga~text_en", "speech_en~text_ga", "speech_ga~speech_en",
]
CONDITION_LABEL = {
    "text_ga~text_en": "Irish text ~ English text",
    "speech_en~text_en": "English speech ~ English text",
    "speech_ga~text_ga": "Irish speech ~ Irish text",
    "speech_ga~text_en": "Irish speech ~ English text",
    "speech_en~text_ga": "English speech ~ Irish text",
    "speech_ga~speech_en": "Irish speech ~ English speech",
}

# label in the embedding files -> (IWSLT variant/checkpoint, CLUAS model/checkpoint).
# aligned's step_000080 is what IWSLT and CLUAS call its "10%" checkpoint.
DOWNSTREAM_KEY = {
    "base-base": (("base", "base"), ("baseline", "base")),
    "text-10": (("text", "10%"), ("text", "10pct")),
    "text-100": (("text", "final"), ("text", "final")),
    "speech-10": (("speech", "10%"), ("speech", "10pct")),
    "speech-100": (("speech", "final"), ("speech", "final")),
    "both-10": (("both", "10%"), ("both", "10pct")),
    "both-100": (("both", "final"), ("both", "final")),
    "aligned-67": (("aligned", "10%"), ("aligned", "10pct")),
    "aligned-100": (("aligned", "final"), ("aligned", "final")),
}

# A checkpoint is called degenerate when its IWSLT translation score is in the garbage band
# (< 5 chrF++; the handoff's scale reference puts 2-3 at "garbage"). Stated as a rule rather
# than a hand-picked list so the flag is reproducible. It catches speech-final and both-final.
DEGENERATE_CHRFPP = 5.0


def read_long(path):
    with open(path, encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        r["layer"] = int(r["layer"])
        r["mexa"] = float(r["mexa"])
        r["train_fraction"] = float(r["train_fraction"])
        r["step"] = int(r["step"])
    return rows


def summarise(rows):
    """(label, condition, pooling) -> dict with mean/max over layers and the best layer."""
    grouped = defaultdict(list)
    for r in rows:
        grouped[(r["label"], r["condition"], r["pooling"])].append((r["layer"], r["mexa"]))
    out = {}
    for key, pairs in grouped.items():
        pairs.sort()
        values = np.array([v for _, v in pairs])
        meta = next(r for r in rows if (r["label"], r["condition"], r["pooling"]) == key)
        out[key] = {
            "model": meta["model"], "step": meta["step"],
            "train_fraction": meta["train_fraction"],
            "mean": float(values.mean()), "max": float(values.max()),
            "best_layer": int(values.argmax()), "n_layers": len(values),
            "by_layer": values,
        }
    return out


def read_downstream(evals_dir):
    iwslt, cluas = {}, {}
    with open(os.path.join(evals_dir, "IWSLT.csv"), encoding="utf-8-sig", newline="") as f:
        for r in csv.DictReader(f):
            iwslt[(r["variant"], r["checkpoint"])] = r
    with open(os.path.join(evals_dir, "cluas_overall.csv"), encoding="utf-8-sig",
              newline="") as f:
        for r in csv.DictReader(f):
            cluas[(r["model"], r["checkpoint"])] = r
    return iwslt, cluas


# ---------------------------------------------------------------------------
# CSV outputs
# ---------------------------------------------------------------------------
def write_summary(summary, path):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["model", "step", "train_fraction", "label", "condition", "pooling",
                    "mexa_mean_over_layers", "mexa_max_over_layers", "best_layer", "n_layers"])
        for (label, condition, pooling), s in sorted(
                summary.items(), key=lambda kv: (RECIPE_ORDER.index(kv[1]["model"]),
                                                 kv[1]["train_fraction"], kv[0][1], kv[0][2])):
            w.writerow([s["model"], s["step"], s["train_fraction"], label, condition, pooling,
                        round(s["mean"], 6), round(s["max"], 6), s["best_layer"],
                        s["n_layers"]])
    print(f"wrote {path}")


def write_validation(summary, iwslt, cluas, path):
    """MEXA beside the downstream numbers, for the 9 model states that have both.

    FLEURS columns are left out entirely rather than written empty - that benchmark is still
    running, and an empty column reads as a measured zero.
    """
    labels = [l for l in {k[0] for k in summary} if l in DOWNSTREAM_KEY]
    labels.sort(key=lambda l: (RECIPE_ORDER.index(summary[(l, CONDITION_ORDER[0],
                                                           "weighted")]["model"]),
                               summary[(l, CONDITION_ORDER[0], "weighted")]["train_fraction"]))
    header = ["model", "step", "train_fraction", "label", "pooling"]
    header += [f"mexa_max_{c}" for c in CONDITION_ORDER]
    header += ["iwslt_chrfpp", "cluas_listening_gain_pct", "cluas_just_transcript_pct",
               "degenerate"]
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(header)
        for label in labels:
            ikey, ckey = DOWNSTREAM_KEY[label]
            chrfpp = float(iwslt[ikey]["chrfpp"])
            for pooling in ("weighted", "lasttoken"):
                s0 = summary[(label, CONDITION_ORDER[0], pooling)]
                row = [s0["model"], s0["step"], s0["train_fraction"], label, pooling]
                row += [round(summary[(label, c, pooling)]["max"], 4) for c in CONDITION_ORDER]
                row += [round(chrfpp, 3),
                        float(cluas[ckey]["listening_gain_pct"]),
                        float(cluas[ckey]["just_transcript_pct"]),
                        "yes" if chrfpp < DEGENERATE_CHRFPP else "no"]
                w.writerow(row)
    print(f"wrote {path}")


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------
def style_axes(ax):
    ax.set_facecolor("#fcfcfb")
    ax.grid(True, color=GRID, linewidth=0.6, zorder=0)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(BASELINE)
        ax.spines[side].set_linewidth(1.0)
    ax.tick_params(colors=MUTED, labelsize=8, length=3)


def figure_layer_profile(summary, pooling, path):
    """MEXA by layer, one panel per condition, one line per model state.

    Colour carries the RECIPE; opacity carries how far through training the checkpoint is.
    That keeps 33 lines readable without inventing 33 hues.
    """
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 3, figsize=(13, 7), sharex=True, sharey=True)
    for ax, condition in zip(axes.ravel(), CONDITION_ORDER):
        style_axes(ax)
        for (label, cond, pool), s in summary.items():
            if cond != condition or pool != pooling:
                continue
            recipe = s["model"]
            if recipe == "base":
                ax.plot(s["by_layer"], color=RECIPE_COLOUR["base"], linewidth=2.0,
                        linestyle=(0, (4, 2)), zorder=5)
            else:
                # 10% -> faint, 100% -> solid; aligned's two points sit at the top of its range
                alpha = 0.25 + 0.75 * min(1.0, s["train_fraction"])
                ax.plot(s["by_layer"], color=RECIPE_COLOUR[recipe], linewidth=1.4,
                        alpha=alpha, zorder=3)
        ax.set_title(CONDITION_LABEL[condition], fontsize=10, color=INK, loc="left")
        ax.set_ylim(-0.02, 1.02)
    for ax in axes[1]:
        ax.set_xlabel("layer", fontsize=9, color=INK2)
    for ax in axes[:, 0]:
        ax.set_ylabel("MEXA alignment score", fontsize=9, color=INK2)
    handles = [plt.Line2D([], [], color=RECIPE_COLOUR[r], linewidth=2.0,
                          linestyle=(0, (4, 2)) if r == "base" else "-", label=r)
               for r in RECIPE_ORDER]
    fig.legend(handles=handles, loc="lower center", ncol=5, frameon=False, fontsize=9,
               labelcolor=INK2, bbox_to_anchor=(0.5, -0.005))
    fig.suptitle(f"Alignment by layer, all 33 model states ({pooling} pooling; "
                 f"fainter = earlier in training)", fontsize=11, color=INK, x=0.02, ha="left")
    fig.tight_layout(rect=(0, 0.04, 1, 0.97))
    fig.savefig(path, dpi=170, facecolor="#fcfcfb")
    plt.close(fig)
    print(f"wrote {path}")


def figure_trajectory(summary, pooling, aggregate, path):
    """Aggregated MEXA against training progress, one line per recipe, faceted by condition.

    `aligned` is a continuation of `both`'s 90% checkpoint (omni_aligned.yaml: epoch_start
    0.90), so its line is drawn LEAVING the `both` line at 0.90 rather than floating on its
    own axis - the fork is the point.
    """
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 3, figsize=(13, 7), sharex=True, sharey=True)
    for ax, condition in zip(axes.ravel(), CONDITION_ORDER):
        style_axes(ax)
        series = defaultdict(list)
        for (label, cond, pool), s in summary.items():
            if cond != condition or pool != pooling:
                continue
            series[s["model"]].append((s["train_fraction"], s[aggregate]))
        for recipe in RECIPE_ORDER:
            points = sorted(series.get(recipe, []))
            if not points:
                continue
            if recipe == "base":
                ax.axhline(points[0][1], color=RECIPE_COLOUR["base"], linewidth=1.6,
                           linestyle=(0, (4, 2)), zorder=4)
                continue
            if recipe == "aligned":
                # graft the fork point on so the line visibly leaves `both` at 90%
                fork = [p for p in sorted(series.get("both", [])) if abs(p[0] - 0.9) < 1e-6]
                points = fork + points
            x = [p[0] for p in points]
            y = [p[1] for p in points]
            ax.plot(x, y, color=RECIPE_COLOUR[recipe], linewidth=2.0, marker="o",
                    markersize=4.5, zorder=5 if recipe == "aligned" else 3)
        ax.set_title(CONDITION_LABEL[condition], fontsize=10, color=INK, loc="left")
        ax.set_ylim(-0.02, 1.02)
        ax.set_xlim(-0.03, 1.06)
    for ax in axes[1]:
        ax.set_xlabel("fraction of training completed", fontsize=9, color=INK2)
    for ax in axes[:, 0]:
        ax.set_ylabel(f"MEXA ({aggregate} over layers)", fontsize=9, color=INK2)
    handles = [plt.Line2D([], [], color=RECIPE_COLOUR[r], linewidth=2.0,
                          linestyle=(0, (4, 2)) if r == "base" else "-",
                          marker="" if r == "base" else "o", markersize=4.5,
                          label="base (untrained reference)" if r == "base" else r)
               for r in RECIPE_ORDER]
    fig.legend(handles=handles, loc="lower center", ncol=5, frameon=False, fontsize=9,
               labelcolor=INK2, bbox_to_anchor=(0.5, -0.005))
    fig.suptitle(f"Alignment against training progress ({pooling} pooling, {aggregate} over "
                 f"layers). aligned forks from both at 90%.",
                 fontsize=11, color=INK, x=0.02, ha="left")
    fig.tight_layout(rect=(0, 0.04, 1, 0.97))
    fig.savefig(path, dpi=170, facecolor="#fcfcfb")
    plt.close(fig)
    print(f"wrote {path}")


VALIDATION_PANELS = [
    ("text_ga~text_en", "cluas_transcript", "CLUAS reading Irish text (% of marks)"),
    ("speech_ga~text_en", "iwslt", "IWSLT Irish speech to English (chrF++)"),
    ("speech_ga~text_ga", "cluas_gain", "CLUAS listening gain (% of marks)"),
]


def figure_validation(summary, iwslt, cluas, pooling, path):
    """MEXA against the downstream numbers. Degenerate checkpoints are marked, because a
    single dead point can manufacture a correlation on its own."""
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.4))
    for ax, (condition, metric, ylabel) in zip(axes, VALIDATION_PANELS):
        style_axes(ax)
        for label, (ikey, ckey) in DOWNSTREAM_KEY.items():
            key = (label, condition, pooling)
            if key not in summary:
                continue
            s = summary[key]
            chrfpp = float(iwslt[ikey]["chrfpp"])
            y = {"iwslt": chrfpp,
                 "cluas_gain": float(cluas[ckey]["listening_gain_pct"]),
                 "cluas_transcript": float(cluas[ckey]["just_transcript_pct"])}[metric]
            degenerate = chrfpp < DEGENERATE_CHRFPP
            ax.scatter(s["max"], y, s=90, color=RECIPE_COLOUR[s["model"]],
                       edgecolor="#fcfcfb", linewidth=2.0, zorder=5,
                       marker="X" if degenerate else "o")
            ax.annotate(label, (s["max"], y), textcoords="offset points", xytext=(7, 4),
                        fontsize=7.5, color=INK2, zorder=6)
        ax.set_xlabel(f"MEXA: {CONDITION_LABEL[condition]}", fontsize=9, color=INK2)
        ax.set_ylabel(ylabel, fontsize=9, color=INK2)
    handles = [plt.Line2D([], [], marker="X", linestyle="", color=MUTED, markersize=9,
                          label="degenerate (IWSLT chrF++ < 5)"),
               plt.Line2D([], [], marker="o", linestyle="", color=MUTED, markersize=9,
                          label="other checkpoints")]
    fig.legend(handles=handles, loc="lower center", ncol=2, frameon=False, fontsize=9,
               labelcolor=INK2, bbox_to_anchor=(0.5, -0.01))
    fig.suptitle(f"MEXA against downstream results, 9 model states ({pooling} pooling, best "
                 f"layer). Points from one run are a trajectory, not an independent sample.",
                 fontsize=11, color=INK, x=0.02, ha="left")
    fig.tight_layout(rect=(0, 0.06, 1, 0.95))
    fig.savefig(path, dpi=170, facecolor="#fcfcfb")
    plt.close(fig)
    print(f"wrote {path}")


# ---------------------------------------------------------------------------
def write_tables(summary, path, pooling="weighted"):
    """One markdown table per condition: model states ranked by the answer column."""
    lines = ["# MEXA tables", "",
             "Share of 100 parallel sentences whose true counterpart is the mutual nearest",
             "neighbour. Chance is about 0.01. One table per comparison; the final column is",
             "the one that answers it and the rows are sorted by it.", ""]
    for condition in CONDITION_ORDER:
        lines += [f"## {CONDITION_LABEL[condition]} ({pooling} pooling)", "",
                  "| model | training | mean over layers | best layer | **MEXA at best layer** |",
                  "|---|--:|--:|--:|--:|"]
        entries = [(label, s) for (label, cond, pool), s in summary.items()
                   if cond == condition and pool == pooling]
        for label, s in sorted(entries, key=lambda kv: -kv[1]["max"]):
            progress = "reference" if s["model"] == "base" else f"{s['train_fraction']:.0%}"
            lines.append(f"| {label} | {progress} | {s['mean']:.3f} | {s['best_layer']} | "
                         f"**{s['max']:.2f}** |")
        lines.append("")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"wrote {path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--long", required=True, help="mexa_long.csv from mexa_score.py")
    ap.add_argument("--evals", required=True, help="Qomhra2-Paper/evals directory")
    args = ap.parse_args()

    rows = read_long(args.long)
    summary = summarise(rows)
    labels = {k[0] for k in summary}
    print(f"[report] {len(rows)} rows, {len(labels)} model states, "
          f"{len({k[1] for k in summary})} conditions")

    iwslt, cluas = read_downstream(args.evals)
    os.makedirs(args.evals, exist_ok=True)

    write_summary(summary, os.path.join(args.evals, "mexa_summary.csv"))
    write_validation(summary, iwslt, cluas, os.path.join(args.evals, "mexa_validation.csv"))
    write_tables(summary, os.path.join(args.evals, "mexa_tables.md"))
    figure_layer_profile(summary, "weighted",
                         os.path.join(args.evals, "mexa_layer_profile.png"))
    figure_trajectory(summary, "weighted", "max",
                      os.path.join(args.evals, "mexa_trajectory.png"))
    figure_validation(summary, iwslt, cluas, "weighted",
                      os.path.join(args.evals, "mexa_validation_scatter.png"))


if __name__ == "__main__":
    main()
