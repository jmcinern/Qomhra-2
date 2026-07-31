#!/usr/bin/env python3
"""Create review tables and accessible graphs for the four pre-evaluations."""
import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
import numpy as np
import pandas as pd
import seaborn as sns


MODEL_ORDER = ["base", "ab1", "ab2", "ab3", "ab4"]
MODEL_LABEL = {
    "base": "Base",
    "ab1": "AB1 · text",
    "ab2": "AB2 · speech",
    "ab3": "AB3 · mixed",
    "ab4": "AB4 · aligned",
}
MEXA_MODEL = {
    # `base_native` -- NOT `expanded_base` -- is base's MEXA value. expanded_base is
    # base pushed through raw unit ids it has no training on, which understated its
    # cross-modal alignment by roughly an order of magnitude (Irish speech<->text
    # 0.09 vs 0.92). It is deliberately absent from this map so it can never be
    # plotted as "base" again. See FINAL_EVALS_HANDOFF.md caveat 3.
    "base_native": "base",
    "ab1_text": "ab1",
    "ab2_speech": "ab2",
    "ab3_text_speech": "ab3",
    "ab4_text_speech_asr": "ab4",
}
MODEL_COLOUR = {
    "base": "#52514e",
    "ab1": "#2a78d6",
    "ab2": "#d95f02",
    "ab3": "#1b9e77",
    "ab4": "#b77900",
}
INK = "#171717"
GRID = "#deddd8"
BACKGROUND = "#fcfcfb"

MEXA_CONDITIONS = [
    ("speech_ga~text_ga", "GA speech\n↔ GA text"),
    ("speech_en~text_en", "EN speech\n↔ EN text"),
]


def setup_style():
    sns.set_theme(style="whitegrid")
    plt.rcParams.update({
        "figure.facecolor": BACKGROUND,
        "axes.facecolor": BACKGROUND,
        "savefig.facecolor": BACKGROUND,
        "font.size": 10,
        "axes.labelcolor": INK,
        "axes.titlecolor": INK,
        "text.color": INK,
        "grid.color": GRID,
        "grid.linewidth": 0.7,
    })


def pct(value):
    return "—" if pd.isna(value) else f"{float(value):.1f}"


def markdown_table(headers, rows):
    lines = [
        "| " + " | ".join(headers) + " |",
        "|" + "|".join("---" if i == 0 else "---:" for i in range(len(headers))) + "|",
    ]
    lines.extend("| " + " | ".join(map(str, row)) + " |" for row in rows)
    return "\n".join(lines)


def save(fig, out):
    fig.savefig(out, dpi=220, bbox_inches="tight")
    plt.close(fig)
    print(out)


def mexa_summary(path, base_native_path=None):
    """Ablation MEXA rows from `path`, base's row from the native-tower CSV.

    They have to come from different files: ab1-ab4 are measured through unit ids,
    base through its audio tower, and there is no single file holding both because
    there is no single pathway that is correct for both.
    """
    data = pd.read_csv(path)
    if base_native_path is not None and Path(base_native_path).exists():
        data = pd.concat([data, pd.read_csv(base_native_path)], ignore_index=True)
    data["short_model"] = data["model"].map(MEXA_MODEL)
    data = data[data["short_model"].notna()].copy()
    final_rows = []
    for model in MODEL_ORDER:
        part = data[data["short_model"] == model]
        if part.empty:
            continue
        if model == "base":
            final_rows.append(part)
        else:
            final_fraction = part["train_fraction"].max()
            final_rows.append(part[np.isclose(part["train_fraction"], final_fraction)])
    final = pd.concat(final_rows, ignore_index=True)
    rows = []
    for model in MODEL_ORDER:
        for condition, _ in MEXA_CONDITIONS:
            part = final[
                (final["short_model"] == model)
                & (final["condition"] == condition)
            ]
            if part.empty:
                continue
            best_index = part["centered_mexa_at10"].idxmax()
            best = part.loc[best_index]
            rows.append({
                "model": model,
                "condition": condition,
                "score": float(best["centered_mexa_at10"]),
                "best_layer": int(best["layer"]),
            })
    return pd.DataFrame(rows)


def plot_mexa(pooled_path, boundary_path, out_dir, base_native_path=None):
    frames = [
        ("Pooled unit states", mexa_summary(pooled_path, base_native_path)),
        ("ASR-boundary state", mexa_summary(boundary_path, base_native_path)),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(15, 4.8), constrained_layout=True)
    norm = TwoSlopeNorm(vmin=0.0, vcenter=0.0832868, vmax=1.0)
    image = None
    for axis, (title, data) in zip(axes, frames):
        matrix = (
            data.pivot(index="model", columns="condition", values="score")
            .reindex(index=MODEL_ORDER, columns=[c for c, _ in MEXA_CONDITIONS])
        )
        image = sns.heatmap(
            matrix,
            ax=axis,
            cmap="PuOr_r",
            norm=norm,
            cbar=False,
            annot=True,
            fmt=".2f",
            linewidths=0.6,
            linecolor=BACKGROUND,
            xticklabels=[label for _, label in MEXA_CONDITIONS],
            yticklabels=[MODEL_LABEL[m] for m in MODEL_ORDER],
        )
        axis.set_title(title, fontweight="bold")
        axis.set_xlabel("")
        axis.set_ylabel("")
        axis.tick_params(axis="x", rotation=0)
        axis.tick_params(axis="y", rotation=0)
    colourbar = fig.colorbar(image.collections[0], ax=axes, shrink=0.85, pad=0.02)
    colourbar.set_ticks([0.0, 0.0832868, 0.25, 0.5, 0.75, 1.0])
    colourbar.set_ticklabels(["0", "chance\n0.083", "0.25", "0.50", "0.75", "1.00"])
    colourbar.set_label("Centered MEXA@10 · best decoder layer")
    fig.suptitle(
        "MEXA · monolingual mutual top-10 alignment at the final checkpoint",
        fontsize=15,
        fontweight="bold",
    )
    save(fig, out_dir / "01_mexa_at10.png")
    return frames


def plot_fleurs(path, out_dir):
    all_data = pd.read_csv(path)
    data = (
        all_data[all_data["model"].isin(MODEL_ORDER)]
        .drop_duplicates("model", keep="last")
        .set_index("model")
        .reindex(MODEL_ORDER)
    )
    x = np.arange(len(MODEL_ORDER))
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), constrained_layout=True)

    width = 0.36
    for offset, column, label, colour, hatch in [
        (-width / 2, "asr_ga_wer", "Irish ASR", "#2a78d6", ""),
        (width / 2, "asr_en_wer", "English ASR", "#d95f02", "//"),
    ]:
        axes[0].bar(
            x + offset, data[column], width, label=label, color=colour,
            hatch=hatch, edgecolor="white",
        )
    axes[0].set_title("Speech recognition · lower is better", fontweight="bold")
    axes[0].set_ylabel("WER (%)")
    axes[0].set_ylim(0, max(130, np.nanmax(data[["asr_ga_wer", "asr_en_wer"]].values) * 1.08))
    axes[0].legend(frameon=False)

    translation = [
        ("text_ga2en_chrfpp", "GA text → EN", "#2a78d6", ""),
        ("text_en2ga_chrfpp", "EN text → GA", "#d95f02", "//"),
        ("st_ga2en_chrfpp", "GA speech → EN", "#1b9e77", ".."),
        ("st_en2ga_chrfpp", "EN speech → GA", "#b77900", "xx"),
    ]
    width = 0.19
    for index, (column, label, colour, hatch) in enumerate(translation):
        offset = (index - 1.5) * width
        axes[1].bar(
            x + offset, data[column], width, label=label, color=colour,
            hatch=hatch, edgecolor="white",
        )
    axes[1].set_title("Translation · higher is better", fontweight="bold")
    axes[1].set_ylabel("chrF++")
    axes[1].set_ylim(0, max(50, np.nanmax(data[[c for c, _, _, _ in translation]].values) * 1.12))
    axes[1].legend(frameon=False, ncols=2, fontsize=9)

    for axis in axes:
        axis.set_xticks(x, [MODEL_LABEL[m] for m in MODEL_ORDER], rotation=20, ha="right")
        axis.spines[["top", "right"]].set_visible(False)
    fig.suptitle("FLEURS · multilingual task performance (n=34)", fontsize=15, fontweight="bold")
    save(fig, out_dir / "02_fleurs_tasks.png")

    trajectory = all_data[
        all_data["model"].str.fullmatch(r"ab1_(?:10|20|30|40|50|60|70|80|90)pct")
        | all_data["model"].eq("ab1")
    ].copy()
    trajectory["training_pct"] = trajectory["model"].str.extract(r"(\d+)pct")[0]
    trajectory.loc[trajectory["model"].eq("ab1"), "training_pct"] = 100
    trajectory["training_pct"] = trajectory["training_pct"].astype(int)
    trajectory = trajectory.sort_values("training_pct")
    if len(trajectory) == 10:
        fig, axes = plt.subplots(1, 3, figsize=(16, 4.7), constrained_layout=True)
        panels = [
            (
                axes[0],
                [("asr_ga_wer", "Irish ASR", "#2a78d6", "o", "-"),
                 ("asr_en_wer", "English ASR", "#d95f02", "s", "--")],
                "ASR · lower is better",
                "WER (%)",
            ),
            (
                axes[1],
                [("text_ga2en_chrfpp", "GA text → EN", "#2a78d6", "o", "-"),
                 ("text_en2ga_chrfpp", "EN text → GA", "#d95f02", "s", "--")],
                "Text translation",
                "chrF++",
            ),
            (
                axes[2],
                [("st_ga2en_chrfpp", "GA speech → EN", "#1b9e77", "o", "-"),
                 ("st_en2ga_chrfpp", "EN speech → GA", "#b77900", "s", "--")],
                "Speech translation",
                "chrF++",
            ),
        ]
        for axis, series, title, ylabel in panels:
            for column, label, colour, marker, linestyle in series:
                axis.plot(
                    trajectory["training_pct"],
                    trajectory[column],
                    marker=marker,
                    linestyle=linestyle,
                    linewidth=2.2,
                    label=label,
                    color=colour,
                )
            axis.set_title(title, fontweight="bold")
            axis.set_xlabel("Pre-ablation training completed (%)")
            axis.set_ylabel(ylabel)
            axis.set_xticks(range(10, 101, 10))
            axis.legend(frameon=False, fontsize=9)
            axis.spines[["top", "right"]].set_visible(False)
        fig.suptitle("FLEURS · AB1 text-only checkpoint trajectory", fontsize=15, fontweight="bold")
        save(fig, out_dir / "02b_fleurs_ab1_trajectory.png")
    return data, trajectory


def plot_iwslt(path, out_dir):
    data = pd.read_csv(path).set_index("model").reindex(MODEL_ORDER)
    x = np.arange(len(MODEL_ORDER))
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), constrained_layout=True)

    bottoms = np.zeros(len(data))
    route_columns = [
        ("translation_pct", "English / translation", "#2a78d6", ""),
        ("transcription_pct", "Irish / transcription", "#1b9e77", ".."),
        ("other_n", "Other language", "#b77900", "xx"),
        ("empty_n", "Empty", "#b8b6af", "//"),
    ]
    for column, label, colour, hatch in route_columns:
        values = (
            data[column].to_numpy(dtype=float)
            if column.endswith("_pct")
            else 100 * data[column].to_numpy(dtype=float) / data["n"].to_numpy(dtype=float)
        )
        axes[0].bar(
            x, values, bottom=bottoms, label=label, color=colour,
            hatch=hatch, edgecolor="white",
        )
        bottoms += values
    axes[0].set_ylim(0, 100)
    axes[0].set_ylabel("Outputs (%)")
    axes[0].set_title("What did the model emit?", fontweight="bold")
    axes[0].legend(frameon=False, fontsize=9)

    columns = [
        ("original_translation_chrfpp_all", "Translation chrF++ · all", "#2a78d6", ""),
        ("mean_sentence_routed_chrfpp", "Routed chrF++", "#b77900", "//"),
    ]
    width = 0.36
    for index, (column, label, colour, hatch) in enumerate(columns):
        axes[1].bar(
            x + (index - 0.5) * width, data[column], width, label=label,
            color=colour, hatch=hatch, edgecolor="white",
        )
    axes[1].set_ylabel("chrF++")
    score_max = np.nanmax(data[[column for column, _, _, _ in columns]].to_numpy())
    axes[1].set_ylim(0, max(30, score_max * 1.12))
    axes[1].set_title("Task score · higher is better", fontweight="bold")
    axes[1].legend(frameon=False)

    for axis in axes:
        axis.set_xticks(x, [MODEL_LABEL[m] for m in MODEL_ORDER], rotation=20, ha="right")
        axis.spines[["top", "right"]].set_visible(False)
    fig.suptitle("IWSLT · Irish speech response behaviour (n=112)", fontsize=15, fontweight="bold")
    save(fig, out_dir / "03_iwslt_routes_and_scores.png")
    return data


def plot_cluas(path, out_dir):
    data = pd.read_csv(path)
    pivot = data.pivot(index="model", columns="condition", values="official_percent").reindex(MODEL_ORDER)
    gain = (
        data.drop_duplicates("model").set_index("model")["listening_gain_marks"]
        .reindex(MODEL_ORDER)
    )
    x = np.arange(len(MODEL_ORDER))
    fig, axes = plt.subplots(
        1, 2, figsize=(14, 5), constrained_layout=True,
        gridspec_kw={"width_ratios": [2.2, 1]},
    )
    conditions = [
        ("just_transcript", "Transcript", "#2a78d6", ""),
        ("no_context", "Blind", "#b8b6af", "//"),
        ("just_audio", "Audio", "#d95f02", ".."),
    ]
    width = 0.25
    for index, (column, label, colour, hatch) in enumerate(conditions):
        axes[0].bar(
            x + (index - 1) * width, pivot[column], width, label=label,
            color=colour, hatch=hatch, edgecolor="white",
        )
    axes[0].set_ylabel("Marks earned (%)")
    axes[0].set_ylim(0, 100)
    axes[0].set_title("Knowledge, guessing, and listening", fontweight="bold")
    axes[0].legend(frameon=False)

    colours = ["#1b9e77" if value > 0 else "#b8b6af" if value == 0 else "#d95f02" for value in gain]
    axes[1].bar(x, gain, color=colours)
    axes[1].axhline(0, color=INK, linewidth=0.9)
    axes[1].set_ylabel("Audio − blind marks")
    axes[1].set_title("Listening gain", fontweight="bold")
    axes[1].set_ylim(min(-3, gain.min() - 1), max(10, gain.max() + 1))
    for index, value in enumerate(gain):
        axes[1].text(index, value + (0.3 if value >= 0 else -0.45), f"{value:+.0f}", ha="center")

    for axis in axes:
        axis.set_xticks(x, [MODEL_LABEL[m] for m in MODEL_ORDER], rotation=20, ha="right")
        axis.spines[["top", "right"]].set_visible(False)
    fig.suptitle("CLUAS · listening comprehension (33 questions, 74 marks)", fontsize=15, fontweight="bold")
    save(fig, out_dir / "04_cluas_marks.png")
    return data


def write_tables(out, mexa_frames, fleurs, fleurs_trajectory, iwslt, cluas):
    lines = [
        "# Pre-ablation evaluation tables",
        "",
        "Raw model outputs are not repaired. `cap` columns report the percentage of "
        "examples that reached `max_new_tokens`; these cases require caution because "
        "some otherwise valid answers are visibly truncated.",
        "",
        "## 1. MEXA",
        "",
        "Centered mutual top-10 retrieval; chance is 0.0833. Values below are the "
        "best decoder-layer score at the final checkpoint, with that layer in parentheses. "
        "The current centered-MEXA@10 run contains the two monolingual speech–text pairs; "
        "the four cross-lingual pairs have not been scored with this metric yet.",
    ]
    for name, frame in mexa_frames:
        table = frame.copy()
        table["cell"] = table.apply(lambda r: f"{r.score:.3f} (L{r.best_layer})", axis=1)
        pivot = table.pivot(index="model", columns="condition", values="cell").reindex(MODEL_ORDER)
        lines.extend(["", f"### {name}", ""])
        lines.append(markdown_table(
            ["Model"] + [label.replace("\n", " ") for _, label in MEXA_CONDITIONS],
            [[MODEL_LABEL[m]] + [pivot.loc[m, c] if c in pivot else "—" for c, _ in MEXA_CONDITIONS]
             for m in MODEL_ORDER],
        ))

    lines.extend(["", "## 2. FLEURS", ""])
    frows = []
    for model in MODEL_ORDER:
        row = fleurs.loc[model]
        frows.append([
            MODEL_LABEL[model], pct(row.asr_ga_wer), pct(row.asr_en_wer),
            pct(row.text_ga2en_chrfpp), pct(row.text_en2ga_chrfpp),
            pct(row.st_ga2en_chrfpp), pct(row.st_en2ga_chrfpp),
        ])
    lines.append(markdown_table(
        ["Model", "GA ASR WER↓", "EN ASR WER↓", "GA txt→EN chrF++↑",
         "EN txt→GA chrF++↑", "GA sp→EN chrF++↑", "EN sp→GA chrF++↑"],
        frows,
    ))
    if len(fleurs_trajectory) == 10:
        lines.extend(["", "### FLEURS AB1 checkpoint trajectory", ""])
        lines.append(markdown_table(
            ["Training %", "GA ASR WER↓", "EN ASR WER↓", "GA txt→EN chrF++↑",
             "EN txt→GA chrF++↑", "GA sp→EN chrF++↑", "EN sp→GA chrF++↑"],
            [[int(row.training_pct), pct(row.asr_ga_wer), pct(row.asr_en_wer),
              pct(row.text_ga2en_chrfpp), pct(row.text_en2ga_chrfpp),
              pct(row.st_ga2en_chrfpp), pct(row.st_en2ga_chrfpp)]
             for row in fleurs_trajectory.itertuples()],
        ))
    lines.extend(["", "### FLEURS max-token cap rate (%)", ""])
    lines.append(markdown_table(
        ["Model", "GA ASR", "EN ASR", "GA txt→EN", "EN txt→GA", "GA sp→EN", "EN sp→GA"],
        [[MODEL_LABEL[m]] + [
            pct(100 * fleurs.loc[m, c] / fleurs.loc[m, "n"])
            for c in ("asr_ga_cap_hits", "asr_en_cap_hits", "text_ga2en_cap_hits",
                      "text_en2ga_cap_hits", "st_ga2en_cap_hits", "st_en2ga_cap_hits")
        ] for m in MODEL_ORDER],
    ))

    lines.extend(["", "## 3. IWSLT", ""])
    lines.append(markdown_table(
        ["Model", "Translation outputs %", "Transcription outputs %", "Other", "Empty",
         "Translation chrF++", "Routed chrF++", "Cap %"],
        [[MODEL_LABEL[m], pct(iwslt.loc[m, "translation_pct"]),
          pct(iwslt.loc[m, "transcription_pct"]), int(iwslt.loc[m, "other_n"]),
          int(iwslt.loc[m, "empty_n"]), pct(iwslt.loc[m, "original_translation_chrfpp_all"]),
          pct(iwslt.loc[m, "mean_sentence_routed_chrfpp"]),
          pct(100 * iwslt.loc[m, "cap_n"] / iwslt.loc[m, "n"])]
         for m in MODEL_ORDER],
    ))

    lines.extend(["", "## 4. CLUAS", ""])
    cp = cluas.pivot(index="model", columns="condition", values="official_percent").reindex(MODEL_ORDER)
    cm = cluas.pivot(index="model", columns="condition", values="official_marks").reindex(MODEL_ORDER)
    cg = cluas.drop_duplicates("model").set_index("model")["listening_gain_marks"].reindex(MODEL_ORDER)
    cc = cluas.pivot(index="model", columns="condition", values="cap_n").reindex(MODEL_ORDER)
    cq = cluas.pivot(index="model", columns="condition", values="questions").reindex(MODEL_ORDER)
    lines.append(markdown_table(
        ["Model", "Transcript", "Blind", "Audio", "Listening gain", "Caps: transcript/blind/audio"],
        [[MODEL_LABEL[m], f"{cm.loc[m, 'just_transcript']:.0f}/74 ({cp.loc[m, 'just_transcript']:.1f}%)",
          f"{cm.loc[m, 'no_context']:.0f}/74 ({cp.loc[m, 'no_context']:.1f}%)",
          f"{cm.loc[m, 'just_audio']:.0f}/74 ({cp.loc[m, 'just_audio']:.1f}%)",
          f"{cg.loc[m]:+.0f}",
          "/".join(pct(100 * cc.loc[m, c] / cq.loc[m, c])
                   for c in ("just_transcript", "no_context", "just_audio"))]
         for m in MODEL_ORDER],
    ))
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(out)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--review-dir", default="ablations/eval/output/review_csv")
    parser.add_argument("--mexa-dir", default="Qomhra2-Paper/evals/discrete_mexa/monolingual")
    parser.add_argument("--out-dir", default="ablations/eval/output/review_visualisations")
    args = parser.parse_args()
    review = Path(args.review_dir)
    mexa = Path(args.mexa_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    setup_style()

    mexa_frames = plot_mexa(
        mexa / "pooled_units.csv", mexa / "asr_boundary.csv", out_dir,
        base_native_path=review / "mexa_base_native.csv",
    )
    fleurs, fleurs_trajectory = plot_fleurs(review / "fleurs_numbers.csv", out_dir)
    iwslt = plot_iwslt(review / "iwslt_numbers.csv", out_dir)
    cluas = plot_cluas(review / "cluas_numbers.csv", out_dir)
    write_tables(
        out_dir / "evaluation_tables.md",
        mexa_frames,
        fleurs,
        fleurs_trajectory,
        iwslt,
        cluas,
    )


if __name__ == "__main__":
    main()
