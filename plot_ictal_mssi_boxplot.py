#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Ictal mSSI boxplot analysis for AST-Net_GCN predictions.

Definition
----------
mSSI (model-derived Seizure Severity Index) = P(Ictal) for every segment whose
experimental ground-truth phase is Ictal. A higher mSSI means that the EEG
segment is more similar to the model's learned prototypical ictal pattern.

Important interpretation safeguards
-----------------------------------
1. All *ground-truth* Ictal segments are retained. Do NOT filter on pred_class,
   because excluding low-P(Ictal) segments would bias treatment comparisons.
2. Sponge is displayed as a mechanism-inhibition control, never annotated as a
   therapeutic effect.
3. Prefer --unit animal for inferential statistics. Segment-level results are
   descriptive because segments from the same animal are correlated.
4. This script visualizes a model-derived EEG phenotype, not direct seizure
   severity, SRS, FRs, or histopathology.

Input CSV must contain at least:
  group, true_phase, prob_Ictal
Optional columns for animal-level aggregation:
  animal_id, subject_id, rat_id, animal, subject, file_path, path, seg_id

Examples
--------
# Recommended: infer animal_id from per-segment identifiers/paths when possible
python plot_ictal_mssi_boxplot.py \
  --input /path/to/per_segment_predictions.csv \
  --output-dir /path/to/ictal_mssi \
  --unit animal

# If an animal identifier cannot be recovered, use this descriptive fallback
python plot_ictal_mssi_boxplot.py \
  --input /path/to/per_segment_predictions.csv \
  --output-dir /path/to/ictal_mssi \
  --unit segment
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Iterable, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats


GROUP_ORDER = ["pilo组", "VPA组", "mir过表达组", "sponges组"]
GROUP_LABELS = {
    "pilo组": "PILO\nuntreated model",
    "VPA组": "VPA\npositive control",
    "mir过表达组": "miR-23b-3p\noverexpression",
    "sponges组": "miR-23b-3p sponge\nmechanistic control",
}
GROUP_COLORS = {
    "pilo组": "#4D4D4D",
    "VPA组": "#4E79A7",
    "mir过表达组": "#59A14F",
    "sponges组": "#A5A5A5",
}

DIRECT_ANIMAL_COLUMNS = ["animal_id", "subject_id", "rat_id", "animal", "subject"]
PATH_COLUMNS = ["file_path", "path", "seg_id"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create ictal mSSI boxplots from AST-Net_GCN per-segment predictions."
    )
    parser.add_argument("--input", required=True, type=Path,
                        help="Path to per_segment_predictions.csv")
    parser.add_argument("--output-dir", required=True, type=Path,
                        help="Directory for figure and CSV outputs")
    parser.add_argument("--unit", choices=["animal", "segment"], default="animal",
                        help="Statistical/plotting unit. Animal is recommended; segment is descriptive only.")
    parser.add_argument("--animal-regex", default=None,
                        help=("Optional regex containing one capture group for animal ID when it must be "
                              "extracted from a path/seg_id, e.g. '(?:rat|animal)[_-]?(\\d+)'."))
    parser.add_argument("--summary", choices=["mean", "median"], default="mean",
                        help="Within-animal summary across Ictal segments (default: mean).")
    parser.add_argument("--seed", type=int, default=2026,
                        help="Random seed used for visual jitter.")
    return parser.parse_args()


def find_animal_id(df: pd.DataFrame, regex: Optional[str]) -> tuple[pd.Series, str]:
    """Return one animal ID per row and document how it was obtained."""
    for col in DIRECT_ANIMAL_COLUMNS:
        if col in df.columns and df[col].notna().any():
            ids = df[col].astype(str).str.strip()
            valid = ids.ne("") & ids.ne("nan")
            if valid.all():
                return ids, f"direct column: {col}"

    source_col = next((col for col in PATH_COLUMNS if col in df.columns), None)
    if source_col is None:
        raise ValueError(
            "Animal-level analysis requested, but no animal_id/subject_id/rat_id/animal/subject "
            "column or path-like column (file_path/path/seg_id) is available. "
            "Add an animal_id column, provide --animal-regex, or use --unit segment."
        )

    source = df[source_col].astype(str)
    if regex:
        extracted = source.str.extract(regex, expand=False)
        if isinstance(extracted, pd.DataFrame):
            extracted = extracted.iloc[:, 0]
        if extracted.isna().any():
            n_missing = int(extracted.isna().sum())
            raise ValueError(
                f"--animal-regex did not identify an animal ID in {n_missing} rows of {source_col}."
            )
        return extracted.astype(str), f"regex from {source_col}: {regex}"

    # Conservative automatic patterns. These only activate if every row resolves.
    patterns = [
        r"(?i)(?:^|[/_\\-])(rat|animal|subject|mouse)[_\\-]?(\d+)(?:$|[/_\\-.])",
        r"(?i)(?:^|[/_\\-])(r\d+)(?:$|[/_\\-.])",
    ]
    for pattern in patterns:
        extracted = source.str.extract(pattern, expand=True)
        if extracted.shape[1] == 2:
            ids = extracted[0].fillna("") + "_" + extracted[1].fillna("")
        else:
            ids = extracted.iloc[:, 0]
        valid = ids.notna() & ids.astype(str).str.strip().ne("") & ids.astype(str).ne("_")
        if valid.all():
            return ids.astype(str), f"auto-regex from {source_col}: {pattern}"

    raise ValueError(
        f"Could not reliably infer animal IDs from {source_col}. Add an animal_id column or supply "
        "--animal-regex. The script will not silently treat segments as independent animals."
    )


def star_label(p: float) -> str:
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return "ns"


def paired_comparisons(summary: pd.DataFrame) -> pd.DataFrame:
    """Compare each experimental group with PILO; sponge is reported but not efficacy-labelled."""
    rows = []
    pilo = summary.loc[summary["group"] == "pilo组", "mSSI"].dropna().to_numpy()
    if len(pilo) == 0:
        return pd.DataFrame()

    for group in GROUP_ORDER[1:]:
        values = summary.loc[summary["group"] == group, "mSSI"].dropna().to_numpy()
        if len(values) == 0:
            continue
        u, p = stats.mannwhitneyu(pilo, values, alternative="two-sided")
        rows.append({
            "comparison": f"{group} vs pilo组",
            "group": group,
            "control": "pilo组",
            "n_group": len(values),
            "n_pilo": len(pilo),
            "median_group": float(np.median(values)),
            "median_pilo": float(np.median(pilo)),
            "mean_group": float(np.mean(values)),
            "mean_pilo": float(np.mean(pilo)),
            "delta_mean_group_minus_pilo": float(np.mean(values) - np.mean(pilo)),
            "mannwhitney_U": float(u),
            "p_value_unadjusted": float(p),
            "significance_unadjusted": star_label(float(p)),
            "interpretation": (
                "mechanistic-control comparison; do not interpret as therapeutic efficacy"
                if group == "sponges组"
                else "lower mSSI is EEG-pattern attenuation relative to PILO"
            ),
        })
    return pd.DataFrame(rows)


def bh_adjust(p_values: Iterable[float]) -> np.ndarray:
    """Benjamini-Hochberg FDR correction without an additional dependency."""
    p = np.asarray(list(p_values), dtype=float)
    if p.size == 0:
        return p
    order = np.argsort(p)
    ranked = p[order]
    adjusted = ranked * len(p) / np.arange(1, len(p) + 1)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    adjusted = np.clip(adjusted, 0, 1)
    out = np.empty_like(adjusted)
    out[order] = adjusted
    return out


def annotate_bracket(ax: plt.Axes, x1: float, x2: float, y: float, h: float, text: str) -> None:
    ax.plot([x1, x1, x2, x2], [y, y + h, y + h, y], color="#333333", linewidth=1.0)
    ax.text((x1 + x2) / 2, y + h + 0.008, text, ha="center", va="bottom", fontsize=9)


def make_boxplot(summary: pd.DataFrame, comparisons: pd.DataFrame, output: Path,
                 unit: str, seed: int) -> None:
    rng = np.random.default_rng(seed)
    available = [g for g in GROUP_ORDER if g in set(summary["group"])]
    values = [summary.loc[summary["group"] == g, "mSSI"].dropna().to_numpy() for g in available]

    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 11,
        "axes.spines.top": False,
        "axes.spines.right": False,
    })
    fig, ax = plt.subplots(figsize=(9.0, 6.2), constrained_layout=True)
    positions = np.arange(1, len(available) + 1)
    box = ax.boxplot(
        values, positions=positions, widths=0.58, patch_artist=True,
        showfliers=False,
        medianprops={"color": "#1A1A1A", "linewidth": 1.8},
        whiskerprops={"color": "#555555", "linewidth": 1.1},
        capprops={"color": "#555555", "linewidth": 1.1},
        boxprops={"edgecolor": "#444444", "linewidth": 1.1},
    )
    for patch, group in zip(box["boxes"], available):
        patch.set_facecolor(GROUP_COLORS[group])
        patch.set_alpha(0.68)

    for pos, group, group_values in zip(positions, available, values):
        jitter = rng.normal(loc=0.0, scale=0.055, size=len(group_values))
        ax.scatter(np.full(len(group_values), pos) + jitter, group_values,
                   s=28 if unit == "animal" else 6,
                   color=GROUP_COLORS[group], edgecolors="white" if unit == "animal" else "none",
                   linewidths=0.35, alpha=0.78 if unit == "animal" else 0.12,
                   zorder=3, rasterized=(unit == "segment"))

    ax.set_ylim(-0.02, 1.17)
    ax.set_yticks(np.arange(0, 1.01, 0.2))
    ax.set_ylabel("Model-derived seizure severity index\n(mSSI = P(Ictal))")
    ax.set_xticks(positions)
    ax.set_xticklabels([GROUP_LABELS[g] for g in available], fontsize=10)
    ax.set_title("Ictal EEG phenotype across intervention groups", fontsize=14, weight="bold", pad=16)
    subtitle = (
        "Higher mSSI = EEG more similar to the model's prototypical ictal pattern\n"
        f"Analysis unit: {'animal-level ' + 'mean' if unit == 'animal' else 'Ictal segment'}; all ground-truth Ictal samples retained"
    )
    ax.text(0.5, 1.01, subtitle, transform=ax.transAxes, ha="center", va="bottom", fontsize=9, color="#444444")
    ax.grid(axis="y", color="#D9D9D9", linewidth=0.7, alpha=0.7)
    ax.set_axisbelow(True)

    for pos, group, group_values in zip(positions, available, values):
        ax.text(pos, -0.12, f"n = {len(group_values)}", ha="center", va="top", fontsize=9,
                transform=ax.get_xaxis_transform())
        ax.text(pos, 1.035, f"median = {np.median(group_values):.3f}", ha="center", va="bottom", fontsize=8,
                color="#333333")

    # Show only the planned efficacy comparisons (VPA and miR); sponge remains visible but unbracketed.
    if not comparisons.empty:
        planned = comparisons.loc[comparisons["group"].isin(["VPA组", "mir过表达组"])].copy()
        y = 1.045
        for _, row in planned.iterrows():
            group = row["group"]
            if group not in available:
                continue
            x2 = int(available.index(group)) + 1
            annotate_bracket(ax, 1, x2, y, 0.015, f"vs PILO: {star_label(row['p_value_fdr_bh'])}")
            y += 0.065

    fig.text(
        0.5, -0.02,
        "Sponge is a miR-23b-3p inhibition (mechanistic) control; its distribution is displayed without a therapeutic-efficacy claim.",
        ha="center", va="top", fontsize=8.5, color="#555555",
    )
    fig.savefig(output.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(output.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def write_caption(output: Path, unit: str, id_source: str) -> None:
    caption = f"""# Suggested figure caption\n\n**Ictal model-derived seizure severity index (mSSI) across intervention groups.**\n\nFor all EEG segments labelled experimentally as Ictal, mSSI was defined as the model's Ictal softmax probability, mSSI = P(Ictal). Higher values indicate EEG patterns more similar to the model's learned prototypical ictal state. Boxplots show the distribution of {('animal-level mean mSSI values' if unit == 'animal' else 'individual Ictal segments')}; center lines indicate medians, boxes indicate interquartile ranges, and whiskers extend to 1.5 × IQR. PILO is the untreated epilepsy-model control, VPA is a positive antiseizure control, miR-23b-3p overexpression is the candidate intervention, and sponge is a miR-23b-3p inhibition mechanistic control. The sponge distribution is shown for mechanistic context and is not interpreted as therapeutic efficacy. Animal-ID source: {id_source}. These are model-derived EEG phenotype results and do not replace behavioral seizure, SRS, FRs, or histopathology endpoints.\n"""
    output.write_text(caption, encoding="utf-8")


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if not args.input.exists():
        raise FileNotFoundError(f"Input file does not exist: {args.input}")

    raw = pd.read_csv(args.input)
    required = {"group", "true_phase", "prob_Ictal"}
    missing = required.difference(raw.columns)
    if missing:
        raise ValueError(f"Input CSV lacks required columns: {sorted(missing)}")

    ictal = raw.loc[
        (raw["true_phase"].astype(str).str.strip() == "Ictal")
        & raw["group"].isin(GROUP_ORDER)
    ].copy()
    if ictal.empty:
        raise ValueError("No Ictal rows from the expected four groups were found.")
    ictal = ictal.rename(columns={"prob_Ictal": "mSSI"})
    ictal["mSSI"] = pd.to_numeric(ictal["mSSI"], errors="coerce")
    ictal = ictal.dropna(subset=["mSSI"])
    if ((ictal["mSSI"] < 0) | (ictal["mSSI"] > 1)).any():
        raise ValueError("prob_Ictal/mSSI contains values outside [0, 1].")

    if args.unit == "animal":
        animal_ids, id_source = find_animal_id(ictal, args.animal_regex)
        ictal["animal_id"] = animal_ids
        agg_func = "mean" if args.summary == "mean" else "median"
        summary = (
            ictal.groupby(["group", "animal_id"], as_index=False)
            .agg(mSSI=("mSSI", agg_func), n_ictal_segments=("mSSI", "size"))
        )
        unit_note = f"animal-level {agg_func}; {id_source}"
    else:
        summary = ictal[["group", "mSSI"]].copy()
        summary["animal_id"] = "segment-level descriptive analysis"
        summary["n_ictal_segments"] = 1
        id_source = "not used (segment-level descriptive analysis)"
        unit_note = id_source

    summary["group"] = pd.Categorical(summary["group"], GROUP_ORDER, ordered=True)
    summary = summary.sort_values(["group", "animal_id"]).reset_index(drop=True)
    group_summary = (
        summary.groupby("group", observed=False)["mSSI"]
        .agg(n="count", mean="mean", median="median", std="std", q1=lambda x: x.quantile(0.25), q3=lambda x: x.quantile(0.75))
        .reset_index()
    )
    comparisons = paired_comparisons(summary)
    if not comparisons.empty:
        comparisons["p_value_fdr_bh"] = bh_adjust(comparisons["p_value_unadjusted"])
        comparisons["significance_fdr_bh"] = comparisons["p_value_fdr_bh"].map(star_label)

    ictal.to_csv(args.output_dir / "ictal_mssi_per_segment.csv", index=False, encoding="utf-8-sig")
    summary.to_csv(args.output_dir / f"ictal_mssi_{args.unit}_summary.csv", index=False, encoding="utf-8-sig")
    group_summary.to_csv(args.output_dir / f"ictal_mssi_{args.unit}_group_summary.csv", index=False, encoding="utf-8-sig")
    comparisons.to_csv(args.output_dir / f"ictal_mssi_{args.unit}_vs_pilo_stats.csv", index=False, encoding="utf-8-sig")
    make_boxplot(summary, comparisons, args.output_dir / f"ictal_mssi_{args.unit}_boxplot", args.unit, args.seed)
    write_caption(args.output_dir / "ictal_mssi_figure_caption.md", args.unit, id_source)

    print("Ictal mSSI analysis complete")
    print(f"Input rows retained: {len(ictal)} true-Ictal segments")
    print(f"Analysis unit: {unit_note}")
    print(f"Output directory: {args.output_dir}")
    print(group_summary.to_string(index=False))
    if not comparisons.empty:
        print("\nPILO comparisons (FDR-BH adjusted):")
        print(comparisons[["comparison", "delta_mean_group_minus_pilo", "p_value_fdr_bh", "significance_fdr_bh", "interpretation"]].to_string(index=False))


if __name__ == "__main__":
    main()
