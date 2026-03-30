"""
Calculate radiomics statistics and IoU scores from explanation NPZ files.

Based on statystyki/calculate_image_stats* notebooks.

Usage:
    uv run python calculate_stats.py \
        --npz_dir statystyki/explanations2 \
        --csv_dir output \
        --output_dir results
"""

import argparse
import os
import re
import sys
from pathlib import Path

MODEL_NAME_MAP = {
    "your-entity_ecg-xai-models_model-AAAA_v0": "Model 2",
    "your-entity_ecg-xai-models_model-BBBB_v0": "Model 5",
    "your-entity_ecg-xai-models_model-CCCC_v0": "Model 1",
}

AUGMENTATION_DISPLAY_NAMES = {
    "all": "Discolored",
    "full": "Discolored",
    "no_header": "No metadata",
    "no_header_images": "No metadata",
    "text": "Handwriting",
    "text_images": "Handwriting",
    "wrinkles": "Wrinkles",
    "wrinkles_images": "Wrinkles",
    "dewarped": "Dewarped",
    "dewarped_images": "Dewarped",
}

DEFAULT_NPZ_DIR = "statystyki/explanations2"


def normalize_model_name(name: str) -> str:
    """Normalize model name variants (e.g. ':v0' → '_v0') then apply display mapping."""
    normalized = name.replace(":v", "_v")
    return MODEL_NAME_MAP.get(normalized, normalized)


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.manifold import TSNE
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------

def validate_inputs(args) -> None:
    errors = []

    npz_dir = Path(args.npz_dir)
    if not npz_dir.exists():
        errors.append(f"--npz_dir does not exist: {npz_dir}")
    elif not npz_dir.is_dir():
        errors.append(f"--npz_dir is not a directory: {npz_dir}")
    else:
        # Expect at least one subdirectory level (model/mode/...)
        subdirs = [p for p in npz_dir.iterdir() if p.is_dir()]
        if not subdirs:
            errors.append(f"--npz_dir has no subdirectories (expected model/mode/sample.npz layout): {npz_dir}")
        else:
            has_npz = any(
                f.suffix == ".npz"
                for sub in subdirs
                for sub2 in sub.iterdir() if sub2.is_dir()
                for f in sub2.iterdir() if f.is_file()
            )
            if not has_npz:
                errors.append(
                    f"--npz_dir contains no .npz files at depth model/mode/*.npz: {npz_dir}"
                )

    if args.csv_dir is not None:
        csv_dir = Path(args.csv_dir)
        if not csv_dir.exists():
            errors.append(f"--csv_dir does not exist: {csv_dir}")
        elif not csv_dir.is_dir():
            errors.append(f"--csv_dir is not a directory: {csv_dir}")
        else:
            pattern = re.compile(r"^\d+-\d+.*\.csv$")
            csv_files = [f for f in csv_dir.iterdir() if pattern.match(f.name)]
            if not csv_files:
                errors.append(
                    f"--csv_dir contains no CSV files matching pattern <from>-<to>*.csv: {csv_dir}"
                )

    if not 0.0 < args.pca_variance <= 1.0:
        errors.append(f"--pca_variance must be in (0, 1], got: {args.pca_variance}")

    if args.tile_size < 1:
        errors.append(f"--tile_size must be >= 1, got: {args.tile_size}")

    if args.max_per_group < 1:
        errors.append(f"--max_per_group must be >= 1, got: {args.max_per_group}")

    if errors:
        print("Input validation failed:")
        for e in errors:
            print(f"  ERROR: {e}")
        sys.exit(1)


# ---------------------------------------------------------------------------
# NPZ helpers
# ---------------------------------------------------------------------------

def extract_npz_paths(root_dir: str, max_per_group: int = 100) -> pd.DataFrame:
    data = []
    for dirpath, _, filenames in os.walk(root_dir):
        for file in filenames:
            if file.endswith(".npz"):
                full_path = os.path.join(dirpath, file)
                parts = full_path.split(os.sep)
                if len(parts) >= 3:
                    data.append({
                        "full_path": full_path,
                        "model_name": parts[-3],
                        "mode": parts[-2],
                        "npz_name": parts[-1],
                    })
    df = pd.DataFrame(data)
    if df.empty:
        return df
    df["model_name"] = df["model_name"].map(normalize_model_name)
    df = df.groupby(["model_name", "mode"]).head(max_per_group).reset_index(drop=True)
    return df


def split_into_tiles(image: np.ndarray, tile_size: int = 30,
                     legacy: bool = False) -> np.ndarray:
    if legacy:
        # Original notebook implementation (calculate_image_stats*.ipynb):
        # hardcodes w=512, h=1024 regardless of actual image shape.
        # On real data (512, 1024, 3) this means:
        #   - rows loop 0..1020 but image only has 512 → tiles 18-33 are empty (False)
        #   - cols loop 0..510, covering only the left half of the image
        w, h = 512, 1024
    else:
        # Fixed: use actual spatial dimensions
        h, w = image.shape[0], image.shape[1]
    h = (h // tile_size) * tile_size
    w = (w // tile_size) * tile_size
    tiles = np.zeros((h // tile_size, w // tile_size), dtype=int)
    for i in range(0, h, tile_size):
        for j in range(0, w, tile_size):
            tile = image[i:i + tile_size, j:j + tile_size]
            tiles[i // tile_size, j // tile_size] = np.any(tile)
    return tiles


def random_iou_null(k_base: int, k_aug: int, n: int) -> float:
    """
    Expected IoU for two randomly placed binary masks of sizes k_base and k_aug
    over n tiles (hypergeometric expectation).

    E[intersection] = k_base * k_aug / n
    E[union]        = k_base + k_aug - k_base * k_aug / n
    """
    if n == 0 or k_base == 0 or k_aug == 0:
        return 0.0
    e_inter = k_base * k_aug / n
    e_union = k_base + k_aug - e_inter
    return e_inter / e_union if e_union > 0 else 0.0


def calculate_iou(df: pd.DataFrame, baseline_mode: str = "clear_images",
                  legacy_tiles: bool = False) -> pd.DataFrame:
    grouped = df.groupby(["model_name", "npz_name"])
    results = []

    for (model_name, npz_name), group in grouped:
        baseline_mask = None
        other_masks = []

        for _, row in group.iterrows():
            data = np.load(row["full_path"], allow_pickle=True)
            if "lime" not in data:
                continue
            tiled = split_into_tiles(data["lime"], legacy=legacy_tiles)
            if row["mode"] == baseline_mode:
                baseline_mask = tiled
            else:
                other_masks.append((row["mode"], tiled))

        iou_dict: dict = {}
        if baseline_mask is not None:
            n_tiles = baseline_mask.size
            k_base = int(baseline_mask.sum())
            for mode, mask in other_masks:
                intersection = np.logical_and(baseline_mask, mask).sum()
                union = np.logical_or(baseline_mask, mask).sum()
                iou_dict[f"iou_{mode}"] = intersection / union if union > 0 else 0.0
                k_aug = int(mask.sum())
                iou_dict[f"null_iou_{mode}"] = random_iou_null(k_base, k_aug, n_tiles)
        else:
            iou_dict["error"] = f"No {baseline_mode} mask found"

        results.append({"model_name": model_name, "npz_name": npz_name, **iou_dict})

    return pd.DataFrame(results)


def compute_iou_significance(iou_df: pd.DataFrame, iou_cols: list) -> pd.DataFrame:
    """
    Kruskal-Wallis H-test: does IoU differ significantly across augmentation modes?

    This is a non-parametric one-way ANOVA run per model.  Samples within each
    augmentation group are independent (different ECG/augmentation combinations).
    H₀: the IoU distributions are identical across all augmentation modes.

    Returns: model | n_augmentations | H_statistic | p_value | significant_95 | significant_99
    """
    rows = []
    for model_name, group in iou_df.groupby("model_name"):
        arrays = [group[col].dropna().values for col in iou_cols]
        arrays = [a for a in arrays if len(a) >= 2]
        if len(arrays) < 2:
            continue
        stat, p = stats.kruskal(*arrays)
        rows.append({
            "model": model_name,
            "test": "Kruskal-Wallis",
            "n_augmentations": len(arrays),
            "H_statistic": round(float(stat), 4),
            "p_value": round(float(p), 4),
            "significant_95": p < 0.05,
            "significant_99": p < 0.01,
        })
    return pd.DataFrame(rows)


def compute_pairwise_augmentation_significance(
    iou_df: pd.DataFrame, iou_cols: list
) -> pd.DataFrame:
    """
    Pairwise Wilcoxon signed-rank tests between every pair of augmentation IoU
    distributions, per model.  Paired by sample (same ECG across augmentations).
    Applies Holm-Bonferroni correction for the number of pairs within each model.

    Returns: model | aug_a | aug_b | W_statistic | p_value | p_corrected | significant_05 | significant_01
    """
    from itertools import combinations

    rows = []
    for model_name, group in iou_df.groupby("model_name"):
        pair_results = []
        for col_a, col_b in combinations(iou_cols, 2):
            paired = group[[col_a, col_b]].dropna()
            if len(paired) < 2:
                continue
            a = paired[col_a].values
            b = paired[col_b].values
            stat, p = stats.wilcoxon(a, b, alternative="two-sided")
            label_a = AUGMENTATION_DISPLAY_NAMES.get(col_a.replace("iou_", ""), col_a.replace("iou_", ""))
            label_b = AUGMENTATION_DISPLAY_NAMES.get(col_b.replace("iou_", ""), col_b.replace("iou_", ""))
            pair_results.append({
                "model": model_name,
                "aug_a": label_a,
                "aug_b": label_b,
                "W_statistic": round(float(stat), 2),
                "p_value": float(p),
            })
        # Holm-Bonferroni correction
        pair_results.sort(key=lambda r: r["p_value"])
        n_pairs = len(pair_results)
        for rank_i, r in enumerate(pair_results):
            r["p_corrected"] = min(r["p_value"] * (n_pairs - rank_i), 1.0)
        # enforce monotonicity: corrected p can't decrease as rank increases
        for i in range(1, len(pair_results)):
            pair_results[i]["p_corrected"] = max(
                pair_results[i]["p_corrected"], pair_results[i - 1]["p_corrected"]
            )
        for r in pair_results:
            r["significant_05"] = r["p_corrected"] < 0.05
            r["significant_01"] = r["p_corrected"] < 0.01
        rows.extend(pair_results)
    return pd.DataFrame(rows)


def generate_pairwise_latex_table(
    pairwise_df: pd.DataFrame, out: Path
) -> None:
    """
    LaTeX table of pairwise Holm-Bonferroni-corrected p-values, one sub-table
    per model.  Lower triangle shows corrected p-values.
    """
    lines = []
    for model_name, group in pairwise_df.groupby("model"):
        augs = sorted(set(group["aug_a"]) | set(group["aug_b"]))
        n = len(augs)
        lookup = {}
        for _, row in group.iterrows():
            lookup[(row["aug_a"], row["aug_b"])] = row["p_corrected"]
            lookup[(row["aug_b"], row["aug_a"])] = row["p_corrected"]

        lines.append(r"% " + model_name)
        lines.append(r"\begin{tabular}{r" + "c" * n + "}")
        lines.append(r"\toprule")
        header = [r"\textbf{" + model_name + "}"] + [r"\textbf{" + a + "}" for a in augs]
        lines.append(" & ".join(header) + r" \\")
        lines.append(r"\midrule")
        for a in augs:
            cells = [r"\textbf{" + a + "}"]
            for b in augs:
                if a == b:
                    cells.append("---")
                else:
                    p = lookup.get((a, b), float("nan"))
                    if np.isnan(p):
                        cells.append("---")
                    elif p < 0.001:
                        exp = int(np.floor(np.log10(p))) if p > 0 else 0
                        mantissa = p / 10 ** exp
                        cells.append(rf"${mantissa:.1f}\times10^{{{exp}}}$")
                    else:
                        cells.append(f"${p:.4f}$")
            lines.append(" & ".join(cells) + r" \\")
        lines.append(r"\bottomrule")
        lines.append(r"\multicolumn{" + str(n + 1) + r"}{l}{\footnotesize{"
                     r"Wilcoxon signed-rank (paired), two-sided, Holm-Bonferroni corrected.}} \\")
        lines.append(r"\end{tabular}")
        lines.append("")

    latex = "\n".join(lines) + "\n"
    out_path = out / "pairwise_augmentation_table.tex"
    out_path.write_text(latex)
    print(f"Pairwise augmentation table saved to: {out_path}")
    print(latex)


def compute_feature_group_stats(feat_df: pd.DataFrame, group_col: str,
                                feature_cols: list) -> pd.DataFrame:
    """
    Per-group (mode or model_name) mean, std, 95% CI, and 99% CI for each feature.
    Also runs Kruskal-Wallis across groups.
    Returns: feature | <group_col> | n | mean | std | ci95_lower | ci95_upper | ci99_lower | ci99_upper | kw_statistic | kw_p_value
    """
    rows = []
    groups = feat_df[group_col].dropna().unique()
    for feat in feature_cols:
        group_data = {
            g: feat_df.loc[feat_df[group_col] == g, feat].dropna().values
            for g in groups
        }
        arrays = [v for v in group_data.values() if len(v) >= 2]
        if len(arrays) >= 2:
            kw_stat, kw_p = stats.kruskal(*arrays)
        else:
            kw_stat, kw_p = float("nan"), float("nan")

        for g, vals in group_data.items():
            n = len(vals)
            if n == 0:
                continue
            mean = float(np.mean(vals))
            std = float(np.std(vals, ddof=1)) if n > 1 else 0.0
            row = {
                "feature": feat,
                group_col: g,
                "n": n,
                "mean": round(mean, 6),
                "std": round(std, 6),
                "kw_statistic": round(kw_stat, 4) if not np.isnan(kw_stat) else None,
                "kw_p_value": round(kw_p, 4) if not np.isnan(kw_p) else None,
            }
            for conf in (0.95, 0.99):
                if n > 1:
                    se = std / np.sqrt(n)
                    t_crit = stats.t.ppf((1 + conf) / 2, df=n - 1)
                    ci_lower = mean - t_crit * se
                    ci_upper = mean + t_crit * se
                else:
                    ci_lower = ci_upper = mean
                pct = int(conf * 100)
                row[f"ci{pct}_lower"] = round(ci_lower, 6)
                row[f"ci{pct}_upper"] = round(ci_upper, 6)
            rows.append(row)
    return pd.DataFrame(rows)


def compute_iou_summary(iou_df: pd.DataFrame, iou_cols: list) -> pd.DataFrame:
    """
    Returns a long-form table with mean, std, 95% CI, and 99% CI per model/augmentation.
    CIs are computed using a t-distribution (scipy.stats.t).
    """
    rows = []
    for model_name, group in iou_df.groupby("model_name"):
        for col in iou_cols + (["avg_iou"] if "avg_iou" in iou_df.columns else []):
            values = group[col].dropna().values
            n = len(values)
            mean = float(np.mean(values))
            std = float(np.std(values, ddof=1)) if n > 1 else 0.0
            row = {
                "model": model_name,
                "augmentation": col.replace("iou_", ""),
                "mean_iou": round(mean, 4),
                "std": round(std, 4),
                "n": n,
            }
            for conf in (0.95, 0.99):
                if n > 1:
                    se = std / np.sqrt(n)
                    t_crit = stats.t.ppf((1 + conf) / 2, df=n - 1)
                    ci_lower = mean - t_crit * se
                    ci_upper = mean + t_crit * se
                else:
                    ci_lower = ci_upper = mean
                pct = int(conf * 100)
                row[f"ci{pct}_lower"] = round(ci_lower, 4)
                row[f"ci{pct}_upper"] = round(ci_upper, 4)
            rows.append(row)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Radiomics CSV helpers
# ---------------------------------------------------------------------------

def load_radiomics_csvs(csv_dir: str) -> pd.DataFrame:
    pattern = re.compile(r"^\d+-\d+.*\.csv$")
    paths = []
    for root, _, files in os.walk(csv_dir):
        for f in files:
            if pattern.match(f):
                paths.append(os.path.join(root, f))
    if not paths:
        return pd.DataFrame()
    dfs = [pd.read_csv(p) for p in sorted(paths)]
    df = pd.concat(dfs, ignore_index=True)
    unnamed = [c for c in df.columns if c.startswith("Unnamed")]
    if unnamed:
        df = df.drop(columns=unnamed)
    if "model_name" in df.columns:
        df["model_name"] = df["model_name"].map(normalize_model_name)
    return df


# ---------------------------------------------------------------------------
# LaTeX table
# ---------------------------------------------------------------------------

def compute_cell_significance(iou_df: pd.DataFrame, iou_cols: list) -> dict:
    """
    Per-cell one-sample Wilcoxon signed-rank test on the excess IoU over the
    random-region null (hypergeometric expectation given actual mask sizes).

    For each (model, augmentation): excess_i = iou_i - null_iou_i (per sample).
    H0: median excess = 0  vs  H1: median excess > 0.

    For avg_iou the per-sample average excess across augmentations is used.

    Returns dict: (model_name, col) -> {"p_value": float, "significant": bool}.
    """
    result: dict = {}

    for model_name, group in iou_df.groupby("model_name"):
        for col in iou_cols:
            null_col = f"null_{col}"
            if null_col not in group.columns:
                result[(model_name, col)] = {"p_value": float("nan"), "significant": False}
                continue
            excess = (group[col] - group[null_col]).dropna().values
            if len(excess) < 2:
                result[(model_name, col)] = {"p_value": float("nan"), "significant": False}
                continue
            _, p = stats.wilcoxon(excess, alternative="greater")
            result[(model_name, col)] = {"p_value": float(p), "significant": p < 0.05}

        # avg_iou: per-sample mean excess across augmentation columns
        avail = [c for c in iou_cols if f"null_{c}" in group.columns]
        if avail:
            obs_avg = group[avail].mean(axis=1)
            null_avg = group[[f"null_{c}" for c in avail]].mean(axis=1)
            excess_avg = (obs_avg - null_avg).dropna().values
            if len(excess_avg) >= 2:
                _, p = stats.wilcoxon(excess_avg, alternative="greater")
                result[(model_name, "avg_iou")] = {"p_value": float(p), "significant": p < 0.05}
            else:
                result[(model_name, "avg_iou")] = {"p_value": float("nan"), "significant": False}

    return result


def _iou_table_parts(iou_df: pd.DataFrame, iou_cols: list) -> tuple:
    """Shared computation for both LaTeX table functions."""
    aug_labels = [AUGMENTATION_DISPLAY_NAMES.get(c.replace("iou_", ""), c.replace("iou_", ""))
                  for c in iou_cols]
    label_to_col = {lbl: col for lbl, col in zip(aug_labels, iou_cols)}
    label_to_col["Average IoU"] = "avg_iou"

    model_rows: dict[str, dict] = {}
    for model_name, group in iou_df.groupby("model_name"):
        row: dict = {}
        aug_means = []
        for col, label in zip(iou_cols, aug_labels):
            vals = group[col].dropna().values
            n = len(vals)
            mean = float(np.mean(vals))
            se = float(np.std(vals, ddof=1)) / np.sqrt(n) if n > 1 else 0.0
            half = stats.t.ppf(0.975, df=n - 1) * se if n > 1 else 0.0
            row[label] = (mean, half)
            aug_means.append(mean)
        sample_avgs = iou_df.loc[iou_df["model_name"] == model_name, iou_cols].mean(axis=1).dropna().values
        n_avg = len(sample_avgs)
        se_avg = float(np.std(sample_avgs, ddof=1)) / np.sqrt(n_avg) if n_avg > 1 else 0.0
        half_avg = stats.t.ppf(0.975, df=n_avg - 1) * se_avg if n_avg > 1 else 0.0
        row["Average IoU"] = (float(np.mean(sample_avgs)), half_avg)
        model_rows[model_name] = row

    cell_sig = compute_cell_significance(iou_df, iou_cols)
    model_order = sorted(model_rows.keys())
    iou_col_labels = aug_labels + ["Average IoU"]
    return model_rows, cell_sig, model_order, iou_col_labels, label_to_col, len(aug_labels)


def generate_latex_table(iou_df: pd.DataFrame, iou_cols: list, out: Path) -> None:
    """
    IoU table: mean ± 95% CI per cell, shaded green/grey by significance vs
    random-region null (one-sample Wilcoxon on per-sample excess IoU).
    Requires \\usepackage{xcolor} and \\usepackage{colortbl} in preamble.
    """
    model_rows, cell_sig, model_order, iou_col_labels, label_to_col, n_aug = \
        _iou_table_parts(iou_df, iou_cols)

    n_total = 1 + len(iou_col_labels)

    def fmt_iou(mean: float, half: float) -> str:
        return rf"${mean:.3f} \pm {half:.3f}$"

    lines = []
    lines.append(r"\begin{tabular}{r" + "c" * len(iou_col_labels) + "}")
    lines.append(r"\toprule")
    lines.append(r" & \multicolumn{" + str(n_aug + 1) + r"}{c}{\textbf{Image manipulation}} \\")
    lines.append(r"\cmidrule(lr){2-" + str(n_aug + 2) + "}")
    header = [r"\textbf{Id}"] + [r"\textbf{" + lbl + "}" for lbl in iou_col_labels]
    lines.append(" & ".join(header) + r" \\")
    lines.append(r"\midrule")
    for model_name in model_order:
        row = model_rows[model_name]
        cells = [model_name] + [fmt_iou(*row[lbl]) for lbl in iou_col_labels]
        lines.append(" & ".join(cells) + r" \\")
    lines.append(r"\bottomrule")
    lines.append(r"\multicolumn{" + str(n_total) + r"}{l}{\footnotesize{"
                 r"Values are mean $\pm$ 95\% CI (t-distribution).}} \\")
    lines.append(r"\end{tabular}")

    latex = "\n".join(lines) + "\n"
    out_path = out / "iou_table.tex"
    out_path.write_text(latex)
    print(f"IoU table saved to: {out_path}")
    print(latex)


def generate_pvalue_latex_table(iou_df: pd.DataFrame, iou_cols: list, out: Path) -> None:
    """
    P-value table: one-sample Wilcoxon p-values (observed vs random-region null)
    per cell, shaded green/grey by significance.
    Requires \\usepackage{xcolor} and \\usepackage{colortbl} in preamble.
    """
    model_rows, cell_sig, model_order, iou_col_labels, label_to_col, n_aug = \
        _iou_table_parts(iou_df, iou_cols)

    n_total = 1 + len(iou_col_labels)

    def sci(p: float) -> str:
        if np.isnan(p):
            return r"\text{---}"
        exp = int(np.floor(np.log10(p))) if p > 0 else 0
        mantissa = p / 10 ** exp
        return rf"{mantissa:.2f} \times 10^{{{exp}}}"

    def fmt_p(model_name: str, label: str) -> str:
        entry = cell_sig.get((model_name, label_to_col[label]), {})
        p = entry.get("p_value", float("nan"))
        return f"${sci(p)}$"

    lines = []
    lines.append(r"\begin{tabular}{r" + "c" * len(iou_col_labels) + "}")
    lines.append(r"\toprule")
    lines.append(r" & \multicolumn{" + str(n_aug + 1) + r"}{c}{\textbf{Image manipulation}} \\")
    lines.append(r"\cmidrule(lr){2-" + str(n_aug + 2) + "}")
    header = [r"\textbf{Id}"] + [r"\textbf{" + lbl + "}" for lbl in iou_col_labels]
    lines.append(" & ".join(header) + r" \\")
    lines.append(r"\midrule")
    for model_name in model_order:
        cells = [model_name] + [fmt_p(model_name, lbl) for lbl in iou_col_labels]
        lines.append(" & ".join(cells) + r" \\")
    lines.append(r"\bottomrule")
    lines.append(r"\multicolumn{" + str(n_total) + r"}{l}{\footnotesize{"
                 r"$p$-values: one-sample Wilcoxon signed-rank test, H$_0$: median excess IoU "
                 r"over random-region null $= 0$, one-sided.}} \\")
    lines.append(r"\end{tabular}")

    latex = "\n".join(lines) + "\n"
    out_path = out / "pvalue_table.tex"
    out_path.write_text(latex)
    print(f"P-value table saved to: {out_path}")
    print(latex)


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def plot_iou_per_augmentation(iou_df: pd.DataFrame, out: Path, confidence: float) -> None:
    """Bar chart with CI error bars: mean IoU per augmentation, grouped by model."""
    iou_cols = [c for c in iou_df.columns if c.startswith("iou_")]
    if not iou_cols:
        return

    melted = iou_df[["model_name"] + iou_cols].melt(
        id_vars="model_name", var_name="augmentation", value_name="iou"
    )
    melted["augmentation"] = melted["augmentation"].str.replace("iou_", "", regex=False)

    pct = int(confidence * 100)
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(data=melted, x="augmentation", y="iou", hue="model_name",
                capsize=0.05, errorbar=("ci", pct), err_kws={"linewidth": 1.5}, ax=ax)
    ax.set_title(f"Mean IoU vs baseline (clear image) per augmentation — {pct}% CI")
    ax.set_xlabel("Augmentation")
    ax.set_ylabel("Mean IoU")
    ax.legend(title="Model")
    plt.tight_layout()
    fname = out / f"iou_per_augmentation_{pct}ci.png"
    fig.savefig(fname, dpi=150)
    plt.close(fig)
    print(f"Saved: {fname}")


def plot_iou_boxplot(iou_df: pd.DataFrame, out: Path) -> None:
    """Box plot: IoU distribution per augmentation mode."""
    iou_cols = [c for c in iou_df.columns if c.startswith("iou_")]
    if not iou_cols:
        return

    melted = iou_df[["model_name"] + iou_cols].melt(
        id_vars="model_name", var_name="augmentation", value_name="iou"
    )
    melted["augmentation"] = melted["augmentation"].str.replace("iou_", "", regex=False)

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.boxplot(data=melted, x="augmentation", y="iou", hue="model_name", ax=ax)
    ax.set_title("IoU distribution vs baseline (clear image) per augmentation")
    ax.set_xlabel("Augmentation")
    ax.set_ylabel("IoU")
    ax.legend(title="Model")
    plt.tight_layout()
    fig.savefig(out / "iou_boxplot.png", dpi=150)
    plt.close(fig)
    print(f"Saved: {out / 'iou_boxplot.png'}")


def plot_tsne(features: pd.DataFrame, df_meta: pd.DataFrame, out: Path,
              pca_variance: float = 0.99) -> None:
    """t-SNE scatter plots coloured by mode and model_name."""
    scaler = StandardScaler()
    scaled = scaler.fit_transform(features)

    pca = PCA(n_components=pca_variance)
    pca_features = pca.fit_transform(scaled)

    tsne = TSNE(n_components=2, random_state=42)
    tsne_features = tsne.fit_transform(pca_features)

    plot_df = df_meta.copy()
    plot_df["tSNE-1"] = tsne_features[:, 0]
    plot_df["tSNE-2"] = tsne_features[:, 1]

    for col, fname in [("mode", "tsne_mode.png"), ("model_name", "tsne_model.png")]:
        if col not in plot_df.columns:
            continue
        fig, ax = plt.subplots(figsize=(8, 8))
        sns.scatterplot(
            data=plot_df, x="tSNE-1", y="tSNE-2",
            hue=col, palette="Set1", s=60, ax=ax
        )
        ax.set_xlabel("t-SNE component 1")
        ax.set_ylabel("t-SNE component 2")
        ax.legend(title=col, loc="upper left")
        plt.tight_layout()
        fig.savefig(out / fname, dpi=150)
        plt.close(fig)
        print(f"Saved: {out / fname}")


# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------

def run_pca_classification(features: pd.DataFrame, labels: pd.Series,
                           pca_variance: float = 0.99) -> dict:
    scaler = StandardScaler()
    scaled = scaler.fit_transform(features)

    pca = PCA(n_components=pca_variance)
    pca_features = pca.fit_transform(scaled)

    X_train, X_test, y_train, y_test = train_test_split(
        pca_features, labels, test_size=0.2, random_state=42, stratify=labels
    )
    clf = RandomForestClassifier(random_state=42)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    report = classification_report(y_test, y_pred)
    return {
        "n_components": pca_features.shape[1],
        "explained_variance": float(pca.explained_variance_ratio_.sum()),
        "report": report,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description="Radiomics statistics and IoU calculation")
    parser.add_argument("--npz_dir", default=DEFAULT_NPZ_DIR,
                        help=f"Root dir with model/mode/sample.npz layout (default: {DEFAULT_NPZ_DIR})")
    parser.add_argument("--csv_dir", default=None, help="Directory with radiomics CSV chunks")
    parser.add_argument("--output_dir", default="./results", help="Output directory")
    parser.add_argument("--baseline_mode", default="clear_images", help="Mode name used as IoU baseline")
    parser.add_argument("--tile_size", type=int, default=30, help="Tile size for IoU binarisation")
    parser.add_argument("--max_per_group", type=int, default=100, help="Max NPZ samples per model/mode group")
    parser.add_argument("--pca_variance", type=float, default=0.99, help="PCA variance to retain (0-1)")
    parser.add_argument("--no_legacy_tiles", action="store_false", dest="legacy_tiles",
                        help="Use fixed tiling (actual image dimensions) instead of the "
                             "original notebook tiling (hardcoded w=512, h=1024). "
                             "Legacy mode is the default as it reproduces notebook IoU values.")
    parser.set_defaults(legacy_tiles=True)
    return parser.parse_args()


def main():
    args = parse_args()
    validate_inputs(args)

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # --- IoU ---
    print(f"Scanning NPZ files in: {args.npz_dir}")
    npz_df = extract_npz_paths(args.npz_dir, max_per_group=args.max_per_group)
    if npz_df.empty:
        print("No NPZ files found — skipping IoU calculation.")
    else:
        print(f"Found {len(npz_df)} NPZ entries across "
              f"{npz_df['model_name'].nunique()} models and "
              f"{npz_df['mode'].nunique()} modes.")

        if args.legacy_tiles:
            print("Using legacy notebook tiling (w=512, h=1024 hardcoded).")
        iou_df = calculate_iou(npz_df, baseline_mode=args.baseline_mode,
                               legacy_tiles=args.legacy_tiles)

        iou_cols = [c for c in iou_df.columns if c.startswith("iou_")]
        if iou_cols:
            # Exclude dewarped from the average — it is only used in the
            # dewarped-vs-wrinkles comparison table.
            avg_cols = [c for c in iou_cols if c != "iou_dewarped"]
            iou_df["avg_iou"] = iou_df[avg_cols].mean(axis=1) if avg_cols else iou_df[iou_cols].mean(axis=1)

        iou_df.to_csv(out / "iou_results.csv", index=False)
        print(f"IoU results saved to: {out / 'iou_results.csv'}")

        if iou_cols:
            summary = compute_iou_summary(iou_df, iou_cols)
            summary.to_csv(out / "iou_per_model.csv", index=False)
            print(f"\nPer-model IoU with 95% and 99% CI saved to: {out / 'iou_per_model.csv'}")
            print(summary.to_string(index=False))

            significance = compute_iou_significance(iou_df, iou_cols)
            if not significance.empty:
                significance.to_csv(out / "iou_significance.csv", index=False)
                print(f"\nKruskal-Wallis IoU significance test saved to: {out / 'iou_significance.csv'}")
                print(significance.to_string(index=False))
                print(f"  significant at α=0.05: {significance['significant_95'].sum()}/{len(significance)} models")
                print(f"  significant at α=0.01: {significance['significant_99'].sum()}/{len(significance)} models")

            # Exclude dewarped from pairwise tests — it is only used in the
            # dewarped-vs-wrinkles comparison (Table 3).
            pw_cols = [c for c in iou_cols if c != "iou_dewarped"]
            pairwise = compute_pairwise_augmentation_significance(iou_df, pw_cols)
            if not pairwise.empty:
                pairwise.to_csv(out / "pairwise_augmentation.csv", index=False)
                print(f"\nPairwise Wilcoxon signed-rank (Holm-Bonferroni) saved to: {out / 'pairwise_augmentation.csv'}")
                print(pairwise.to_string(index=False))
                n_sig = pairwise["significant_05"].sum()
                print(f"  significant at α=0.05: {n_sig}/{len(pairwise)} pairs")

            generate_latex_table(iou_df, iou_cols, out)
            generate_pvalue_latex_table(iou_df, iou_cols, out)
            if not pairwise.empty:
                generate_pairwise_latex_table(pairwise, out)

            for conf in (0.95, 0.99):
                plot_iou_per_augmentation(iou_df, out, conf)
            plot_iou_boxplot(iou_df, out)

    # --- Radiomics features ---
    if args.csv_dir:
        print(f"\nLoading radiomics CSVs from: {args.csv_dir}")
        feat_df = load_radiomics_csvs(args.csv_dir)
        if feat_df.empty:
            print("No radiomics CSVs found.")
        else:
            feat_df.to_csv(out / "all_features.csv", index=False)
            print(f"Merged features ({len(feat_df)} rows, {len(feat_df.columns)} cols) "
                  f"saved to: {out / 'all_features.csv'}")

            feature_cols = feat_df.select_dtypes(include=["number"]).columns
            features = feat_df[feature_cols]

            meta_cols = [c for c in ["mode", "model_name", "full_path"] if c in feat_df.columns]
            plot_tsne(features, feat_df[meta_cols], out, pca_variance=args.pca_variance)

            for group_col in ["mode", "model_name"]:
                if group_col not in feat_df.columns or feat_df[group_col].nunique() < 2:
                    continue
                feat_stats = compute_feature_group_stats(feat_df, group_col, list(feature_cols))
                feat_stats.to_csv(out / f"feature_stats_by_{group_col}.csv", index=False)
                print(f"\nPer-{group_col} feature stats saved to: {out / f'feature_stats_by_{group_col}.csv'}")
                sig_base = feat_stats.dropna(subset=["kw_p_value"]).drop_duplicates("feature")["kw_p_value"]
                print(f"  significant at α=0.05: {(sig_base < 0.05).sum()}, at α=0.01: {(sig_base < 0.01).sum()}")

            for label_col in ["mode", "model_name"]:
                if label_col not in feat_df.columns or feat_df[label_col].nunique() < 2:
                    continue
                print(f"\nPCA + RandomForest — target: {label_col}")
                result = run_pca_classification(features, feat_df[label_col],
                                                pca_variance=args.pca_variance)
                print(f"  PCA components: {result['n_components']} "
                      f"(explained variance: {result['explained_variance']:.3f})")
                print(result["report"])

                report_path = out / f"classification_{label_col}.txt"
                with open(report_path, "w") as f:
                    f.write(f"Target: {label_col}\n")
                    f.write(f"PCA components: {result['n_components']} "
                            f"(explained variance: {result['explained_variance']:.3f})\n\n")
                    f.write(result["report"])
                print(f"  Report saved to: {report_path}")


if __name__ == "__main__":
    main()
