from __future__ import annotations

import argparse
import os
import shutil
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", tempfile.gettempdir())

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


OUT_DIR = Path("outputs/si_figures")
SI_DIR = OUT_DIR / "si"
CONTACT_DIR = OUT_DIR / "contact_sheets"

V8_DIR: Path | None = None
SI5_TABLES = Path("si5_tables")


VAR_CODE = {
    "Precipitation": "hist_pr",
    "Temperature": "hist_tas",
    "Solar Radiation": "ssr",
    "Snow Depth Water Equivalent": "sd",
    "Snow Cover": "snowc",
    "Snowfall": "sf",
    "Volumetric Soil Water Layer 1": "swvl1",
    "Volumetric Soil Water Deep": "swvl_deep",
    "Soil Temperature": "stl1",
    "Wind Speed": "wind_speed",
    "Potential Evapotranspiration": "PET",
}

DISPLAY_NAME_BY_CODE = {
    "hist_pr": "Precipitation",
    "hist_tas": "Air temperature",
    "ssr": "Radiation",
    "sd": "Snow depth",
    "snowc": "Snow cover",
    "sf": "Snowfall",
    "swvl1": "Soil water L1",
    "swvl_deep": "Deep soil water",
    "stl1": "Soil temperature",
    "wind_speed": "Wind",
    "PET": "Potential evapotranspiration",
}

RANK_DATA = pd.DataFrame(
    [
        ("hist_pr", 3.153, 1, 1, 1),
        ("swvl1", 1.565, 2, 2, 2),
        ("hist_tas", 1.231, 3, 4, 5),
        ("swvl_deep", 1.026, 4, 9, 3),
        ("ssr", 0.788, 5, 7, 9),
        ("sd", 0.787, 6, 3, 4),
        ("PET", 0.771, 7, 5, 6),
        ("snowc", 0.770, 8, 6, 7),
        ("stl1", 0.644, 9, 8, 8),
        ("sf", 0.506, 10, 10, 10),
        ("wind_speed", 0.390, 11, 11, 11),
    ],
    columns=["variable", "ig_mean", "ig_rank", "rank_nse", "rank_kge"],
)


def setup() -> None:
    for path in (SI_DIR, CONTACT_DIR):
        path.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8.6,
            "axes.labelsize": 8.8,
            "axes.titlesize": 9.0,
            "xtick.labelsize": 7.8,
            "ytick.labelsize": 7.8,
            "legend.fontsize": 7.4,
            "axes.linewidth": 0.65,
            "savefig.dpi": 400,
            "figure.dpi": 300,
        }
    )


def copy_existing_v8_figures() -> list[Path]:
    copied: list[Path] = []
    if V8_DIR is None:
        return copied
    if not V8_DIR.exists():
        raise FileNotFoundError(V8_DIR)
    for src in sorted(V8_DIR.glob("*.png")):
        dst = SI_DIR / src.name.replace("_v8", "_v9")
        shutil.copy2(src, dst)
        copied.append(dst)
    return copied


def style_axis(ax) -> None:
    ax.tick_params(length=2.2, width=0.55)
    ax.grid(axis="y", color=(0, 0, 0, 0.10), lw=0.28, linestyle=(0, (1.0, 3.5)))
    for spine in ax.spines.values():
        spine.set_linewidth(0.62)


def save_si5_1_hist() -> Path:
    df = pd.read_csv(SI5_TABLES / "si5_1_per_basin_closure_residuals.csv")
    fig, ax = plt.subplots(figsize=(6.2, 3.7))
    ax.hist(df["mae_abs_e_m3s"], bins=24, color="#4c78a8", edgecolor="white", linewidth=0.4)
    median = df["mae_abs_e_m3s"].median()
    ax.axvline(median, color="#d62728", lw=1.3, label=f"median = {median:.3f}")
    ax.set_xlabel("Per-basin MAE(|e|) (m³/s)")
    ax.set_ylabel("Number of basins")
    ax.legend(frameon=False, loc="upper right")
    style_axis(ax)
    out = SI_DIR / "si5_1_per_basin_closure_mae_hist_v9.png"
    fig.tight_layout(pad=0.55)
    fig.savefig(out, bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)
    return out


def save_si5_2_flow() -> Path:
    df = pd.read_csv(SI5_TABLES / "si5_1_closure_by_flow_stratum_within_basin_q.csv")
    order = ["low_q_lt_basin_q25", "mid_basin_q25_to_q90", "high_q_gt_basin_q90"]
    labels = ["Low flow\n(< basin Q25)", "Mid flow\n(Q25-Q90)", "High flow\n(> basin Q90)"]
    df["flow_stratum_basin_q"] = pd.Categorical(df["flow_stratum_basin_q"], order, ordered=True)
    df = df.sort_values("flow_stratum_basin_q")
    fig, ax1 = plt.subplots(figsize=(6.45, 3.7))
    x = np.arange(len(df))
    bars = ax1.bar(x, df["mae_abs_e_m3s"], color="#59a14f", width=0.56, edgecolor="0.3", linewidth=0.3)
    ax1.set_ylabel("MAE(|e|) (m³/s)")
    ax1.set_xlabel("Flow stratum based on within-basin Qsim quantiles")
    ax1.set_xticks(x, labels)
    ax2 = ax1.twinx()
    ax2.plot(x, df["nmae_by_mean_qsim"], color="#e15759", marker="o", lw=1.2, ms=4)
    ax2.set_ylabel("NMAE by mean Qsim")
    for bar, value in zip(bars, df["mae_abs_e_m3s"]):
        ax1.text(bar.get_x() + bar.get_width() / 2, value, f"{value:.3f}", ha="center", va="bottom", fontsize=7.0)
    style_axis(ax1)
    ax2.tick_params(length=2.2, width=0.55)
    for spine in ax2.spines.values():
        spine.set_linewidth(0.62)
    out = SI_DIR / "si5_2_closure_by_flow_stratum_v9.png"
    fig.tight_layout(pad=0.55)
    fig.savefig(out, bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)
    return out


def save_si5_3_season() -> Path:
    df = pd.read_csv(SI5_TABLES / "si5_1_closure_by_season.csv")
    order = ["DJF", "MAM", "JJA", "SON"]
    df["season"] = pd.Categorical(df["season"], order, ordered=True)
    df = df.sort_values("season")
    fig, ax = plt.subplots(figsize=(5.7, 3.45))
    x = np.arange(len(df))
    bars = ax.bar(x, df["mae_abs_e_m3s"], color="#f28e2b", width=0.58, edgecolor="0.3", linewidth=0.3)
    ax.set_ylabel("MAE(|e|) (m³/s)")
    ax.set_xlabel("Season")
    ax.set_xticks(x, df["season"].astype(str))
    for bar, value in zip(bars, df["mae_abs_e_m3s"]):
        ax.text(bar.get_x() + bar.get_width() / 2, value, f"{value:.3f}", ha="center", va="bottom", fontsize=7.0)
    style_axis(ax)
    out = SI_DIR / "si5_3_closure_by_season_v9.png"
    fig.tight_layout(pad=0.55)
    fig.savefig(out, bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)
    return out


def save_si5_4_heatmap() -> Path:
    raw = pd.read_csv(SI5_TABLES / "si5_2_per_seed_raw_variable_unsigned_ig_importance.csv")
    matrix = raw.drop(columns=["run"]).set_index("seed")
    matrix.columns = [VAR_CODE.get(col, col) for col in matrix.columns]
    display_columns = [DISPLAY_NAME_BY_CODE.get(col, col) for col in matrix.columns]
    fig, ax = plt.subplots(figsize=(7.0, 3.1))
    im = ax.imshow(matrix.to_numpy(), aspect="auto", cmap="YlGnBu")
    ax.set_yticks(np.arange(len(matrix.index)), [idx.replace("_", " ") for idx in matrix.index])
    ax.set_xticks(np.arange(len(matrix.columns)), display_columns, rotation=35, ha="right")
    ax.set_xlabel("Dynamic input variable")
    ax.set_ylabel("Seed run")
    cbar = fig.colorbar(im, ax=ax, shrink=0.86, pad=0.02)
    cbar.set_label("Mean |IG| (m³/s)", fontsize=7.8)
    cbar.ax.tick_params(labelsize=7.0, length=2.0)
    for spine in ax.spines.values():
        spine.set_linewidth(0.62)
    out = SI_DIR / "si5_4_seed_variable_importance_heatmap_v9.png"
    fig.tight_layout(pad=0.45)
    fig.savefig(out, bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)
    return out


def save_si5_5_cv_bar() -> Path:
    df = pd.read_csv(SI5_TABLES / "si5_3_raw_variable_cv_rank_stability.csv")
    df = df.sort_values("cv", ascending=True)
    labels = [DISPLAY_NAME_BY_CODE.get(code, code) for code in df["variable_code"].fillna(df["variable"])]
    fig, ax = plt.subplots(figsize=(6.35, 4.0))
    ax.barh(labels, df["cv"], color="#4c78a8", edgecolor="0.3", linewidth=0.3)
    ax.set_xlabel("Across-seed coefficient of variation")
    ax.set_ylabel("Dynamic input variable")
    style_axis(ax)
    out = SI_DIR / "si5_5_raw_variable_cv_bar_v9.png"
    fig.tight_layout(pad=0.55)
    fig.savefig(out, bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)
    return out


def save_si6_1_rank_comparison() -> Path:
    df = RANK_DATA.copy()
    df.to_csv(SI_DIR / "si6_1_rank_comparison_source_v9.csv", index=False)
    display_names = [DISPLAY_NAME_BY_CODE.get(v, v) for v in df["variable"]]
    fig, ax = plt.subplots(figsize=(6.6, 4.5))
    y = np.arange(len(df))
    ax.plot(df["ig_rank"], y, marker="o", lw=1.3, color="#2f65d9", label="IG rank")
    ax.plot(df["rank_nse"], y, marker="s", lw=1.1, color="#e15759", label="Ablation rank by -ΔNSE")
    ax.plot(df["rank_kge"], y, marker="^", lw=1.1, color="#59a14f", label="Ablation rank by -ΔKGE")
    for _, row in df.iterrows():
        yi = int(row["ig_rank"]) - 1
        ax.plot([row["ig_rank"], row["rank_nse"]], [yi, yi], color="#e15759", alpha=0.22, lw=0.75)
        ax.plot([row["ig_rank"], row["rank_kge"]], [yi, yi], color="#59a14f", alpha=0.22, lw=0.75)
    ax.set_yticks(y, display_names)
    ax.invert_yaxis()
    ax.set_xlim(0.5, 11.5)
    ax.set_xticks(range(1, 12))
    ax.set_xlabel("Rank among 11 dynamic inputs (1 = strongest)")
    ax.set_ylabel("Dynamic input variable")
    ax.legend(frameon=False, loc="lower right", fontsize=7.2)
    ax.grid(axis="x", color=(0, 0, 0, 0.12), lw=0.28, linestyle=(0, (1.0, 3.5)))
    for spine in ax.spines.values():
        spine.set_linewidth(0.62)
    out = SI_DIR / "si6_1_ig_ablation_rank_comparison_v9.png"
    fig.tight_layout(pad=0.55)
    fig.savefig(out, bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)
    return out


def find_font(size: int, bold: bool = False):
    candidates = [
        "/System/Library/Fonts/Supplemental/Arial Bold.ttf" if bold else "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/System/Library/Fonts/Supplemental/Helvetica Bold.ttf" if bold else "/System/Library/Fonts/Supplemental/Helvetica.ttf",
    ]
    for candidate in candidates:
        try:
            return ImageFont.truetype(candidate, size)
        except OSError:
            continue
    return ImageFont.load_default()


def make_contact_sheet(paths: list[Path]) -> Path:
    thumbs = []
    max_w = 680
    label_font = find_font(18, bold=True)
    for path in paths:
        img = Image.open(path).convert("RGB")
        scale = min(max_w / img.width, 1.0)
        resized = img.resize((int(img.width * scale), int(img.height * scale)))
        tile = Image.new("RGB", (max_w, resized.height + 42), "white")
        tile.paste(resized, ((max_w - resized.width) // 2, 34))
        ImageDraw.Draw(tile).text((8, 6), path.name, fill="black", font=label_font)
        thumbs.append(tile)
    cols = 2
    gap = 34
    rows = int(np.ceil(len(thumbs) / cols))
    row_heights = [max(thumbs[i].height for i in range(r * cols, min((r + 1) * cols, len(thumbs)))) for r in range(rows)]
    sheet_w = cols * max_w + (cols + 1) * gap
    sheet_h = sum(row_heights) + (rows + 1) * gap
    sheet = Image.new("RGB", (sheet_w, sheet_h), "white")
    y = gap
    for r in range(rows):
        x = gap
        for c in range(cols):
            idx = r * cols + c
            if idx >= len(thumbs):
                break
            sheet.paste(thumbs[idx], (x, y))
            x += max_w + gap
        y += row_heights[r] + gap
    out = CONTACT_DIR / "supporting_information_figures_contact_sheet.png"
    sheet.save(out)
    return out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build final Supporting Information validation figures.")
    parser.add_argument("--si5-tables", type=Path, required=True, help="Directory containing SI5 source tables.")
    parser.add_argument("--source-figures-dir", type=Path, help="Optional directory of previously finalized SI figures to copy.")
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def configure(args: argparse.Namespace) -> None:
    global SI5_TABLES, V8_DIR, OUT_DIR, SI_DIR, CONTACT_DIR
    SI5_TABLES = args.si5_tables.resolve()
    V8_DIR = args.source_figures_dir.resolve() if args.source_figures_dir else None
    OUT_DIR = args.output_dir.resolve()
    SI_DIR = OUT_DIR / "si"
    CONTACT_DIR = OUT_DIR / "contact_sheets"


def main() -> None:
    configure(parse_args())
    setup()
    copied = copy_existing_v8_figures()
    generated = [
        save_si5_1_hist(),
        save_si5_2_flow(),
        save_si5_3_season(),
        save_si5_4_heatmap(),
        save_si5_5_cv_bar(),
        save_si6_1_rank_comparison(),
    ]
    contact = make_contact_sheet([*generated, *copied])
    print(f"Copied {len(copied)} v8 figures into v9 directory")
    print(f"Generated {len(generated)} v9 figures")
    print(contact)


if __name__ == "__main__":
    main()
