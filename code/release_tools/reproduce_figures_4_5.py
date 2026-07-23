from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.patches import Patch


PACKAGE_ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = PACKAGE_ROOT / "figures" / "reproduced"
MAIN_DIR = OUT_DIR
GROUP_C = PACKAGE_ROOT / "derived_results" / "attribution" / "groupC_basin"
GROUP_DEF = PACKAGE_ROOT / "derived_results" / "attribution" / "groupD_E_F_daily"

XLIM = (129.0, 146.2)
YLIM = (30.8, 45.8)

GROUP_ORDER = [
    "pr",
    "soil_water",
    "snow_physics",
    "temp",
    "radiation",
    "pet",
    "soil_temp",
    "wind",
]
LABELS = {
    "pr": "Precipitation",
    "soil_water": "Soil water",
    "snow_physics": "Snow physics",
    "temp": "Air temperature",
    "radiation": "Radiation",
    "soil_temp": "Soil temperature",
    "wind": "Wind",
    "pet": "Potential evapotranspiration",
    "other": "Other",
}
COLORS = {
    "pr": "#2f65d9",
    "soil_water": "#1f78b4",
    "snow_physics": "#6f58c9",
    "temp": "#f28e2b",
    "radiation": "#edc948",
    "soil_temp": "#9c755f",
    "wind": "#59a14f",
    "pet": "#e15759",
    "other": "#b8b8b8",
}
SOFT_GUIDE = (1.0, 1.0, 1.0, 0.12)
SOFT_EDGE = (1.0, 1.0, 1.0, 0.18)
DASHED_GRID = (0, (0.8, 3.4))
MAG_GRID = (0.0, 0.0, 0.0, 0.16)
MAG_DASHED_GRID = (0, (0.9, 3.2))
CASE_BASINS = {
    "001": {"xytext": (143.35, 44.85), "ha": "left"},
    "107": {"xytext": (140.95, 38.35), "ha": "left"},
    "123": {"xytext": (131.55, 31.95), "ha": "left"},
    "129": {"xytext": (133.75, 33.25), "ha": "left"},
}

RAW_TO_GROUP = {
    "Precipitation": "pr",
    "Temperature": "temp",
    "Solar Radiation": "radiation",
    "Snow Depth Water Equivalent": "snow_physics",
    "Snow Cover": "snow_physics",
    "Snowfall": "snow_physics",
    "Volumetric Soil Water Layer 1": "soil_water",
    "Volumetric Soil Water Deep": "soil_water",
    "Soil Temperature": "soil_temp",
    "Wind Speed": "wind",
    "Potential Evapotranspiration": "pet",
}


def setup() -> None:
    MAIN_DIR.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8.2,
            "axes.labelsize": 8.4,
            "axes.titlesize": 8.8,
            "xtick.labelsize": 7.4,
            "ytick.labelsize": 7.4,
            "legend.fontsize": 7.4,
            "axes.linewidth": 0.65,
            "savefig.dpi": 400,
            "figure.dpi": 300,
        }
    )


def norm_id(value) -> str:
    if value is None:
        return ""
    s = str(value).strip()
    digits = "".join(re.findall(r"\d", s))
    if digits:
        return digits
    try:
        return str(int(float(s)))
    except Exception:
        return s


def load_mapping() -> tuple[dict[str, str], dict[str, str]]:
    df = pd.read_csv(BASIN_LIST)
    basin_col = [c for c in df.columns if "Basin" in c][0]
    grdc_col = [c for c in df.columns if "GRDC" in c][0]
    grdc_to_sawada: dict[str, str] = {}
    sawada_to_grdc: dict[str, str] = {}
    for _, row in df.iterrows():
        sawada = f"{int(row[basin_col]):03d}"
        grdc = norm_id(row[grdc_col])
        grdc_to_sawada[grdc] = sawada
        sawada_to_grdc[sawada] = grdc
    return grdc_to_sawada, sawada_to_grdc


def iter_polygon_parts(geom) -> Iterable[Polygon]:
    if isinstance(geom, Polygon):
        yield geom
    elif isinstance(geom, MultiPolygon):
        yield from geom.geoms


def plot_geom(ax, geom, facecolor="none", edgecolor="black", lw=0.4, alpha=1.0, zorder=1):
    for poly in iter_polygon_parts(geom):
        x, y = poly.exterior.xy
        ax.fill(x, y, facecolor=facecolor, edgecolor=edgecolor, linewidth=lw, alpha=alpha, zorder=zorder)
        for interior in poly.interiors:
            xi, yi = interior.xy
            ax.fill(xi, yi, facecolor="white", edgecolor=edgecolor, linewidth=max(lw * 0.5, 0.1), zorder=zorder + 0.01)


def load_basin_geoms():
    grdc_to_sawada, _ = load_mapping()
    target = set(grdc_to_sawada)
    data = json.loads(BASIN_GEOJSON.read_text())
    id_field = None
    for cand in ("GRDC_NO", "grdc_no", "BASIN_ID", "id", "ID", "station", "Name", "name"):
        props0 = data["features"][0].get("properties") or data["features"][0].get("attributes") or {}
        if cand in props0:
            id_field = cand
            break
    if id_field is None:
        raise RuntimeError("Cannot find basin ID field in stationbasins.geojson")
    records = []
    for feat in data["features"]:
        props = feat.get("properties") or feat.get("attributes") or {}
        grdc = norm_id(props.get(id_field))
        if grdc not in target:
            continue
        geom_json = feat["geometry"]
        if "type" in geom_json:
            geom = shape(geom_json)
        elif "rings" in geom_json:
            polys = [Polygon(ring) for ring in geom_json["rings"] if len(ring) >= 4]
            geom = polys[0] if len(polys) == 1 else MultiPolygon(polys)
        else:
            raise RuntimeError("Unsupported basin geometry format")
        records.append({"grdc": grdc, "basin_id": grdc_to_sawada[grdc], "geom": geom})
    return sorted(records, key=lambda r: r["basin_id"])


def load_japan_geoms():
    reader = shapefile.Reader(str(COUNTRY_SHP), encoding="utf-8", encodingErrors="ignore")
    fields = [f[0] for f in reader.fields[1:]]
    geoms = []
    for shp, rec in zip(reader.shapes(), reader.records()):
        vals = dict(zip(fields, rec))
        if vals.get("ADMIN") == "Japan" or vals.get("ADM0_A3") == "JPN":
            geoms.append(shape(shp.__geo_interface__))
    return geoms


def style_map(ax, ylabel=True, xlabel=True) -> None:
    ax.set_xlim(*XLIM)
    ax.set_ylim(*YLIM)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xticks([130, 134, 138, 142, 146])
    ax.set_yticks([32, 36, 40, 44])
    ax.tick_params(length=2.2, width=0.55, labelsize=7.4)
    ax.set_xlabel("Longitude (degE)" if xlabel else "")
    ax.set_ylabel("Latitude (degN)" if ylabel else "")
    for spine in ax.spines.values():
        spine.set_linewidth(0.65)


def save_fig1() -> Path:
    basins = load_basin_geoms()
    japan = load_japan_geoms()
    fig, ax = plt.subplots(figsize=(6.0, 5.2))
    for geom in japan:
        plot_geom(ax, geom, facecolor="#f7f7f7", edgecolor="#9c9c9c", lw=0.45, zorder=0)
    for rec in basins:
        is_case = rec["basin_id"] in CASE_BASINS
        plot_geom(
            ax,
            rec["geom"],
            facecolor="#fef3c7" if is_case else "#8ecae6",
            edgecolor="#b91c1c" if is_case else "#1f2937",
            lw=0.72 if is_case else 0.30,
            zorder=4 if is_case else 2,
        )
    style_map(ax)
    for rec in basins:
        spec = CASE_BASINS.get(rec["basin_id"])
        if not spec:
            continue
        pt = rec["geom"].representative_point()
        ax.plot(pt.x, pt.y, marker="o", ms=3.5, mfc="#dc2626", mec="white", mew=0.55, zorder=8)
        ax.annotate(
            f"Basin {rec['basin_id']}",
            xy=(pt.x, pt.y),
            xytext=spec["xytext"],
            textcoords="data",
            ha=spec["ha"],
            va="center",
            fontsize=7.2,
            fontweight="bold",
            color="#7f1d1d",
            arrowprops=dict(arrowstyle="-", color="#7f1d1d", lw=0.55, shrinkA=2.0, shrinkB=2.5),
            bbox=dict(boxstyle="round,pad=0.13", fc="white", ec="#fecaca", lw=0.45, alpha=0.96),
            zorder=9,
        )
    ax.text(
        0.02,
        0.98,
        "135 study basins",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=7.6,
        bbox=dict(boxstyle="round,pad=0.18", fc="white", ec="0.78", lw=0.45),
    )
    out = MAIN_DIR / "figure1_study_basins_case_labels_wrr_polish_v1.png"
    fig.tight_layout(pad=0.4)
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out


def save_si_basin_id_map() -> Path:
    src = PAPER_READY / "si/si_01_basin_no_map.png"
    out = SI_DIR / "figure_si1_1_basin_id_map_wrr_polish_v1.png"
    shutil.copyfile(src, out)
    return out


def save_fig2() -> Path:
    basins = load_basin_geoms()
    japan = load_japan_geoms()
    metrics = pd.read_csv(RUN_ROOT / "test/ture/model_epoch020/test_metrics.csv")
    metrics["basin_id"] = metrics["basin"].astype(int).map(lambda x: f"{x:03d}")
    nse = dict(zip(metrics["basin_id"], metrics["NSE"]))

    def cat(v):
        if not np.isfinite(v):
            return "No data"
        if v > 0.8:
            return "NSE > 0.8"
        if v >= 0.5:
            return "0.5 <= NSE <= 0.8"
        return "NSE < 0.5"

    color = {"NSE > 0.8": "#1b9e77", "0.5 <= NSE <= 0.8": "#f0c808", "NSE < 0.5": "#d95f02", "No data": "#cccccc"}
    fig = plt.figure(figsize=(7.2, 4.05))
    ax = fig.add_axes([0.065, 0.13, 0.585, 0.77])
    ax_cdf = fig.add_axes([0.715, 0.13, 0.235, 0.77])
    for geom in japan:
        plot_geom(ax, geom, facecolor="#fafafa", edgecolor="#a6a6a6", lw=0.45, zorder=0)
    for rec in basins:
        label = cat(float(nse.get(rec["basin_id"], np.nan)))
        plot_geom(ax, rec["geom"], facecolor=color[label], edgecolor="#1f2937", lw=0.28, zorder=2)
    handles = [
        Patch(facecolor=color["NSE > 0.8"], edgecolor="black", label="NSE > 0.8"),
        Patch(facecolor=color["0.5 <= NSE <= 0.8"], edgecolor="black", label="0.5 <= NSE <= 0.8"),
        Patch(facecolor=color["NSE < 0.5"], edgecolor="black", label="NSE < 0.5"),
    ]
    ax.legend(handles=handles, loc="upper left", frameon=True, framealpha=0.95, borderpad=0.28, labelspacing=0.22, handlelength=1.2)
    style_map(ax)
    ax.text(-0.05, 1.035, "(a)", transform=ax.transAxes, fontsize=9.2, fontweight="bold", va="bottom", clip_on=False)

    for metric, line_color, marker, label in [
        ("NSE", "#1b9e77", "o", "NSE"),
        ("KGE", "#2f65d9", "s", "KGE"),
    ]:
        vals = np.sort(pd.to_numeric(metrics[metric], errors="coerce").dropna().to_numpy())
        y = (np.arange(len(vals)) + 1) / len(vals)
        ax_cdf.scatter(
            vals,
            y,
            s=10,
            marker=marker,
            color=line_color,
            alpha=0.68,
            linewidths=0,
            label=label,
            zorder=3,
        )
        median = float(np.nanmedian(vals))
        ax_cdf.axvline(median, color=line_color, lw=0.75, ls=(0, (2.0, 2.2)), alpha=0.75)
    ax_cdf.axvline(0.5, color="0.35", lw=0.58, ls=(0, (2.0, 2.2)), alpha=0.65)
    ax_cdf.axvline(0.8, color="0.35", lw=0.58, ls=(0, (2.0, 2.2)), alpha=0.65)
    ax_cdf.set_xlim(0.0, 1.0)
    ax_cdf.set_ylim(0.0, 1.0)
    ax_cdf.set_xlabel("Metric value")
    ax_cdf.set_ylabel("Cumulative fraction")
    ax_cdf.grid(color="0.90", lw=0.42)
    ax_cdf.legend(loc="upper left", frameon=False, fontsize=7.4)
    ax_cdf.text(-0.12, 1.035, "(b)", transform=ax_cdf.transAxes, fontsize=9.2, fontweight="bold", va="bottom", clip_on=False)
    for spine in ax_cdf.spines.values():
        spine.set_linewidth(0.65)
    out = MAIN_DIR / "figure2_performance_map_wrr_polish_v2.png"
    fig.savefig(out, bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)
    return out


def save_fig3() -> Path:
    df = pd.read_csv(ANALYSIS / "global_analysis/global_feature_importance_aggregated.csv")
    df["label"] = df["group"].map(LABELS)
    df = df.sort_values("mean_importance", ascending=True)
    fig, ax = plt.subplots(figsize=(6.8, 3.45))
    y = np.arange(len(df))
    ax.barh(
        y,
        df["mean_importance"],
        xerr=df["std_importance"],
        color=[COLORS[g] for g in df["group"]],
        edgecolor="0.25",
        linewidth=0.34,
        capsize=2.0,
    )
    ax.set_yticks(y, df["label"])
    ax.set_xlabel("Mean unsigned IG contribution (m³/s)")
    ax.grid(axis="x", color="0.90", lw=0.45)
    ax.set_axisbelow(True)
    for i, v in enumerate(df["mean_importance"]):
        ax.text(v + 0.06, i + 0.18, f"{v:.2f}", va="bottom", ha="left", fontsize=6.9)
    ax.set_xlim(0, max(df["mean_importance"] + df["std_importance"]) * 1.18)
    out = MAIN_DIR / "figure3_groupA_global_horizontal_wrr_polish_v1.png"
    fig.tight_layout(pad=0.45)
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out


def basin_group_values(basin_id: str) -> pd.DataFrame:
    data = json.loads((GROUP_C / f"basin_{basin_id}/detailed_analysis.json").read_text())
    vals = {g: 0.0 for g in GROUP_ORDER}
    errs = {g: 0.0 for g in GROUP_ORDER}
    for raw, stats in data["variable_stats"].items():
        group = RAW_TO_GROUP.get(raw)
        if not group:
            continue
        vals[group] += float(stats.get("mean_total_abs_physical", 0.0))
        errs[group] = math.sqrt(errs[group] ** 2 + float(stats.get("std_total_abs_physical", 0.0)) ** 2)
    return pd.DataFrame({"group": GROUP_ORDER, "value": [vals[g] for g in GROUP_ORDER], "err": [errs[g] for g in GROUP_ORDER]})


def save_fig4() -> Path:
    basin_ids = ["001", "123"]
    basin_data = {basin_id: basin_group_values(basin_id) for basin_id in basin_ids}
    common_xmax = max(float(df["value"].max()) for df in basin_data.values()) * 1.16

    fig, axes = plt.subplots(1, 2, figsize=(7.35, 3.35), sharex=True, sharey=True)
    y = np.arange(len(GROUP_ORDER))
    for ax, basin_id, panel in zip(axes, basin_ids, ["(a)", "(b)"]):
        df = basin_data[basin_id]
        y = np.arange(len(df))
        ax.barh(
            y,
            df["value"],
            color=[COLORS[g] for g in df["group"]],
            edgecolor="0.28",
            linewidth=0.32,
            height=0.68,
        )
        ax.set_yticks(y)
        if ax is axes[0]:
            ax.set_yticklabels([LABELS[g] for g in df["group"]])
        else:
            ax.tick_params(axis="y", labelleft=False, length=0)
        ax.grid(axis="x", color="0.91", lw=0.42)
        ax.set_axisbelow(True)
        ax.set_title(f"{panel} Basin {basin_id}", fontsize=8.9, pad=4, fontweight="normal")
        ax.set_xlim(0, max(common_xmax, 1.0))
        ax.set_xlabel("Mean unsigned IG contribution (m³/s)")
        for spine in ax.spines.values():
            spine.set_linewidth(0.65)
    axes[0].invert_yaxis()
    out = MAIN_DIR / "figure4_basin_fingerprints_two_panel_wrr_polish_v3.png"
    fig.tight_layout(pad=0.55, w_pad=1.25)
    fig.savefig(out, bbox_inches="tight", pad_inches=0.035)
    plt.close(fig)
    return out


def top_groups_for_basin(basin_id: str, n=3) -> list[str]:
    df = basin_group_values(basin_id).sort_values("value", ascending=False)
    return df["group"].head(n).tolist()


def _read_basin_csv(basin_id: str, name: str) -> pd.DataFrame:
    return pd.read_csv(GROUP_DEF / f"basin_{basin_id}/{name}", index_col=0)


def _with_other(df: pd.DataFrame, groups: list[str]) -> pd.DataFrame:
    out = df.copy()
    present = [g for g in groups if g in out.columns]
    all_present = [g for g in GROUP_ORDER if g in out.columns]
    out["other"] = out[all_present].sum(axis=1) - out[present].sum(axis=1)
    out["other"] = out["other"].clip(lower=0)
    return out[present + ["other"]]


def _share(df: pd.DataFrame) -> pd.DataFrame:
    denom = df.sum(axis=1).replace(0, np.nan)
    return df.div(denom, axis=0).fillna(0)


def add_panel_label(ax, panel: str) -> None:
    ax.text(
        -0.065,
        1.045,
        panel,
        transform=ax.transAxes,
        fontsize=8.8,
        fontweight="bold",
        ha="left",
        va="bottom",
        clip_on=False,
    )


def add_panel_label_inside(ax, panel: str) -> None:
    if not panel:
        return
    ax.text(
        0.018,
        0.965,
        panel,
        transform=ax.transAxes,
        fontsize=8.2,
        fontweight="bold",
        ha="left",
        va="top",
        bbox=dict(boxstyle="round,pad=0.08", fc="white", ec="none", alpha=0.82),
        zorder=10,
    )


def plot_seasonal_share(ax, basin_id: str, top_groups: list[str], panel: str):
    df = _read_basin_csv(basin_id, "seasonal_contributions.csv")
    vals = _share(_with_other(df, top_groups))
    bottom = np.zeros(len(df))
    x = np.arange(len(df.index))
    for g in vals.columns:
        ax.bar(x, vals[g], bottom=bottom, color=COLORS[g], label=LABELS[g], width=0.64, edgecolor=SOFT_EDGE, linewidth=0.05)
        bottom += vals[g].to_numpy()
    ax.set_xticks(x, df.index, rotation=22, ha="right")
    ax.set_ylim(0, 1)
    ax.set_ylabel("IG share")
    ax.grid(axis="y", color=SOFT_GUIDE, lw=0.26, linestyle=DASHED_GRID)
    add_panel_label_inside(ax, panel)


def plot_doy_share(ax, basin_id: str, top_groups: list[str], panel: str):
    df = pd.read_csv(GROUP_DEF / f"basin_{basin_id}/doy_mean_contributions.csv")
    vals = _share(_with_other(df.set_index("doy"), top_groups))
    ax.stackplot(vals.index, [vals[g].to_numpy() for g in vals.columns], colors=[COLORS[g] for g in vals.columns], linewidth=0.035, edgecolor=SOFT_EDGE)
    ax.set_xlim(1, 365)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Day of year")
    ax.set_ylabel("IG share")
    ax.grid(color=SOFT_GUIDE, lw=0.26, linestyle=DASHED_GRID)
    add_panel_label_inside(ax, panel)


def plot_daily_share(ax, basin_id: str, top_groups: list[str], panel: str):
    df = pd.read_csv(GROUP_DEF / f"basin_{basin_id}/daily_series_importance.csv")
    df.iloc[:, 0] = pd.to_datetime(df.iloc[:, 0])
    date_col = df.columns[0]
    vals = _with_other(df.set_index(date_col), top_groups)
    vals = vals.drop(columns=["total"], errors="ignore")
    vals = _share(vals)
    ax.stackplot(vals.index, [vals[g].to_numpy() for g in vals.columns], colors=[COLORS[g] for g in vals.columns], linewidth=0.035, edgecolor=SOFT_EDGE)
    ax.set_ylim(0, 1)
    ax.set_ylabel("IG share")
    ax.grid(color=SOFT_GUIDE, lw=0.26, linestyle=DASHED_GRID)
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    trim_daily_x_axis(ax, basin_id)
    ax.tick_params(axis="x", rotation=0)
    add_panel_label_inside(ax, panel)


def plot_seasonal_abs(ax, basin_id: str, top_groups: list[str], panel: str):
    df = _read_basin_csv(basin_id, "seasonal_contributions.csv")
    vals = _with_other(df, top_groups)
    bottom = np.zeros(len(df))
    x = np.arange(len(df.index))
    for g in vals.columns:
        ax.bar(x, vals[g], bottom=bottom, color=COLORS[g], label=LABELS[g], width=0.68, edgecolor=SOFT_EDGE, linewidth=0.05)
        bottom += vals[g].to_numpy()
    ax.set_xticks(x, df.index, rotation=22, ha="right")
    ax.set_ylabel("IG (m³/s)")
    ax.grid(axis="y", color=SOFT_GUIDE, lw=0.26, linestyle=DASHED_GRID)
    add_panel_label_inside(ax, panel)


def plot_doy_abs(ax, basin_id: str, top_groups: list[str], panel: str):
    df = pd.read_csv(GROUP_DEF / f"basin_{basin_id}/doy_mean_contributions.csv")
    vals = _with_other(df.set_index("doy"), top_groups)
    ax.stackplot(vals.index, [vals[g].to_numpy() for g in vals.columns], colors=[COLORS[g] for g in vals.columns], linewidth=0.035, edgecolor=SOFT_EDGE)
    ax.set_xlim(1, 365)
    ax.set_xlabel("Day of year")
    ax.set_ylabel("IG (m³/s)")
    ax.grid(color=SOFT_GUIDE, lw=0.26, linestyle=DASHED_GRID)
    add_panel_label_inside(ax, panel)


def plot_daily_abs(ax, basin_id: str, top_groups: list[str], panel: str):
    df = pd.read_csv(GROUP_DEF / f"basin_{basin_id}/daily_series_importance.csv")
    df.iloc[:, 0] = pd.to_datetime(df.iloc[:, 0])
    date_col = df.columns[0]
    vals = _with_other(df.set_index(date_col), top_groups).drop(columns=["total"], errors="ignore")
    ax.stackplot(vals.index, [vals[g].to_numpy() for g in vals.columns], colors=[COLORS[g] for g in vals.columns], linewidth=0.03, edgecolor=SOFT_EDGE)
    ax.set_ylabel("IG (m³/s)")
    ax.grid(color=SOFT_GUIDE, lw=0.26, linestyle=DASHED_GRID)
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.tick_params(axis="x", rotation=0)
    add_panel_label_inside(ax, panel)


def top_daily_groups_for_basin(basin_id: str, n: int = 5) -> list[str]:
    df = pd.read_csv(GROUP_DEF / f"basin_{basin_id}/daily_series_importance.csv")
    value_cols = [g for g in GROUP_ORDER if g in df.columns]
    means = df[value_cols].mean(axis=0).sort_values(ascending=False)
    return means.head(n).index.tolist()


def daily_date_limits(basin_id: str) -> tuple[pd.Timestamp, pd.Timestamp]:
    df = pd.read_csv(GROUP_DEF / f"basin_{basin_id}/daily_series_importance.csv", usecols=[0])
    dates = pd.to_datetime(df.iloc[:, 0])
    return dates.min(), dates.max()


def trim_daily_x_axis(ax, basin_id: str) -> None:
    x0, x1 = daily_date_limits(basin_id)
    ax.set_xlim(x0, x1 + pd.Timedelta(days=1))
    ax.margins(x=0)


def plot_daily_abs_top5(ax, basin_id: str, panel: str):
    df = pd.read_csv(GROUP_DEF / f"basin_{basin_id}/daily_series_importance.csv")
    df.iloc[:, 0] = pd.to_datetime(df.iloc[:, 0])
    date_col = df.columns[0]
    top_groups = top_daily_groups_for_basin(basin_id, n=5)
    vals = df.set_index(date_col)[top_groups]
    for group in top_groups:
        ax.plot(
            vals.index,
            vals[group],
            color=COLORS[group],
            lw=0.64,
            alpha=0.88,
            label=LABELS[group],
            solid_capstyle="round",
        )
    ax.set_ylim(bottom=0)
    ax.set_ylabel("IG (m³/s)")
    ax.grid(color=MAG_GRID, lw=0.23, linestyle=MAG_DASHED_GRID)
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    trim_daily_x_axis(ax, basin_id)
    ax.tick_params(axis="x", rotation=0)
    ax.legend(
        loc="upper right",
        ncol=min(5, len(top_groups)),
        frameon=True,
        framealpha=0.9,
        fontsize=6.55,
        borderpad=0.22,
        labelspacing=0.18,
        handlelength=1.25,
        columnspacing=0.72,
    )
    add_panel_label_inside(ax, panel)


def save_fig5_simplified() -> Path:
    fig = plt.figure(figsize=(7.55, 7.8))
    gs = fig.add_gridspec(
        4,
        3,
        height_ratios=[1.0, 0.92, 1.0, 0.92],
        hspace=0.58,
        wspace=0.34,
    )
    axes_share_001 = [fig.add_subplot(gs[0, i]) for i in range(3)]
    ax_mag_001 = fig.add_subplot(gs[1, :])
    axes_share_123 = [fig.add_subplot(gs[2, i]) for i in range(3)]
    ax_mag_123 = fig.add_subplot(gs[3, :])
    plot_groups = ["pr", "soil_water", "snow_physics"]

    plot_seasonal_share(axes_share_001[0], "001", plot_groups, "")
    plot_doy_share(axes_share_001[1], "001", plot_groups, "")
    plot_daily_share(axes_share_001[2], "001", plot_groups, "")
    plot_daily_abs_top5(ax_mag_001, "001", "")

    plot_seasonal_share(axes_share_123[0], "123", plot_groups, "")
    plot_doy_share(axes_share_123[1], "123", plot_groups, "")
    plot_daily_share(axes_share_123[2], "123", plot_groups, "")
    plot_daily_abs_top5(ax_mag_123, "123", "")

    all_axes = [*axes_share_001, ax_mag_001, *axes_share_123, ax_mag_123]
    for ax in all_axes:
        ax.tick_params(length=2.2, width=0.55)
        for spine in ax.spines.values():
            spine.set_linewidth(0.62)
    legend_groups = plot_groups + ["other"]
    handles = [Patch(facecolor=COLORS[g], label=LABELS[g]) for g in legend_groups]
    fig.legend(handles=handles, loc="lower center", ncol=4, frameon=False, bbox_to_anchor=(0.5, -0.002), handlelength=1.2, columnspacing=1.05)
    for ax, title in [
        (axes_share_001[0], "(a) Group D: seasonal"),
        (axes_share_001[1], "(b) Group E: mean day-of-year"),
        (axes_share_001[2], "(c) Group F: daily predictions"),
        (ax_mag_001, "(d) Group F: daily unsigned IG magnitude, top 5 controls"),
        (axes_share_123[0], "(e) Group D: seasonal"),
        (axes_share_123[1], "(f) Group E: mean day-of-year"),
        (axes_share_123[2], "(g) Group F: daily predictions"),
        (ax_mag_123, "(h) Group F: daily unsigned IG magnitude, top 5 controls"),
    ]:
        ax.set_title(title, loc="left", pad=6, fontweight="normal", fontsize=8.8)
    fig.text(0.013, 0.765, "Basin 001", rotation=90, ha="center", va="center", fontsize=8.8, fontweight="bold")
    fig.text(0.013, 0.335, "Basin 123", rotation=90, ha="center", va="center", fontsize=8.8, fontweight="bold")
    out = MAIN_DIR / "figure5_share_plus_groupF_top5_lines_wrr_polish_v3.png"
    fig.tight_layout(rect=(0.035, 0.036, 1, 0.985), pad=0.45)
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out


def add_panel_label_to_image(img: Image.Image, label: str, xy=(20, 18)) -> Image.Image:
    out = img.convert("RGB").copy()
    draw = ImageDraw.Draw(out)
    font = find_font(40, bold=True)
    bbox = draw.textbbox(xy, label, font=font)
    pad = 7
    draw.rectangle((bbox[0] - pad, bbox[1] - pad, bbox[2] + pad, bbox[3] + pad), fill="white", outline="0.75")
    draw.text(xy, label, fill="black", font=font)
    return out


def find_font(size: int, bold: bool = False):
    candidates = [
        "/System/Library/Fonts/Supplemental/Arial Bold.ttf" if bold else "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/System/Library/Fonts/Supplemental/Helvetica Bold.ttf" if bold else "/System/Library/Fonts/Supplemental/Helvetica.ttf",
    ]
    for cand in candidates:
        try:
            return ImageFont.truetype(cand, size)
        except OSError:
            continue
    return ImageFont.load_default()


def crop_to_content(img: Image.Image, pad=20, threshold=248) -> Image.Image:
    arr = np.asarray(img.convert("RGB"))
    mask = np.any(arr < threshold, axis=2)
    ys, xs = np.where(mask)
    if len(xs) == 0:
        return img
    x0, x1 = max(0, xs.min() - pad), min(img.width, xs.max() + pad)
    y0, y1 = max(0, ys.min() - pad), min(img.height, ys.max() + pad)
    return img.crop((x0, y0, x1, y1))


def save_fig5_full_si() -> Path:
    fig, axes = plt.subplots(2, 3, figsize=(7.55, 5.05))
    specs = [
        ("001", ["(a)", "(b)", "(c)"]),
        ("123", ["(d)", "(e)", "(f)"]),
    ]
    plot_groups = ["pr", "soil_water", "snow_physics"]
    for row, (basin_id, panels) in enumerate(specs):
        plot_seasonal_abs(axes[row, 0], basin_id, plot_groups, panels[0])
        plot_doy_abs(axes[row, 1], basin_id, plot_groups, panels[1])
        plot_daily_abs(axes[row, 2], basin_id, plot_groups, panels[2])
        axes[row, 0].text(
            -0.34,
            0.5,
            f"Basin {basin_id}",
            transform=axes[row, 0].transAxes,
            rotation=90,
            ha="center",
            va="center",
            fontsize=8.8,
            fontweight="bold",
        )
    for ax in axes.ravel():
        ax.tick_params(length=2.2, width=0.55)
        for spine in ax.spines.values():
            spine.set_linewidth(0.62)
    legend_groups = plot_groups + ["other"]
    handles = [Patch(facecolor=COLORS[g], label=LABELS[g]) for g in legend_groups]
    fig.legend(handles=handles, loc="lower center", ncol=4, frameon=False, bbox_to_anchor=(0.5, -0.008), handlelength=1.2, columnspacing=1.05)
    axes[0, 0].set_title("Group D: seasonal", pad=6)
    axes[0, 1].set_title("Group E: mean day-of-year", pad=6)
    axes[0, 2].set_title("Group F: daily predictions", pad=6)
    out = SI_DIR / "figure_si7_5_absolute_ig_magnitude_wrr_polish_v1.png"
    fig.tight_layout(rect=(0.045, 0.062, 1, 1), pad=0.45, w_pad=0.7, h_pad=0.65)
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out


def _module_from_path(module_name: str, path: Path):
    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    if spec.loader is None:
        raise RuntimeError(f"Cannot load module from {path}")
    spec.loader.exec_module(module)
    return module


def _style_fig6_axis(ax, show_xlabel: bool, show_ylabel: bool):
    ax.set_xlim(*XLIM)
    ax.set_ylim(*YLIM)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xticks([130, 134, 138, 142, 146])
    ax.set_yticks([32, 36, 40, 44])
    ax.tick_params(labelsize=7.2, length=2.1, width=0.55)
    ax.set_xlabel("Longitude (degE)" if show_xlabel else "", fontsize=8.2)
    ax.set_ylabel("Latitude (degN)" if show_ylabel else "", fontsize=8.2)
    for spine in ax.spines.values():
        spine.set_linewidth(0.62)
    ax.grid(False)


def _draw_fig6_base(ax, basins, japan):
    for geom in japan:
        plot_geom(ax, geom, facecolor="none", edgecolor="0.55", lw=0.36, zorder=8)
    for rec in basins:
        plot_geom(ax, rec["geom"], facecolor="none", edgecolor="black", lw=0.16, alpha=0.82, zorder=9)


def _mask_to_japan(data: np.ndarray, transform, japan) -> np.ndarray:
    """Hide raster cells outside Japan so Figure 6 maps do not show neighboring land."""
    mask = geometry_mask(
        [mapping(geom) for geom in japan],
        transform=transform,
        invert=True,
        out_shape=data.shape,
        all_touched=True,
    )
    out = data.copy()
    out[~mask] = np.nan
    return out


def _dominance_class(basin_id: str) -> str:
    df = basin_group_values(basin_id)
    group = df.sort_values("value", ascending=False).iloc[0]["group"]
    if group == "pr":
        return "precipitation"
    if group == "snow_physics":
        return "snow"
    if group == "soil_water":
        return "soil_water"
    if group == "temp":
        return "temperature"
    return "other"


def _plot_fig6_dominance(ax, basins, japan):
    color_map = {
        "precipitation": "#1e40af",
        "snow": "#60a5fa",
        "soil_water": "#92400e",
        "temperature": "#f97316",
        "other": "#6b7280",
    }
    label_map = {
        "precipitation": "Precipitation dominant",
        "snow": "Snow dominant",
        "soil_water": "Soil-water dominant",
        "temperature": "Temperature dominant",
        "other": "Other dominant",
    }
    for geom in japan:
        plot_geom(ax, geom, facecolor="#fbfbfb", edgecolor="0.55", lw=0.45, zorder=0)
    present = set()
    for rec in basins:
        dom = _dominance_class(rec["basin_id"])
        present.add(dom)
        plot_geom(ax, rec["geom"], facecolor=color_map[dom], edgecolor="black", lw=0.17, zorder=3)
    handles = [
        Patch(facecolor=color_map[key], edgecolor="black", linewidth=0.35, label=label_map[key])
        for key in color_map
        if key in present
    ]
    ax.legend(
        handles=handles,
        loc="upper left",
        fontsize=7.2,
        frameon=True,
        framealpha=0.94,
        borderpad=0.24,
        labelspacing=0.18,
        handlelength=1.15,
    )


def _shape_bbox_intersects(bbox) -> bool:
    return not (bbox[2] < XLIM[0] or bbox[0] > XLIM[1] or bbox[3] < YLIM[0] or bbox[1] > YLIM[1])


def _shape_parts_to_patches(shp) -> list[MplPolygon]:
    pts = shp.points
    parts = list(shp.parts) + [len(pts)]
    patches = []
    for start, end in zip(parts[:-1], parts[1:]):
        ring = pts[start:end]
        if len(ring) >= 3:
            patches.append(MplPolygon(ring, closed=True))
    return patches


def _soil_group_label(code: str, grouped: str) -> str:
    if grouped == "Andosol" or code.startswith("D"):
        return "Andosols"
    if code == "I1":
        return "Brown forest soils"
    if code == "C1":
        return "Podzols"
    if grouped == "Immature soil" or code.startswith("J") or code == "K1":
        return "Immature soils"
    return "Other soils"


def _plot_fig6_soil_type(ax, basins, japan):
    soil_module = _module_from_path("soil_type_figure_builder", SOIL_SCRIPT)
    shps = sorted(soil_module.NATIONWIDE_DIR.glob("*.shp"))
    if not shps:
        raise FileNotFoundError(f"No soil-type shapefile found in {soil_module.NATIONWIDE_DIR}")
    reader = shapefile.Reader(str(shps[0]), encoding="utf-8", encodingErrors="ignore")
    fields = [f[0] for f in reader.fields[1:]]
    sg_idx = fields.index("SG_CD")
    class_colors = {
        "Andosols": "#2ca25f",
        "Brown forest soils": "#8c510a",
        "Podzols": "#756bb1",
        "Immature soils": "#ef6548",
        "Other soils": "#bdbdbd",
    }
    patches_by_class = {key: [] for key in class_colors}
    for sr in reader.iterShapeRecords():
        if not _shape_bbox_intersects(sr.shape.bbox):
            continue
        code = str(sr.record[sg_idx]).strip()
        grouped = soil_module.SG_CD_GROUPED.get(code, code)
        label = _soil_group_label(code, grouped)
        patches_by_class[label].extend(_shape_parts_to_patches(sr.shape))
    for label, patches in patches_by_class.items():
        if not patches:
            continue
        coll = PatchCollection(patches, facecolor=class_colors[label], edgecolor="none", linewidth=0, alpha=0.88, zorder=2)
        ax.add_collection(coll)
    _draw_fig6_base(ax, basins, japan)
    handles = [Patch(facecolor=class_colors[k], edgecolor="0.35", linewidth=0.25, label=k) for k in class_colors if patches_by_class[k]]
    ax.legend(
        handles=handles,
        loc="upper left",
        fontsize=7.1,
        frameon=True,
        framealpha=0.90,
        borderpad=0.24,
        labelspacing=0.17,
        handlelength=1.1,
    )


def _plot_soil_depth(ax, basins, japan):
    with rasterio.open(SOIL_DEPTH_RASTER) as src:
        window = from_bounds(XLIM[0], YLIM[0], XLIM[1], YLIM[1], transform=src.transform)
        data = src.read(1, window=window).astype("float32")
        transform = src.window_transform(window)
        nodata = src.nodata
    if nodata is not None:
        data[data == nodata] = np.nan
    data[(data <= 0) | ~np.isfinite(data)] = np.nan
    data = _mask_to_japan(data, transform, japan)
    im = ax.imshow(
        np.ma.masked_invalid(data),
        extent=plotting_extent(data, transform),
        origin="upper",
        aspect="equal",
        cmap="YlOrBr",
        interpolation="nearest",
        vmin=150,
        vmax=200,
        zorder=1,
    )
    _draw_fig6_base(ax, basins, japan)
    cb = plt.colorbar(im, ax=ax, fraction=0.031, pad=0.014)
    cb.set_label("Soil depth (cm)", fontsize=7.2)
    cb.ax.tick_params(labelsize=6.6, length=1.8, width=0.5)


def _plot_dem(ax, basins, japan):
    dem_paths = sorted((WATER_ROOT / "DEM").glob("*.tif"))
    selected = []
    for path in dem_paths:
        with rasterio.open(path) as src:
            b = src.bounds
            if not (b.right < XLIM[0] or b.left > XLIM[1] or b.top < YLIM[0] or b.bottom > YLIM[1]):
                selected.append(path)
    if not selected:
        raise FileNotFoundError("No DEM tiles intersect the Figure 6 extent.")
    datasets = [rasterio.open(path) for path in selected]
    try:
        mosaic, transform = merge(datasets, bounds=(XLIM[0], YLIM[0], XLIM[1], YLIM[1]), res=(0.02, 0.02), nodata=-9999, dtype="float32")
        data = mosaic[0].astype("float32")
    finally:
        for ds in datasets:
            ds.close()
    data[(data <= 0) | (data == -9999) | ~np.isfinite(data)] = np.nan
    data = _mask_to_japan(data, transform, japan)
    im = ax.imshow(
        np.ma.masked_invalid(data),
        extent=plotting_extent(data, transform),
        origin="upper",
        aspect="equal",
        cmap="terrain",
        interpolation="bilinear",
        vmin=0,
        vmax=3700,
        zorder=1,
    )
    _draw_fig6_base(ax, basins, japan)
    cb = plt.colorbar(im, ax=ax, fraction=0.031, pad=0.014)
    cb.set_label("Elevation (m)", fontsize=7.2)
    cb.ax.tick_params(labelsize=6.6, length=1.8, width=0.5)


def save_fig6() -> Path:
    basins = load_basin_geoms()
    japan = load_japan_geoms()
    fig, axes = plt.subplots(2, 2, figsize=(7.85, 7.2), constrained_layout=True)
    axes = axes.ravel()
    _plot_fig6_dominance(axes[0], basins, japan)
    _plot_fig6_soil_type(axes[1], basins, japan)
    _plot_soil_depth(axes[2], basins, japan)
    _plot_dem(axes[3], basins, japan)
    for ax, label, show_xlabel, show_ylabel in zip(
        axes,
        ["(a)", "(b)", "(c)", "(d)"],
        [False, False, True, True],
        [True, False, True, False],
    ):
        _style_fig6_axis(ax, show_xlabel=show_xlabel, show_ylabel=show_ylabel)
        add_panel_label(ax, label)
    out = MAIN_DIR / "figure6_external_plausibility_abcd_clean_wrr_polish_v1.png"
    fig.savefig(out, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)
    return out


def crop_title_band(img: Image.Image, top_px: int) -> Image.Image:
    if top_px <= 0 or top_px >= img.height:
        return img
    return img.crop((0, top_px, img.width, img.height))


def crop_existing_si7() -> list[Path]:
    specs = [
        ("si7_1_signed_groupA_wrr_polish_v1.png", ANALYSIS / "global_analysis/global_feature_importance_signed_aggregated.png", 95),
        ("si7_2a_seasonal_heatmap_wrr_polish_v1.png", ANALYSIS / "temporal_analysis/seasonal_patterns_heatmap.png", 120),
        ("si7_2b_seasonal_top6_wrr_polish_v1.png", ANALYSIS / "temporal_analysis/seasonal_patterns_top6.png", 100),
        ("si7_3_groupC_distribution_wrr_polish_v1.png", NOTE_ROOT / "manuscript_outputs/derived_figures/si7_3_group_c_dominant_attribution_distribution_v1.png", 0),
    ]
    outs = []
    for name, src, top_px in specs:
        if not src.exists():
            continue
        img = crop_to_content(crop_title_band(Image.open(src), top_px), pad=18)
        out = SI_DIR / name
        img.save(out, dpi=(300, 300))
        outs.append(out)
    for basin_id in ["001", "107", "123", "129"]:
        src = ANALYSIS / f"single_basin_extended/basin_{basin_id}/def_summary_three_panels.png"
        if src.exists():
            img = crop_to_content(crop_title_band(Image.open(src), 85), pad=18)
            out = SI_DIR / f"si7_4_basin_{basin_id}_three_panel_wrr_polish_v1.png"
            img.save(out, dpi=(300, 300))
            outs.append(out)
    return outs


def make_contact_sheet(paths: list[Path], out: Path, thumb_w=520) -> None:
    thumbs = []
    for path in paths:
        img = Image.open(path).convert("RGB")
        scale = thumb_w / img.width
        thumb = img.resize((thumb_w, max(1, int(img.height * scale))), Image.Resampling.LANCZOS)
        label_band = 34
        tile = Image.new("RGB", (thumb.width, thumb.height + label_band), "white")
        tile.paste(thumb, (0, label_band))
        draw = ImageDraw.Draw(tile)
        draw.text((8, 7), path.name, fill="black", font=find_font(18, bold=False))
        thumbs.append(tile)
    if not thumbs:
        return
    cols = 2
    rows = math.ceil(len(thumbs) / cols)
    col_w = max(t.width for t in thumbs)
    row_h = max(t.height for t in thumbs)
    canvas = Image.new("RGB", (cols * col_w, rows * row_h), "white")
    for i, tile in enumerate(thumbs):
        x = (i % cols) * col_w
        y = (i // cols) * row_h
        canvas.paste(tile, (x, y))
    out.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out)


def write_notes(paths: dict[str, Path]) -> None:
    notes = [
        "# Runoff-XAI figure quality-control notes",
        "",
        "Generated without rerunning the hydrologic model or IG analysis. This record documents the Figure 4 and Figure 5 build.",
        "",
        "## Outputs",
    ]
    for key, path in paths.items():
        notes.append(f"- {key}: `{path}`")
    notes.extend(
        [
            "",
            "## Source and method notes",
            "- Figure 4 was redrawn from Basin 001 and Basin 123 `detailed_analysis.json` files. Bars are Group C mean unsigned IG contributions in m³/s.",
            "- Figure 4 uses a fixed variable order and a common x-axis scale across the two basins; black whiskers were removed for the main-text comparison.",
            "- Figure 5 was redrawn from the existing `single_basin_extended` CSV outputs. Upper panels show grouped absolute IG share; lower panels show separate top-five daily unsigned IG magnitude line series in m³/s.",
            "- Figure 5 Group F x-axes are trimmed to the first test date through one day after the final test date, preserving the 2004 end tick without large edge blanks.",
            "- Upper Figure 5 panels use low-opacity dashed white guide lines. Lower magnitude panels use thin black dashed guide lines so individual variable series remain readable.",
            "- No model, IG, basin selection, or aggregation values were recomputed manually.",
        ]
    )
    (OUT_DIR / "figure_qc_notes_wrr_polish_v3.md").write_text("\n".join(notes) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Reproduce main-text Figures 4 and 5 from the released dataset.")
    parser.add_argument("--package-root", type=Path, required=True, help="Extracted derived-results package root.")
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def configure(args: argparse.Namespace) -> None:
    global PACKAGE_ROOT, OUT_DIR, MAIN_DIR, GROUP_C, GROUP_DEF
    PACKAGE_ROOT = args.package_root.resolve()
    OUT_DIR = args.output_dir.resolve()
    MAIN_DIR = OUT_DIR
    GROUP_C = PACKAGE_ROOT / "derived_results" / "attribution" / "groupC_basin"
    GROUP_DEF = PACKAGE_ROOT / "derived_results" / "attribution" / "groupD_E_F_daily"


def main() -> None:
    configure(parse_args())
    setup()
    generated: dict[str, Path] = {}
    generated["Figure 4"] = save_fig4()
    generated["Figure 5"] = save_fig5_simplified()
    for key, path in generated.items():
        print(f"{key}: {path}")


if __name__ == "__main__":
    main()
