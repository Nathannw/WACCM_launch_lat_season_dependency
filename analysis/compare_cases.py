#!/usr/bin/env python3
"""Run the standard ID-minus-EQ and BC-only-minus-CC comparisons.

The reported sign convention is always target minus reference. By default the
script runs both built-in comparisons and writes each result to its own data
and figure directories. Passing any directory option switches to a single,
custom comparison.

The script uses paired 12-month means over matching YYYY-MM files starting from
file index [120:] and reports lag-1-autocorrelation-adjusted two-sided p-values
and 95% confidence intervals.
"""

from __future__ import annotations

import argparse
import csv
import re
import sys
from dataclasses import dataclass
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

from analysis.emission_funcs import col_area
from analysis.figure_style import COLORS, panel_label
from analysis.func import (
    Neff_calculation as neff_calculation,
    Neff_calculation_p_val as neff_calculation_p_val,
    find_trop_func as tropopause_bounds,
    pressure_level_weights,
    pressure_weighted_mean,
)

try:
    from netCDF4 import Dataset as NetcdfDataset
except ImportError:  # pragma: no cover - exercised when netCDF4 is unavailable
    NetcdfDataset = None
    from scipy.io import netcdf_file
else:
    netcdf_file = None


PROJECT_DIR = Path(__file__).resolve().parent.parent
COORDS_DIR = PROJECT_DIR / "geo_file"


@dataclass(frozen=True)
class ComparisonConfig:
    reference_dir: Path
    target_dir: Path
    data_dir: Path
    fig_dir: Path


ID_MINUS_EQ_COMPARISON = ComparisonConfig(
    reference_dir=PROJECT_DIR / "results_cloud" / "lon" / "AUG_13_30Gg_all_year_EQ",
    target_dir=PROJECT_DIR / "results_cloud" / "lon" / "AUG_13_all_year_ID",
    data_dir=PROJECT_DIR / "data_file" / "id_minus_eq",
    fig_dir=PROJECT_DIR / "fig" / "id_minus_eq",
)
BC_ONLY_MINUS_CC_COMPARISON = ComparisonConfig(
    reference_dir=PROJECT_DIR / "results_cloud" / "lon" / "JUL_8_30Gg_all_year_CC",
    target_dir=PROJECT_DIR / "results_cloud" / "lon" / "AUG_13_30Gg_all_year_CC_BC_only",
    data_dir=PROJECT_DIR / "data_file" / "BC_only_minus_cc",
    fig_dir=PROJECT_DIR / "fig" / "BC_only_minus_cc",
)
DEFAULT_COMPARISONS = (ID_MINUS_EQ_COMPARISON, BC_ONLY_MINUS_CC_COMPARISON)

DATE_RE = re.compile(r"\.(\d{4})-(\d{2})\.nc$")
N_LONGITUDES = 144
BC_START_LEVEL = 18
TEMP_START_LEVEL = 25
AIR_MOLAR_MASS_G_PER_MOL = 28.97
DU_MOLECULES_PER_CM2 = 4.4615e-8
RADIATIVE_FLUXES = ("FSNT", "FLNT", "FSNTC", "FLNTC")
RADIATIVE_UNITS = "W m^-2"

PLOT_UNITS = {
    "Gg": "Gg",
    "K": "K",
    "DU": "DU",
    RADIATIVE_UNITS: r"W m$^{-2}$",
}


@dataclass
class StatResult:
    mean: float
    ci95: float
    p_value: float
    dof_eff: float
    n_years: int
    significant_95: bool


def open_dataset(path: Path):
    if NetcdfDataset is not None:
        return NetcdfDataset(path)
    return netcdf_file(path, mode="r", mmap=False)


def read_variable(dataset, name: str) -> np.ndarray:
    variable = dataset.variables[name]
    data = np.array(variable[:], dtype=float, copy=True)
    fill_value = getattr(variable, "_FillValue", None)
    missing_value = getattr(variable, "missing_value", None)
    for marker in (fill_value, missing_value):
        if marker is not None:
            marker = float(np.asarray(marker).ravel()[0])
            data[np.isclose(data, marker)] = np.nan
    data[np.abs(data) > 1.0e35] = np.nan
    return data


def date_from_file(path: Path) -> str:
    match = DATE_RE.search(path.name)
    if not match:
        raise ValueError(f"Could not parse YYYY-MM from {path.name}")
    return f"{match.group(1)}-{match.group(2)}"


def case_files_by_date(case_dir: Path) -> dict[str, Path]:
    if not case_dir.exists():
        raise FileNotFoundError(f"Missing case directory: {case_dir}")
    files = sorted(case_dir.glob("*.nc"), key=lambda path: path.name.lower())
    if not files:
        raise FileNotFoundError(f"No NetCDF files found in {case_dir}")
    return {date_from_file(path): path for path in files}


def matching_file_pairs(
    reference_dir: Path, target_dir: Path, skip_months: int
) -> list[tuple[str, Path, Path]]:
    reference = case_files_by_date(reference_dir)
    target = case_files_by_date(target_dir)
    common_dates = sorted(set(reference) & set(target))
    if skip_months:
        common_dates = common_dates[skip_months:]
    if not common_dates:
        raise ValueError(
            "No matching monthly files remain after applying "
            f"skip_months={skip_months}"
        )
    return [(date, reference[date], target[date]) for date in common_dates]


def complete_annual_means(
    dates: list[str], monthly_values: np.ndarray
) -> tuple[list[int], np.ndarray]:
    monthly_values = np.asarray(monthly_values, dtype=float)
    annual_years = []
    annual_values = []
    for start in range(0, len(dates) - len(dates) % 12, 12):
        end = start + 12
        chunk_dates = dates[start:end]
        label_year = int(chunk_dates[-1][:4])
        annual_years.append(label_year)
        annual_values.append(np.nanmean(monthly_values[start:end], axis=0))
    if not annual_values:
        raise ValueError("No complete 12-month windows found in the paired monthly data")
    return annual_years, np.asarray(annual_values, dtype=float)


def summarize_difference(diff: np.ndarray) -> StatResult:
    diff = np.asarray(diff, dtype=float)
    diff = diff[np.isfinite(diff)]
    n = len(diff)
    if n == 0:
        return StatResult(np.nan, np.nan, np.nan, np.nan, 0, False)

    mean = float(np.nanmean(diff))
    if n == 1:
        return StatResult(mean, np.nan, np.nan, np.nan, 1, False)

    rho1 = np.corrcoef(diff[1:], diff[:-1])[0, 1]
    _, dof_eff = neff_calculation(rho1)
    sem = stats.sem(diff)
    if not np.isfinite(sem) or sem == 0.0 or not np.isfinite(dof_eff):
        ci95 = 0.0
        p_value = 1.0 if np.isclose(mean, 0.0) else 0.0
        dof_eff = np.nan
    else:
        confidence = 0.95
        t_crit = stats.t.ppf((1 + confidence) / 2, df=dof_eff)
        ci95 = float(t_crit * sem)
        t_stat = mean / sem
        p_value = float(2 * stats.t.sf(abs(t_stat), dof_eff))
    return StatResult(mean, ci95, p_value, float(dof_eff), n, p_value < 0.05)


def summarize_latitude_difference(diff: np.ndarray) -> list[StatResult]:
    diff = np.asarray(diff, dtype=float)
    t_stat, _ = stats.ttest_1samp(diff, 0, axis=0, nan_policy="omit")
    results = []
    for lat_idx in range(diff.shape[1]):
        x = diff[:, lat_idx]
        x = x[np.isfinite(x)]
        if len(x) <= 1:
            results.append(StatResult(np.nanmean(x), np.nan, np.nan, np.nan, len(x), False))
            continue
        mean = float(np.mean(x))
        sem = stats.sem(x)
        rho1 = np.corrcoef(x[1:], x[:-1])[0, 1]
        _, dof_eff, p_value = neff_calculation_p_val(rho1, t_stat[lat_idx])
        if not np.isfinite(sem) or not np.isfinite(dof_eff):
            ci95 = np.nan
        else:
            ci95 = float(stats.t.ppf(0.975, df=dof_eff) * sem)
        results.append(
            StatResult(mean, ci95, p_value, dof_eff, len(x), p_value < 0.05)
        )
    return results


def summarize_profile_difference(diff: np.ndarray) -> list[StatResult]:
    return summarize_latitude_difference(diff)


def area_weighted_mean(values: np.ndarray, weights: np.ndarray) -> float:
    values = np.ma.masked_invalid(np.ma.array(values))
    weights = np.ma.array(weights, mask=np.ma.getmaskarray(values))
    return float(np.ma.sum(values * weights) / np.ma.sum(weights))


def global_area_weighted_mean(values: np.ndarray, zonal_area: np.ndarray) -> float:
    values = np.ma.masked_invalid(np.ma.array(values))

    if values.shape[-1] == len(zonal_area):
        weights = zonal_area
    elif values.ndim >= 2 and values.shape[-2] == len(zonal_area):
        weights = zonal_area[:, np.newaxis]
    else:
        return float(np.ma.mean(values))

    weights = np.broadcast_to(weights, values.shape)
    weights = np.ma.array(weights, mask=np.ma.getmaskarray(values))
    return float(np.ma.sum(values * weights) / np.ma.sum(weights))


def bc_level_weights(dataset, lat_idx: int) -> np.ndarray:
    nlev = read_variable(dataset, "lev").shape[0]
    weights = np.zeros(nlev - BC_START_LEVEL)
    idx_lower, _, ratio = tropopause_bounds(dataset, lat_idx)
    lower = int(idx_lower[0])
    boundary_fraction = float(ratio[0])

    full_stop = min(max(lower, BC_START_LEVEL), nlev)
    weights[: full_stop - BC_START_LEVEL] = 1.0

    boundary_idx = lower - BC_START_LEVEL
    if 0 <= boundary_idx < len(weights):
        weights[boundary_idx] = boundary_fraction
    return weights


def temperature_by_lat(dataset) -> np.ndarray:
    response_data = read_variable(dataset, "T")
    pressure_weights = pressure_level_weights(dataset)
    nlat = response_data.shape[-1]
    aggregated = []

    for lat_idx in range(nlat):
        idx_lower, idx_upper, ratio = tropopause_bounds(dataset, lat_idx)
        lower = int(idx_lower[0])
        upper = int(idx_upper[0])
        boundary_fraction = float(ratio[0])

        x1 = pressure_weighted_mean(
            response_data[:, TEMP_START_LEVEL:lower, lat_idx],
            pressure_weights[TEMP_START_LEVEL:lower],
        )
        x2 = pressure_weighted_mean(
            response_data[:, TEMP_START_LEVEL:upper, lat_idx],
            pressure_weights[TEMP_START_LEVEL:upper],
        )
        aggregated.append(x1 * (1.0 - boundary_fraction) + x2 * boundary_fraction)

    return np.ma.masked_array(aggregated).filled(np.nan)


def bc_burden_gg(dataset) -> float:
    bc = (
        read_variable(dataset, "bc_a4")[:, BC_START_LEVEL:, :]
        + read_variable(dataset, "bc_a1")[:, BC_START_LEVEL:, :]
    )
    mass = read_variable(dataset, "MASS")[:, BC_START_LEVEL:, :]
    bc = np.ma.masked_invalid(np.ma.array(bc))
    mass = np.ma.masked_invalid(np.ma.array(mass))
    nlat = bc.shape[2]
    total = 0.0

    for lat_idx in range(nlat):
        weights = bc_level_weights(dataset, lat_idx)
        bc_mass_by_level = np.ma.sum(
            bc[:, :, lat_idx] * mass[:, :, lat_idx],
            axis=0,
        )
        total += float(np.ma.sum(bc_mass_by_level * weights))

    return total * N_LONGITUDES / 1.0e6


def ozone_column_du_by_lat(dataset, zonal_area_cm2: np.ndarray) -> np.ndarray:
    mass = np.ma.masked_invalid(np.ma.array(read_variable(dataset, "MASS")[0, :, :]))
    ozone = np.ma.masked_invalid(np.ma.array(read_variable(dataset, "O3")[0, :, :]))
    mol_air = mass * 1000.0 / AIR_MOLAR_MASS_G_PER_MOL
    mol_ozone = ozone * mol_air
    mol_ozone_per_cm2 = np.ma.sum(mol_ozone / zonal_area_cm2, axis=0)
    return np.ma.asarray(mol_ozone_per_cm2 / DU_MOLECULES_PER_CM2).filled(np.nan)


def radiative_flux_global_mean(
    dataset,
    variable_name: str,
    zonal_area_cm2: np.ndarray,
) -> float:
    return global_area_weighted_mean(read_variable(dataset, variable_name), zonal_area_cm2)


def compute_monthly_diagnostics(
    pairs: list[tuple[str, Path, Path]],
) -> dict[str, np.ndarray | list[str]]:
    column_area, lat, _ = col_area(COORDS_DIR)
    zonal_area_cm2 = np.mean(column_area, axis=1)

    dates = []
    bc_ref = []
    bc_target = []
    temp_lat_ref = []
    temp_lat_target = []
    temp_global_ref = []
    temp_global_target = []
    ozone_lat_ref = []
    ozone_lat_target = []
    ozone_global_ref = []
    ozone_global_target = []
    flux_ref = {name: [] for name in RADIATIVE_FLUXES}
    flux_target = {name: [] for name in RADIATIVE_FLUXES}

    for date, reference_path, target_path in pairs:
        with open_dataset(reference_path) as reference, open_dataset(target_path) as target:
            ref_temp_lat = temperature_by_lat(reference)
            target_temp_lat = temperature_by_lat(target)
            ref_ozone_lat = ozone_column_du_by_lat(reference, zonal_area_cm2)
            target_ozone_lat = ozone_column_du_by_lat(target, zonal_area_cm2)
            ref_flux = {
                name: radiative_flux_global_mean(reference, name, zonal_area_cm2)
                for name in RADIATIVE_FLUXES
            }
            target_flux = {
                name: radiative_flux_global_mean(target, name, zonal_area_cm2)
                for name in RADIATIVE_FLUXES
            }

            dates.append(date)
            bc_ref.append(bc_burden_gg(reference))
            bc_target.append(bc_burden_gg(target))
            temp_lat_ref.append(ref_temp_lat)
            temp_lat_target.append(target_temp_lat)
            temp_global_ref.append(global_area_weighted_mean(ref_temp_lat, zonal_area_cm2))
            temp_global_target.append(global_area_weighted_mean(target_temp_lat, zonal_area_cm2))
            ozone_lat_ref.append(ref_ozone_lat)
            ozone_lat_target.append(target_ozone_lat)
            ozone_global_ref.append(area_weighted_mean(ref_ozone_lat, zonal_area_cm2))
            ozone_global_target.append(area_weighted_mean(target_ozone_lat, zonal_area_cm2))
            for name in RADIATIVE_FLUXES:
                flux_ref[name].append(ref_flux[name])
                flux_target[name].append(target_flux[name])

    diagnostics = {
        "dates": dates,
        "lat": lat,
        "bc_ref": np.asarray(bc_ref),
        "bc_target": np.asarray(bc_target),
        "temp_lat_ref": np.asarray(temp_lat_ref),
        "temp_lat_target": np.asarray(temp_lat_target),
        "temp_global_ref": np.asarray(temp_global_ref),
        "temp_global_target": np.asarray(temp_global_target),
        "ozone_lat_ref": np.asarray(ozone_lat_ref),
        "ozone_lat_target": np.asarray(ozone_lat_target),
        "ozone_global_ref": np.asarray(ozone_global_ref),
        "ozone_global_target": np.asarray(ozone_global_target),
    }
    for name in RADIATIVE_FLUXES:
        diagnostics[f"{name}_ref"] = np.asarray(flux_ref[name])
        diagnostics[f"{name}_target"] = np.asarray(flux_target[name])
    return diagnostics


def write_scalar_summary(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "diagnostic",
        "units",
        "reference_mean",
        "target_mean",
        "target_minus_reference_mean",
        "ci95",
        "p_value",
        "significant_95",
        "n_years",
        "dof_eff",
    ]
    with path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_latitude_summary(
    path: Path,
    lat: np.ndarray,
    units: str,
    reference_mean: np.ndarray,
    target_mean: np.ndarray,
    diff_stats: list[StatResult],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(
            [
                "latitude_deg_north",
                "units",
                "reference_mean",
                "target_mean",
                "target_minus_reference_mean",
                "ci95",
                "p_value",
                "significant_95",
                "n_years",
                "dof_eff",
            ]
        )
        for latitude, ref, target, stat in zip(lat, reference_mean, target_mean, diff_stats):
            writer.writerow(
                [
                    f"{latitude:.10g}",
                    units,
                    f"{ref:.10g}",
                    f"{target:.10g}",
                    f"{stat.mean:.10g}",
                    f"{stat.ci95:.10g}",
                    f"{stat.p_value:.10g}",
                    int(stat.significant_95),
                    stat.n_years,
                    f"{stat.dof_eff:.10g}",
                ]
            )


def format_plot_units(units: str) -> str:
    return PLOT_UNITS.get(units, units)


def style_journal_axis(ax) -> None:
    for spine in ax.spines.values():
        spine.set_linewidth(1.0)
    ax.tick_params(axis="both", which="major", labelsize=13, width=1.0)
    ax.tick_params(axis="both", which="minor", width=1.0)


def plot_significance_weighted_line(
    ax,
    x: np.ndarray,
    y: np.ndarray,
    significant: np.ndarray,
    color: str,
) -> None:
    valid = np.isfinite(x) & np.isfinite(y)
    for idx in range(len(x) - 1):
        if not valid[idx] or not valid[idx + 1]:
            continue
        is_significant_segment = significant[idx] or significant[idx + 1]
        ax.plot(
            x[idx : idx + 2],
            y[idx : idx + 2],
            color=color,
            linewidth=2.8 if is_significant_segment else 1.1,
            alpha=0.95 if is_significant_segment else 0.55,
            solid_capstyle="round",
            zorder=3,
        )


def plot_latitude_difference(
    path_base: Path,
    lat: np.ndarray,
    stats_by_lat: list[StatResult],
    xlabel: str,
    color: str,
    label: str,
) -> None:
    path_base.parent.mkdir(parents=True, exist_ok=True)
    mean = np.array([item.mean for item in stats_by_lat])
    ci95 = np.array([item.ci95 for item in stats_by_lat])
    significant = np.array([item.significant_95 for item in stats_by_lat])

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.axvline(0.0, color=COLORS["gray"], linewidth=1.0, alpha=0.75)
    ax.fill_betweenx(
        lat,
        mean - ci95,
        mean + ci95,
        color=color,
        alpha=0.18,
        linewidth=0,
        label="95% CI",
    )
    plot_significance_weighted_line(ax, mean, lat, significant, color)
    ax.set_xlabel(xlabel, fontsize=15)
    ax.set_ylabel("Latitude (°)", fontsize=15)
    ax.set_ylim(float(np.nanmin(lat)), float(np.nanmax(lat)))
    ax.text(
        0.02,
        1.05,
        label,
        transform=ax.transAxes,
        fontsize=15,
        fontweight="bold",
    )
    style_journal_axis(ax)
    ax.legend(
        handles=[
            plt.Rectangle((0, 0), 1, 1, facecolor=color, alpha=0.18, edgecolor="none", label="95% CI"),
            # plt.Line2D([0], [0], color=color, linewidth=2.8, label="p < 0.05"),
            # plt.Line2D([0], [0], color=color, linewidth=1.1, alpha=0.55, label="p >= 0.05"),
        ],
        frameon=False,
        fontsize=11,
        loc="best",
    )
    fig.tight_layout()
    fig.savefig(path_base.with_suffix(".pdf"))
    # fig.savefig(path_base.with_suffix(".png"), dpi=300)
    plt.close(fig)


def plot_scalar_summary(path_base: Path, rows: list[dict[str, object]]) -> None:
    path_base.parent.mkdir(parents=True, exist_ok=True)
    ncols = 3
    nrows = int(np.ceil(len(rows) / ncols))
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(10.5, 3.2 * nrows),
    )
    axes = np.atleast_1d(axes).ravel()

    for idx, (ax, row) in enumerate(zip(axes, rows)):
        mean = float(row["target_minus_reference_mean"])
        ci95 = float(row["ci95"])
        significant = bool(row["significant_95"])
        color = COLORS["blue"] if significant else COLORS["gray"]
        lower = mean - ci95
        upper = mean + ci95
        span = max(abs(lower), abs(upper), abs(mean), 1.0e-12)
        padding = 0.22 * span

        ax.axhline(0.0, color=COLORS["gray"], linewidth=1.0, alpha=0.75)
        ax.errorbar(
            [0],
            [mean],
            yerr=[[ci95], [ci95]],
            fmt="o",
            color=color,
            ecolor=color,
            elinewidth=1.2,
            capsize=3,
            markersize=6.5,
            markeredgecolor="black" if significant else color,
            markeredgewidth=0.7,
            zorder=3,
            alpha=0.9,
        )
        ax.set_title(str(row["diagnostic"]), fontsize=12, pad=8)
        ax.set_xticks([])
        ax.set_ylabel(f"Difference ({format_plot_units(str(row['units']))})", fontsize=12)
        ax.set_xlim(-0.45, 0.45)
        ax.set_ylim(min(lower, 0.0) - padding, max(upper, 0.0) + padding)
        ax.text(
            0.02,
            0.95,
            panel_label(chr(ord("a") + idx)),
            transform=ax.transAxes,
            fontsize=13,
            fontweight="bold",
            va="top",
        )
        ax.text(
            0.98,
            0.94,
            f"{mean:.3g} +/- {ci95:.3g}\np={float(row['p_value']):.3g}",
            ha="right",
            va="top",
            transform=ax.transAxes,
            fontsize=10,
        )
        style_journal_axis(ax)
    for ax in axes[len(rows) :]:
        ax.set_visible(False)
    fig.tight_layout()
    fig.savefig(path_base.with_suffix(".pdf"))
    #fig.savefig(path_base.with_suffix(".png"), dpi=300)
    plt.close(fig)


def stat_row(
    diagnostic: str,
    units: str,
    reference_annual: np.ndarray,
    target_annual: np.ndarray,
) -> dict[str, object]:
    diff = target_annual - reference_annual
    stat = summarize_difference(diff)
    return {
        "diagnostic": diagnostic,
        "units": units,
        "reference_mean": f"{np.nanmean(reference_annual):.10g}",
        "target_mean": f"{np.nanmean(target_annual):.10g}",
        "target_minus_reference_mean": f"{stat.mean:.10g}",
        "ci95": f"{stat.ci95:.10g}",
        "p_value": f"{stat.p_value:.10g}",
        "significant_95": int(stat.significant_95),
        "n_years": stat.n_years,
        "dof_eff": f"{stat.dof_eff:.10g}",
    }


def comparison_label(reference_dir: Path, target_dir: Path) -> str:
    return f"{target_dir.name} - {reference_dir.name}"


def print_scalar_summary(rows: list[dict[str, object]]) -> None:
    print("Global summary values, mean +/- 95% CI:")
    for row in rows:
        mean = float(row["target_minus_reference_mean"])
        ci95 = float(row["ci95"])
        lower = mean - ci95
        upper = mean + ci95
        p_value = float(row["p_value"])
        print(
            f"  {row['diagnostic']}: {mean:.3f} +/- {ci95:.3f} {row['units']} "
            f"(95% CI: [{lower:.3f}, {upper:.3f}], p={p_value:.3f})"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run both standard target-minus-reference comparisons. Passing any "
            "directory option runs one custom comparison instead."
        )
    )
    parser.add_argument("--reference-dir", type=Path)
    parser.add_argument("--target-dir", type=Path)
    parser.add_argument("--data-dir", type=Path)
    parser.add_argument("--fig-dir", type=Path)
    parser.add_argument(
        "--skip-months",
        type=int,
        default=120,
        help="Number of initial matched months to skip before annual averaging.",
    )
    return parser.parse_args()


def requested_comparisons(args: argparse.Namespace) -> tuple[ComparisonConfig, ...]:
    custom_paths = (
        args.reference_dir,
        args.target_dir,
        args.data_dir,
        args.fig_dir,
    )
    if not any(path is not None for path in custom_paths):
        return DEFAULT_COMPARISONS

    base = ID_MINUS_EQ_COMPARISON
    return (
        ComparisonConfig(
            reference_dir=args.reference_dir or base.reference_dir,
            target_dir=args.target_dir or base.target_dir,
            data_dir=args.data_dir or base.data_dir,
            fig_dir=args.fig_dir or base.fig_dir,
        ),
    )


def run_comparison(config: ComparisonConfig, skip_months: int) -> None:
    pairs = matching_file_pairs(config.reference_dir, config.target_dir, skip_months)
    diagnostics = compute_monthly_diagnostics(pairs=pairs)

    dates = list(diagnostics["dates"])
    lat = np.asarray(diagnostics["lat"], dtype=float)

    years, bc_ref_annual = complete_annual_means(dates, diagnostics["bc_ref"])
    _, bc_target_annual = complete_annual_means(dates, diagnostics["bc_target"])
    _, temp_global_ref_annual = complete_annual_means(dates, diagnostics["temp_global_ref"])
    _, temp_global_target_annual = complete_annual_means(dates, diagnostics["temp_global_target"])
    _, ozone_global_ref_annual = complete_annual_means(dates, diagnostics["ozone_global_ref"])
    _, ozone_global_target_annual = complete_annual_means(dates, diagnostics["ozone_global_target"])
    _, temp_lat_ref_annual = complete_annual_means(dates, diagnostics["temp_lat_ref"])
    _, temp_lat_target_annual = complete_annual_means(dates, diagnostics["temp_lat_target"])
    _, ozone_lat_ref_annual = complete_annual_means(dates, diagnostics["ozone_lat_ref"])
    _, ozone_lat_target_annual = complete_annual_means(dates, diagnostics["ozone_lat_target"])
    flux_ref_annual = {}
    flux_target_annual = {}
    for name in RADIATIVE_FLUXES:
        _, flux_ref_annual[name] = complete_annual_means(dates, diagnostics[f"{name}_ref"])
        _, flux_target_annual[name] = complete_annual_means(
            dates,
            diagnostics[f"{name}_target"],
        )
    erf_ref_annual = flux_ref_annual["FSNT"] - flux_ref_annual["FLNT"]
    erf_target_annual = flux_target_annual["FSNT"] - flux_target_annual["FLNT"]
    erfc_ref_annual = flux_ref_annual["FSNTC"] - flux_ref_annual["FLNTC"]
    erfc_target_annual = flux_target_annual["FSNTC"] - flux_target_annual["FLNTC"]

    scalar_rows = [
        stat_row("BC burden", "Gg", bc_ref_annual, bc_target_annual),
        stat_row(
            "Stratospheric temperature global",
            "K",
            temp_global_ref_annual,
            temp_global_target_annual,
        ),
        stat_row(
            "O3 column global",
            "DU",
            ozone_global_ref_annual,
            ozone_global_target_annual,
        ),
        stat_row(
            "FSNT global",
            RADIATIVE_UNITS,
            flux_ref_annual["FSNT"],
            flux_target_annual["FSNT"],
        ),
        stat_row(
            "FLNT global",
            RADIATIVE_UNITS,
            flux_ref_annual["FLNT"],
            flux_target_annual["FLNT"],
        ),
        stat_row(
            "FSNTC global",
            RADIATIVE_UNITS,
            flux_ref_annual["FSNTC"],
            flux_target_annual["FSNTC"],
        ),
        stat_row(
            "FLNTC global",
            RADIATIVE_UNITS,
            flux_ref_annual["FLNTC"],
            flux_target_annual["FLNTC"],
        ),
        stat_row("ERF global", RADIATIVE_UNITS, erf_ref_annual, erf_target_annual),
        stat_row(
            "ERFC global",
            RADIATIVE_UNITS,
            erfc_ref_annual,
            erfc_target_annual,
        ),
    ]

    temp_lat_stats = summarize_profile_difference(temp_lat_target_annual - temp_lat_ref_annual)
    ozone_lat_stats = summarize_profile_difference(ozone_lat_target_annual - ozone_lat_ref_annual)

    config.data_dir.mkdir(parents=True, exist_ok=True)
    config.fig_dir.mkdir(parents=True, exist_ok=True)

    write_scalar_summary(config.data_dir / "global_and_burden_summary.csv", scalar_rows)
    write_latitude_summary(
        config.data_dir / "stratospheric_temperature_by_latitude.csv",
        lat,
        "K",
        np.nanmean(temp_lat_ref_annual, axis=0),
        np.nanmean(temp_lat_target_annual, axis=0),
        temp_lat_stats,
    )
    write_latitude_summary(
        config.data_dir / "o3_column_by_latitude.csv",
        lat,
        "DU",
        np.nanmean(ozone_lat_ref_annual, axis=0),
        np.nanmean(ozone_lat_target_annual, axis=0),
        ozone_lat_stats,
    )

    plot_scalar_summary(config.fig_dir / "global_and_burden_summary", scalar_rows)
    plot_latitude_difference(
        config.fig_dir / "stratospheric_temperature_by_latitude",
        lat,
        temp_lat_stats,
        "Stratospheric temperature difference (K)",
        COLORS["gray"],
        panel_label("a"),
    )
    plot_latitude_difference(
        config.fig_dir / "o3_column_by_latitude",
        lat,
        ozone_lat_stats,
        "O3 column difference (DU)",
        COLORS["gray"],
        panel_label("b"),
    )

    print(f"Comparison: {comparison_label(config.reference_dir, config.target_dir)}")
    print(f"Skipped initial matched monthly files: {skip_months}")
    print(f"Matched monthly files: {len(pairs)}")
    print(f"Complete 12-month windows used: {years[0]}-{years[-1]} ({len(years)} windows)")
    print_scalar_summary(scalar_rows)
    print(f"Saved: {config.data_dir / 'global_and_burden_summary.csv'}")
    print(f"Saved: {config.data_dir / 'stratospheric_temperature_by_latitude.csv'}")
    print(f"Saved: {config.data_dir / 'o3_column_by_latitude.csv'}")
    print(f"Saved figures under: {config.fig_dir}")


def main() -> None:
    args = parse_args()
    comparisons = requested_comparisons(args)
    for index, config in enumerate(comparisons):
        if index:
            print()
        run_comparison(config, args.skip_months)


if __name__ == "__main__":
    main()
