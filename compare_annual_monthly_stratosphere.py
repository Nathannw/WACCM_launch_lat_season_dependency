#!/usr/bin/env python3
"""Compare legacy annual and current monthly 30 Gg climate responses.

The sign convention is case minus control for both data sources. Temperature
uses the pressure-weighted stratospheric diagnostic from ``analysis/`` with a
fixed tropopause boundary at model index 52 for every run.

Outputs:
    data_file/annual_vs_monthly_stratosphere/global_anomalies.csv
    data_file/annual_vs_monthly_stratosphere/annual_minus_monthly_global.csv
    data_file/annual_vs_monthly_stratosphere/latitude_anomalies.csv
    fig/annual_vs_monthly_stratosphere/latitude_anomalies.pdf
    fig/annual_vs_monthly_stratosphere/latitude_anomalies.png
    fig/annual_vs_monthly_stratosphere/annual_minus_monthly_latitude.pdf
    fig/annual_vs_monthly_stratosphere/annual_minus_monthly_latitude.png
"""

from __future__ import annotations

import csv
import re
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch
from scipy import stats

from analysis.figure_style import COLORS, panel_label


PROJECT_DIR = Path(__file__).resolve().parent
CONTROL_FILE = PROJECT_DIR / (
    "FWma2000climo_default_case_fixed.cam.2000-2049.annual.concat_IDL_vars.nc"
)
CASE_FILE = PROJECT_DIR / (
    "FWma2000climo_f19_f19_mg17_BC_30Gg_case_fixed_num_err."
    "cam.2000-2049.annual.concat_IDL_vars.nc"
)
MONTHLY_CONTROL_DIR = PROJECT_DIR / "results_cloud" / "lon" / "JUL_8_control"
# MONTHLY_CASE_DIR = (
#     PROJECT_DIR / "results_cloud" / "lon" / "JUL_8_30Gg_all_year_CC"
# )
MONTHLY_CASE_DIR = (
    PROJECT_DIR / "results_cloud" / "lon" / "AUG_13_30Gg_all_year_CC_BC_only"
)
GW_FILE = PROJECT_DIR / "geo_file" / "coords_1.9x2.5_L88_c150828.nc"
DATA_DIR = PROJECT_DIR / "data_file" / "annual_vs_monthly_stratosphere"
FIG_DIR = PROJECT_DIR / "fig" / "annual_vs_monthly_stratosphere"

ANNUAL_START_YEAR_INDEX = 10
ANNUAL_END_YEAR_INDEX = 24
MONTHLY_SPINUP_FILES = 120
TEMP_START_LEVEL = 25
FIXED_TROPOPAUSE_INDEX = 52
RADIATIVE_FLUXES = ("FSNT", "FLNT", "FSNTC", "FLNTC")
METRIC_UNITS = {
    "Stratospheric temperature": "K",
    "O3 column": "DU",
    "FSNT": "W m-2",
    "FLNT": "W m-2",
    "FSNTC": "W m-2",
    "FLNTC": "W m-2",
    "ERF": "W m-2",
    "ERFC": "W m-2",
}
DATE_RE = re.compile(r"\.(\d{4})-(\d{2})\.nc$")

G = 9.80665
MOLAR_MASS_DRY_AIR_KG = 0.02897
MOLAR_MASS_DRY_AIR_G = 28.97
DU_IN_MOL_PER_CM2 = 4.4615e-8


class Dataset:
    """Open classic NetCDF with scipy, falling back to netCDF4 if available."""

    def __init__(self, path: Path):
        self.path = Path(path)
        self._dataset = None

    def __enter__(self):
        try:
            from scipy.io import netcdf_file

            self._dataset = netcdf_file(self.path, mode="r", mmap=False)
        except Exception:
            from netCDF4 import Dataset as NetCDFDataset

            self._dataset = NetCDFDataset(self.path)
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self._dataset.close()

    @property
    def variables(self):
        return self._dataset.variables


def read_variable(dataset: Dataset, name: str, key=slice(None)) -> np.ndarray:
    variable = dataset.variables[name]
    data = np.array(variable[key], dtype=float, copy=True)
    for attribute in ("_FillValue", "missing_value"):
        marker = getattr(variable, attribute, None)
        if marker is not None:
            marker = float(np.asarray(marker).ravel()[0])
            data[np.isclose(data, marker)] = np.nan
    data[np.abs(data) > 1.0e35] = np.nan
    return data


def latitude_weights() -> np.ndarray:
    with Dataset(GW_FILE) as dataset:
        weights = read_variable(dataset, "gw")
    return weights / np.nansum(weights)


def weighted_mean(values: np.ndarray, weights: np.ndarray, axis=None):
    values = np.asarray(values, dtype=float)
    weights = np.broadcast_to(np.asarray(weights, dtype=float), values.shape)
    valid = np.isfinite(values) & np.isfinite(weights)
    numerator = np.sum(np.where(valid, values * weights, 0.0), axis=axis)
    denominator = np.sum(np.where(valid, weights, 0.0), axis=axis)
    return numerator / denominator


def pressure_level_weights(dataset: Dataset) -> np.ndarray:
    lev = read_variable(dataset, "lev")
    if "ilev" in dataset.variables:
        interfaces = read_variable(dataset, "ilev")
    else:
        midpoints = 0.5 * (lev[:-1] + lev[1:])
        interfaces = np.concatenate(
            ([2.0 * lev[0] - midpoints[0]], midpoints, [2.0 * lev[-1] - midpoints[-1]])
        )
    return np.abs(np.diff(interfaces))


def stratospheric_temperature_by_latitude(
    temperature: np.ndarray,
    level_weights: np.ndarray,
) -> np.ndarray:
    """Pressure-weight temperature from index 25 to the fixed index 52 boundary."""
    temperature = np.asarray(temperature, dtype=float)
    if temperature.ndim == 3:
        temperature = np.nanmean(temperature, axis=0)
    if temperature.ndim != 2 or temperature.shape[0] != len(level_weights):
        raise ValueError(
            "Temperature must have shape (lev, lat) after zonal/time averaging;"
            f"got {temperature.shape}."
        )
    if not TEMP_START_LEVEL < FIXED_TROPOPAUSE_INDEX <= temperature.shape[0]:
        raise ValueError(
            f"Invalid fixed temperature range "
            f"[{TEMP_START_LEVEL}:{FIXED_TROPOPAUSE_INDEX}]."
        )

    level_slice = slice(TEMP_START_LEVEL, FIXED_TROPOPAUSE_INDEX)
    weights = level_weights[level_slice, np.newaxis]
    return weighted_mean(temperature[level_slice, :], weights, axis=0)


def annual_ozone_column_by_latitude(
    ozone: np.ndarray, pdel_dry: np.ndarray
) -> np.ndarray:
    mol_o3_per_m2 = ozone * pdel_dry / (G * MOLAR_MASS_DRY_AIR_KG)
    column_du = np.nansum(mol_o3_per_m2, axis=0) / 1.0e4 / DU_IN_MOL_PER_CM2
    return np.nanmean(column_du, axis=-1)


def monthly_ozone_column_by_latitude(
    ozone: np.ndarray, mass: np.ndarray, area_fraction: np.ndarray
) -> np.ndarray:
    # MASS is the zonal-mean mass of one longitude grid box.  Grid-cell area is
    # proportional to the Gaussian latitude weight; the common area constant
    # cancels only for global weighting, so recover the physical area here.
    earth_radius_cm = 637.8e6
    n_longitudes = 144
    grid_area_cm2 = 4.0 * np.pi * earth_radius_cm**2 * area_fraction / n_longitudes
    mol_air = mass * 1000.0 / MOLAR_MASS_DRY_AIR_G
    mol_o3_per_cm2 = np.nansum(ozone * mol_air / grid_area_cm2, axis=0)
    return mol_o3_per_cm2 / DU_IN_MOL_PER_CM2


def file_date(path: Path) -> str:
    match = DATE_RE.search(path.name)
    if not match:
        raise ValueError(f"Cannot parse YYYY-MM from {path.name}")
    return f"{match.group(1)}-{match.group(2)}"


def matching_monthly_files() -> list[tuple[str, Path, Path]]:
    control = {file_date(path): path for path in MONTHLY_CONTROL_DIR.glob("*.nc")}
    case = {file_date(path): path for path in MONTHLY_CASE_DIR.glob("*.nc")}
    dates = sorted(set(control) & set(case))[MONTHLY_SPINUP_FILES:]
    if not dates:
        raise ValueError("No monthly pairs remain after the spinup exclusion.")
    return [(date, control[date], case[date]) for date in dates]


def complete_annual_means(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    n_months = len(values) - len(values) % 12
    if n_months == 0:
        raise ValueError("At least 12 monthly values are required.")
    return np.nanmean(values[:n_months].reshape(-1, 12, *values.shape[1:]), axis=1)


def empty_diagnostics() -> dict[str, list[np.ndarray | float]]:
    return {
        "temperature_lat": [],
        "temperature_global": [],
        "ozone_lat": [],
        "ozone_global": [],
        **{name: [] for name in RADIATIVE_FLUXES},
    }


def append_diagnostics(
    destination: dict[str, list],
    temperature_lat: np.ndarray,
    ozone_lat: np.ndarray,
    fluxes: dict[str, float],
    area_fraction: np.ndarray,
) -> None:
    destination["temperature_lat"].append(temperature_lat)
    destination["temperature_global"].append(
        float(weighted_mean(temperature_lat, area_fraction))
    )
    destination["ozone_lat"].append(ozone_lat)
    destination["ozone_global"].append(float(weighted_mean(ozone_lat, area_fraction)))
    for name, value in fluxes.items():
        destination[name].append(value)


def arrays(diagnostics: dict[str, list]) -> dict[str, np.ndarray]:
    return {name: np.asarray(values, dtype=float) for name, values in diagnostics.items()}


def annual_diagnostics(
    area_fraction: np.ndarray,
) -> tuple[np.ndarray, dict[str, np.ndarray], dict[str, np.ndarray]]:
    control_output = empty_diagnostics()
    case_output = empty_diagnostics()

    with Dataset(CONTROL_FILE) as control, Dataset(CASE_FILE) as case:
        control_lat = read_variable(control, "lat")
        case_lat = read_variable(case, "lat")
        if not np.allclose(control_lat, case_lat):
            raise ValueError("Annual files do not share the same latitude coordinate.")
        control_lev = read_variable(control, "lev")
        case_lev = read_variable(case, "lev")
        if not np.allclose(control_lev, case_lev):
            raise ValueError("Annual files do not share the same level coordinate.")
        control_level_weights = pressure_level_weights(control)
        case_level_weights = pressure_level_weights(case)

        n_control = control.variables["time"].shape[0]
        n_case = case.variables["time"].shape[0]
        if min(n_control, n_case) <= ANNUAL_END_YEAR_INDEX:
            raise ValueError(
                "Annual files do not contain the requested simulation years "
                f"{ANNUAL_START_YEAR_INDEX}-{ANNUAL_END_YEAR_INDEX}."
            )
        annual_slice = slice(ANNUAL_START_YEAR_INDEX, ANNUAL_END_YEAR_INDEX + 1)
        control_indices = range(ANNUAL_START_YEAR_INDEX, ANNUAL_END_YEAR_INDEX + 1)
        case_indices = range(ANNUAL_START_YEAR_INDEX, ANNUAL_END_YEAR_INDEX + 1)
        control_years = read_variable(control, "date", annual_slice).astype(int) // 10000
        case_years = read_variable(case, "date", annual_slice).astype(int) // 10000
        if not np.array_equal(control_years, case_years):
            raise ValueError("Annual control and case years do not align.")

        for control_index, case_index in zip(control_indices, case_indices):
            for dataset, time_index, level_weights, destination in (
                (
                    control,
                    control_index,
                    control_level_weights,
                    control_output,
                ),
                (
                    case,
                    case_index,
                    case_level_weights,
                    case_output,
                ),
            ):
                temperature = read_variable(
                    dataset, "T", (time_index, slice(None), slice(None), slice(None))
                )
                temperature_lat = stratospheric_temperature_by_latitude(
                    np.nanmean(temperature, axis=-1),
                    level_weights,
                )
                ozone_lat = annual_ozone_column_by_latitude(
                    read_variable(
                        dataset,
                        "O3",
                        (time_index, slice(None), slice(None), slice(None)),
                    ),
                    read_variable(
                        dataset,
                        "PDELDRY",
                        (time_index, slice(None), slice(None), slice(None)),
                    ),
                )
                fluxes = {}
                for name in RADIATIVE_FLUXES:
                    flux_lat = np.nanmean(
                        read_variable(
                            dataset,
                            name,
                            (time_index, slice(None), slice(None)),
                        ),
                        axis=-1,
                    )
                    fluxes[name] = float(weighted_mean(flux_lat, area_fraction))
                append_diagnostics(
                    destination, temperature_lat, ozone_lat, fluxes, area_fraction
                )

    return control_years, arrays(control_output), arrays(case_output)


def monthly_diagnostics(
    pairs: list[tuple[str, Path, Path]], area_fraction: np.ndarray
) -> tuple[list[str], dict[str, np.ndarray], dict[str, np.ndarray]]:
    control_output = empty_diagnostics()
    case_output = empty_diagnostics()
    dates = []

    for date, control_path, case_path in pairs:
        with Dataset(control_path) as control, Dataset(case_path) as case:
            control_lev = read_variable(control, "lev")
            case_lev = read_variable(case, "lev")
            if not np.allclose(control_lev, case_lev):
                raise ValueError(f"Monthly level grids differ at {date}.")

            for dataset, destination in (
                (control, control_output),
                (case, case_output),
            ):
                temperature_lat = stratospheric_temperature_by_latitude(
                    read_variable(dataset, "T", (0, slice(None), slice(None))),
                    pressure_level_weights(dataset),
                )
                ozone_lat = monthly_ozone_column_by_latitude(
                    read_variable(dataset, "O3", (0, slice(None), slice(None))),
                    read_variable(dataset, "MASS", (0, slice(None), slice(None))),
                    area_fraction,
                )
                fluxes = {
                    name: float(
                        weighted_mean(
                            read_variable(dataset, name, (0, slice(None))),
                            area_fraction,
                        )
                    )
                    for name in RADIATIVE_FLUXES
                }
                append_diagnostics(
                    destination, temperature_lat, ozone_lat, fluxes, area_fraction
                )
            dates.append(date)

    control = {name: complete_annual_means(value) for name, value in arrays(control_output).items()}
    case = {name: complete_annual_means(value) for name, value in arrays(case_output).items()}
    return dates, control, case


@dataclass
class DifferenceStats:
    mean: float
    ci95: float
    p_value: float
    dof_eff: float
    n: int


def difference_stats(values: np.ndarray) -> DifferenceStats:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    n = len(values)
    if n == 0:
        return DifferenceStats(np.nan, np.nan, np.nan, np.nan, 0)
    mean = float(np.mean(values))
    if n == 1 or np.isclose(np.std(values), 0.0):
        return DifferenceStats(mean, 0.0, np.nan, float(max(n - 1, 0)), n)

    rho1 = np.corrcoef(values[1:], values[:-1])[0, 1]
    if np.isfinite(rho1) and rho1 > 0.0:
        effective_n = np.clip(n * (1.0 - rho1) / (1.0 + rho1), 3.0, float(n))
    else:
        effective_n = float(n)
    dof_eff = effective_n - 1.0
    sem = float(stats.sem(values))
    ci95 = float(stats.t.ppf(0.975, dof_eff) * sem)
    p_value = float(2.0 * stats.t.sf(abs(mean / sem), dof_eff))
    return DifferenceStats(mean, ci95, p_value, dof_eff, n)


def derived_series(diagnostics: dict[str, np.ndarray], clear_sky: bool) -> np.ndarray:
    shortwave = "FSNTC" if clear_sky else "FSNT"
    longwave = "FLNTC" if clear_sky else "FLNT"
    return diagnostics[shortwave] - diagnostics[longwave]


def metric_series(diagnostics: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    return {
        "Stratospheric temperature": diagnostics["temperature_global"],
        "O3 column": diagnostics["ozone_global"],
        "FSNT": diagnostics["FSNT"],
        "FLNT": diagnostics["FLNT"],
        "FSNTC": diagnostics["FSNTC"],
        "FLNTC": diagnostics["FLNTC"],
        "ERF": derived_series(diagnostics, clear_sky=False),
        "ERFC": derived_series(diagnostics, clear_sky=True),
    }


def write_global_csv(
    comparisons: list[tuple[str, dict[str, np.ndarray], dict[str, np.ndarray]]]
) -> Path:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    path = DATA_DIR / "global_anomalies.csv"
    with path.open("w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(
            [
                "data_source",
                "metric",
                "units",
                "control_mean",
                "case_mean",
                "case_minus_control",
                "ci95",
                "p_value",
                "n_years",
                "dof_eff",
            ]
        )
        for label, control, case in comparisons:
            control_metrics = metric_series(control)
            case_metrics = metric_series(case)
            for metric in control_metrics:
                result = difference_stats(case_metrics[metric] - control_metrics[metric])
                writer.writerow(
                    [
                        label,
                        metric,
                        METRIC_UNITS[metric],
                        f"{np.nanmean(control_metrics[metric]):.10g}",
                        f"{np.nanmean(case_metrics[metric]):.10g}",
                        f"{result.mean:.10g}",
                        f"{result.ci95:.10g}",
                        f"{result.p_value:.10g}",
                        result.n,
                        f"{result.dof_eff:.10g}",
                    ]
                )
    return path


def global_annual_minus_monthly_results(
    annual_control: dict[str, np.ndarray],
    annual_case: dict[str, np.ndarray],
    monthly_control: dict[str, np.ndarray],
    monthly_case: dict[str, np.ndarray],
) -> dict[str, tuple[float, float, DifferenceStats]]:
    annual_control_metrics = metric_series(annual_control)
    annual_case_metrics = metric_series(annual_case)
    monthly_control_metrics = metric_series(monthly_control)
    monthly_case_metrics = metric_series(monthly_case)
    results = {}

    for metric in annual_control_metrics:
        annual_anomaly = annual_case_metrics[metric] - annual_control_metrics[metric]
        monthly_anomaly = monthly_case_metrics[metric] - monthly_control_metrics[metric]
        if annual_anomaly.shape != monthly_anomaly.shape:
            raise ValueError(
                f"Annual and monthly {metric} anomalies must have matching "
                f"shapes for a paired comparison; got {annual_anomaly.shape} "
                f"and {monthly_anomaly.shape}."
            )
        results[metric] = (
            float(np.nanmean(annual_anomaly)),
            float(np.nanmean(monthly_anomaly)),
            difference_stats(annual_anomaly - monthly_anomaly),
        )
    return results


def write_global_difference_csv(
    results: dict[str, tuple[float, float, DifferenceStats]],
) -> Path:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    path = DATA_DIR / "annual_minus_monthly_global.csv"
    with path.open("w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(
            [
                "metric",
                "units",
                "annual_anomaly_mean",
                "monthly_anomaly_mean",
                "annual_minus_monthly",
                "ci95",
                "p_value",
                "significant_95",
                "n_years",
                "dof_eff",
            ]
        )
        for metric, (annual_mean, monthly_mean, result) in results.items():
            writer.writerow(
                [
                    metric,
                    METRIC_UNITS[metric],
                    f"{annual_mean:.10g}",
                    f"{monthly_mean:.10g}",
                    f"{result.mean:.10g}",
                    f"{result.ci95:.10g}",
                    f"{result.p_value:.10g}",
                    int(result.p_value < 0.05),
                    result.n,
                    f"{result.dof_eff:.10g}",
                ]
            )
    return path


def latitude_results(
    control: dict[str, np.ndarray], case: dict[str, np.ndarray], key: str
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    difference = case[key] - control[key]
    results = [difference_stats(difference[:, index]) for index in range(difference.shape[1])]
    return (
        np.array([result.mean for result in results]),
        np.array([result.ci95 for result in results]),
        np.array([result.p_value for result in results]),
    )


def write_latitude_csv(
    lat: np.ndarray,
    comparisons: list[tuple[str, dict[str, np.ndarray], dict[str, np.ndarray]]],
) -> Path:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    path = DATA_DIR / "latitude_anomalies.csv"
    with path.open("w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(
            [
                "data_source",
                "latitude_deg_north",
                "temperature_anomaly_k",
                "temperature_ci95_k",
                "temperature_p_value",
                "o3_anomaly_du",
                "o3_ci95_du",
                "o3_p_value",
            ]
        )
        for label, control, case in comparisons:
            temperature = latitude_results(control, case, "temperature_lat")
            ozone = latitude_results(control, case, "ozone_lat")
            for index, latitude in enumerate(lat):
                writer.writerow(
                    [
                        label,
                        f"{latitude:.10g}",
                        f"{temperature[0][index]:.10g}",
                        f"{temperature[1][index]:.10g}",
                        f"{temperature[2][index]:.10g}",
                        f"{ozone[0][index]:.10g}",
                        f"{ozone[1][index]:.10g}",
                        f"{ozone[2][index]:.10g}",
                    ]
                )
    return path


def plot_latitude_anomalies(
    lat: np.ndarray,
    comparisons: list[tuple[str, dict[str, np.ndarray], dict[str, np.ndarray]]],
) -> tuple[Path, Path]:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 5.5), sharey=True)
    styles = {
        "Annual files": (COLORS["blue"], "-"),
        "Monthly files": (COLORS["vermillion"], "--"),
    }
    panels = (
        ("temperature_lat", "Stratospheric temperature anomaly (K)"),
        ("ozone_lat", "O3 column anomaly (DU)"),
    )

    for panel_index, (axis, (key, xlabel)) in enumerate(zip(axes, panels)):
        axis.axvline(0.0, color=COLORS["gray"], linewidth=0.9, alpha=0.7)
        for label, control, case in comparisons:
            mean, _, _ = latitude_results(control, case, key)
            color, linestyle = styles[label]
            axis.plot(
                mean,
                lat,
                color=color,
                linestyle=linestyle,
                linewidth=2.0,
                label=label,
            )
        axis.set_xlabel(xlabel)
        axis.grid(True, linestyle="--", linewidth=0.5, alpha=0.45)
        axis.text(
            0.02,
            1.02,
            panel_label(chr(ord("a") + panel_index)),
            transform=axis.transAxes,
            fontweight="bold",
        )
    axes[0].set_ylabel("Latitude (deg N)")
    axes[0].set_xlim(-1.0, 4.0)
    axes[0].legend(frameon=False, loc="best")
    fig.tight_layout()

    pdf_path = FIG_DIR / "latitude_anomalies.pdf"
    png_path = FIG_DIR / "latitude_anomalies.png"
    fig.savefig(pdf_path)
    fig.savefig(png_path, dpi=300)
    plt.close(fig)
    return pdf_path, png_path


def annual_minus_monthly_results(
    annual_control: dict[str, np.ndarray],
    annual_case: dict[str, np.ndarray],
    monthly_control: dict[str, np.ndarray],
    monthly_case: dict[str, np.ndarray],
    key: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compare paired annual- and monthly-file anomaly samples by latitude."""
    annual = annual_case[key] - annual_control[key]
    monthly = monthly_case[key] - monthly_control[key]
    if annual.shape != monthly.shape:
        raise ValueError(
            "Annual and monthly anomalies must have matching year/latitude "
            f"shapes for a paired comparison; got {annual.shape} and "
            f"{monthly.shape}."
        )
    paired_difference = annual - monthly
    results = [
        difference_stats(paired_difference[:, index])
        for index in range(paired_difference.shape[1])
    ]
    return (
        np.array([result.mean for result in results]),
        np.array([result.ci95 for result in results]),
        np.array([result.p_value for result in results]),
    )


def shade_significant_latitudes(
    axis: plt.Axes, lat: np.ndarray, significant: np.ndarray
) -> None:
    """Shade grid-cell latitude bands selected by a pointwise test."""
    latitude_edges = np.empty(len(lat) + 1)
    latitude_edges[1:-1] = 0.5 * (lat[:-1] + lat[1:])
    latitude_edges[0] = lat[0] - 0.5 * (lat[1] - lat[0])
    latitude_edges[-1] = lat[-1] + 0.5 * (lat[-1] - lat[-2])
    for index in np.flatnonzero(significant):
        axis.axhspan(
            latitude_edges[index],
            latitude_edges[index + 1],
            color=COLORS["yellow"],
            alpha=0.25,
            linewidth=0,
            zorder=0,
        )


def plot_annual_minus_monthly(
    lat: np.ndarray,
    annual_control: dict[str, np.ndarray],
    annual_case: dict[str, np.ndarray],
    monthly_control: dict[str, np.ndarray],
    monthly_case: dict[str, np.ndarray],
) -> tuple[Path, Path]:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 5.5), sharey=True)
    panels = (
        ("temperature_lat", "Temperature anomaly difference (K)"),
        ("ozone_lat", "O3 anomaly difference (DU)"),
    )

    for panel_index, (axis, (key, xlabel)) in enumerate(zip(axes, panels)):
        mean, ci95, p_value = annual_minus_monthly_results(
            annual_control,
            annual_case,
            monthly_control,
            monthly_case,
            key,
        )
        shade_significant_latitudes(axis, lat, p_value < 0.05)
        axis.axvline(0.0, color=COLORS["gray"], linewidth=0.9, alpha=0.7)
        axis.fill_betweenx(
            lat,
            mean - ci95,
            mean + ci95,
            color=COLORS["blue"],
            alpha=0.18,
            linewidth=0,
        )
        axis.plot(mean, lat, color=COLORS["blue"], linewidth=2.0)
        axis.set_xlabel(xlabel)
        axis.grid(True, linestyle="--", linewidth=0.5, alpha=0.45)
        axis.text(
            0.02,
            1.02,
            panel_label(chr(ord("a") + panel_index)),
            transform=axis.transAxes,
            fontweight="bold",
        )

    axes[0].set_ylabel("Latitude (deg N)")
    axes[0].legend(
        handles=[
            plt.Line2D(
                [0], [0], color=COLORS["blue"], linewidth=2.0,
                label="Annual minus monthly",
            ),
            Patch(
                facecolor=COLORS["blue"], alpha=0.18, edgecolor="none",
                label="95% CI",
            ),
            Patch(
                facecolor=COLORS["yellow"], alpha=0.25, edgecolor="none",
                label="Pointwise paired p < 0.05",
            ),
        ],
        frameon=False,
        loc="best",
    )
    fig.tight_layout()

    pdf_path = FIG_DIR / "annual_minus_monthly_latitude.pdf"
    png_path = FIG_DIR / "annual_minus_monthly_latitude.png"
    fig.savefig(pdf_path)
    fig.savefig(png_path, dpi=300)
    plt.close(fig)
    return pdf_path, png_path


def print_summary(
    comparisons: list[tuple[str, dict[str, np.ndarray], dict[str, np.ndarray]]]
) -> None:
    for label, control, case in comparisons:
        print(f"\n{label} (case minus control)")
        control_metrics = metric_series(control)
        case_metrics = metric_series(case)
        for metric in control_metrics:
            result = difference_stats(case_metrics[metric] - control_metrics[metric])
            print(
                f"  {metric:28s} {result.mean:9.4f} +/- {result.ci95:.4f} "
                f"{METRIC_UNITS[metric]} (p={result.p_value:.4g})"
            )


def print_global_difference_summary(
    results: dict[str, tuple[float, float, DifferenceStats]],
) -> None:
    print("\nAnnual minus monthly (paired global anomalies)")
    for metric, (_, _, result) in results.items():
        significance = "significant" if result.p_value < 0.05 else "not significant"
        print(
            f"  {metric:28s} {result.mean:9.4f} +/- {result.ci95:.4f} "
            f"{METRIC_UNITS[metric]} (p={result.p_value:.4g}; {significance})"
        )


def main() -> None:
    area_fraction = latitude_weights()
    pairs = matching_monthly_files()

    annual_years, annual_control, annual_case = annual_diagnostics(area_fraction)
    dates, monthly_control, monthly_case = monthly_diagnostics(pairs, area_fraction)
    comparisons = [
        ("Annual files", annual_control, annual_case),
        ("Monthly files", monthly_control, monthly_case),
    ]

    with Dataset(CONTROL_FILE) as dataset:
        lat = read_variable(dataset, "lat")

    global_difference_results = global_annual_minus_monthly_results(
        annual_control,
        annual_case,
        monthly_control,
        monthly_case,
    )
    global_path = write_global_csv(comparisons)
    global_difference_path = write_global_difference_csv(global_difference_results)
    latitude_path = write_latitude_csv(lat, comparisons)
    pdf_path, png_path = plot_latitude_anomalies(lat, comparisons)
    difference_pdf_path, difference_png_path = plot_annual_minus_monthly(
        lat,
        annual_control,
        annual_case,
        monthly_control,
        monthly_case,
    )

    print(
        "Annual period: "
        f"{annual_years[0]}-{annual_years[-1]} ({len(annual_years)} annual values)"
    )
    print(
        "Monthly period: "
        f"{dates[0]} to {dates[-1]} ({len(dates)} months; "
        f"{len(dates) // 12} complete 12-month means)"
    )
    print(
        f"Temperature levels: [{TEMP_START_LEVEL}:{FIXED_TROPOPAUSE_INDEX}] "
        f"(fixed tropopause boundary at index {FIXED_TROPOPAUSE_INDEX})."
    )
    print_summary(comparisons)
    print_global_difference_summary(global_difference_results)
    print(f"\nSaved: {global_path}")
    print(f"Saved: {global_difference_path}")
    print(f"Saved: {latitude_path}")
    print(f"Saved: {pdf_path}")
    print(f"Saved: {png_path}")
    print(f"Saved: {difference_pdf_path}")
    print(f"Saved: {difference_png_path}")


if __name__ == "__main__":
    main()
