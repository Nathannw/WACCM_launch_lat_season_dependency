#!/usr/bin/env python3
"""Plot annual time series of global response anomalies from WACCM outputs.

The default comparison is each launch case minus JUL_8_control, matching
analysis/response_data.py. Annual values are calendar-year means from 1999
onward by default. Partial years are kept unless --min-months-per-year is set
to 12.
"""

from __future__ import annotations

import argparse
import csv
import re
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from analysis.figure_style import COLORS

try:
    from netCDF4 import Dataset as NetcdfDataset
except ImportError:  # scipy fallback works for the local NetCDF classic files
    NetcdfDataset = None
    from scipy.io import netcdf_file
else:
    netcdf_file = None


PROJECT_DIR = Path(__file__).resolve().parent
RESULTS_DIR = PROJECT_DIR / "results_cloud" / "lon"
CONTROL_DIR = RESULTS_DIR / "JUL_8_control"
DATA_DIR = PROJECT_DIR / "data_file"
FIG_DIR = PROJECT_DIR / "fig" / "annual_timeseries"
DATE_RE = re.compile(r"\.(\d{4})-(\d{2})\.nc$")

AIR_MOLAR_MASS_G_PER_MOL = 28.97
DU_MOLECULES_PER_CM2 = 4.4615e-8
EARTH_RADIUS_CM = 637.8e6
TEMP_START_LEVEL = 25
FALLBACK_STRAT_END_LEVEL = 53

BASE_RESPONSES = ("T", "O3", "FLNT", "FSNT", "FLNTC", "FSNTC")
DERIVED_RESPONSES = ("ERF", "ERFC")
PLOT_RESPONSES = ("T", "O3", "ERF", "ERFC", "FSNT", "FLNT", "FSNTC", "FLNTC")

RESPONSE_LABELS = {
    "T": "Stratospheric temperature anomaly (K)",
    "O3": "Ozone column anomaly (DU)",
    "ERF": r"All-sky ERF anomaly ($\mathrm{W}\ \mathrm{m}^{-2}$)",
    "ERFC": r"Clear-sky ERF anomaly ($\mathrm{W}\ \mathrm{m}^{-2}$)",
    "FSNT": r"All-sky SW anomaly ($\mathrm{W}\ \mathrm{m}^{-2}$)",
    "FLNT": r"All-sky LW anomaly ($\mathrm{W}\ \mathrm{m}^{-2}$)",
    "FSNTC": r"Clear-sky SW anomaly ($\mathrm{W}\ \mathrm{m}^{-2}$)",
    "FLNTC": r"Clear-sky LW anomaly ($\mathrm{W}\ \mathrm{m}^{-2}$)",
}

LOCATION_COLORS = {
    "SA": COLORS["blue"],
    "AU": COLORS["sky_blue"],
    "EQ": COLORS["green"],
    "CC": COLORS["orange"],
    "CC_BC_only": COLORS["vermillion"],
    "UK": COLORS["purple"],
    "RU": COLORS["black"],
}

SEASON_TITLES = {
    "all_year": "Year-round",
    "summer": "Boreal summer-only",
    "winter": "Boreal winter-only",
}


@dataclass(frozen=True)
class CaseConfig:
    date: str
    season: str
    location: str
    latitude: float

    @property
    def case_id(self) -> str:
        return f"{self.date}_30Gg_{self.season}_{self.location}"

    @property
    def directory(self) -> Path:
        return RESULTS_DIR / self.case_id

    @property
    def label(self) -> str:
        if self.location == "CC_BC_only":
            return "CC BC only"
        return f"{self.location} ({self.latitude:g} deg)"


CASES = (
    CaseConfig("AUG_13", "all_year", "CC_BC_only", 29),
    CaseConfig("JUL_8", "all_year", "SA", -55),
    CaseConfig("JUL_8", "summer", "SA", -55),
    CaseConfig("JUL_8", "winter", "SA", -55),
    CaseConfig("AUG_13", "all_year", "AU", -29),
    CaseConfig("AUG_13", "all_year", "EQ", 0),
    CaseConfig("AUG_13", "summer", "EQ", 0),
    CaseConfig("AUG_13", "winter", "EQ", 0),
    CaseConfig("JUL_8", "all_year", "CC", 29),
    CaseConfig("JUL_8", "summer", "CC", 29),
    CaseConfig("JUL_8", "winter", "CC", 29),
    CaseConfig("AUG_13", "all_year", "UK", 55),
    CaseConfig("JUL_8", "all_year", "RU", 70),
    CaseConfig("JUL_8", "summer", "RU", 70),
    CaseConfig("JUL_8", "winter", "RU", 70),
)


def open_dataset(path: Path):
    if NetcdfDataset is not None:
        return NetcdfDataset(path)
    return netcdf_file(path, mode="r", mmap=False)


def read_variable(dataset, name: str) -> np.ndarray:
    variable = dataset.variables[name]
    data = np.array(variable[:], dtype=float, copy=True)
    for marker_name in ("_FillValue", "missing_value"):
        marker = getattr(variable, marker_name, None)
        if marker is not None:
            marker_value = float(np.asarray(marker).ravel()[0])
            data[np.isclose(data, marker_value)] = np.nan
    data[np.abs(data) > 1.0e35] = np.nan
    return data


def parse_date(path: Path) -> tuple[int, int]:
    match = DATE_RE.search(path.name)
    if not match:
        raise ValueError(f"Could not parse YYYY-MM from {path.name}")
    return int(match.group(1)), int(match.group(2))


def files_by_date(case_dir: Path) -> dict[str, Path]:
    if not case_dir.exists():
        raise FileNotFoundError(f"Missing case directory: {case_dir}")
    files = sorted(case_dir.glob("*.nc"), key=lambda path: path.name.lower())
    if not files:
        raise FileNotFoundError(f"No NetCDF files found in {case_dir}")
    dated_files = {}
    for path in files:
        year, month = parse_date(path)
        dated_files[f"{year:04d}-{month:02d}"] = path
    return dated_files


def matching_files(reference_dir: Path, case_dir: Path, start_year: int) -> list[tuple[str, Path, Path]]:
    reference = files_by_date(reference_dir)
    case = files_by_date(case_dir)
    dates = sorted(
        date for date in set(reference) & set(case)
        if int(date[:4]) >= start_year
    )
    if not dates:
        raise ValueError(f"No matching files from {start_year} for {case_dir}")
    return [(date, reference[date], case[date]) for date in dates]


def zonal_area() -> np.ndarray:
    weights_file = PROJECT_DIR / "geo_file" / "coords_1.9x2.5_L88_c150828.nc"
    with open_dataset(weights_file) as dataset:
        gw = read_variable(dataset, "gw")
    surface_area = 4.0 * np.pi * EARTH_RADIUS_CM * EARTH_RADIUS_CM
    return gw * surface_area / np.nansum(gw)


def weighted_mean(values: np.ndarray, weights: np.ndarray) -> float:
    values = np.ma.masked_invalid(np.ma.array(values))
    weights = np.ma.array(weights, mask=np.ma.getmaskarray(values))
    return float(np.ma.sum(values * weights) / np.ma.sum(weights))


def global_area_weighted_mean(values: np.ndarray, area_weights: np.ndarray) -> float:
    values = np.ma.masked_invalid(np.ma.array(values))
    if values.shape[-1] != len(area_weights):
        return float(np.ma.mean(values))
    weights = np.broadcast_to(area_weights, values.shape)
    weights = np.ma.array(weights, mask=np.ma.getmaskarray(values))
    return float(np.ma.sum(values * weights) / np.ma.sum(weights))


def tropopause_indices(dataset, lat_index: int) -> tuple[int, int, float]:
    trop_pressure = read_variable(dataset, "TROP_P")[:, lat_index] / 100.0
    lev = read_variable(dataset, "lev")
    if lev[0] > lev[-1]:
        lev = lev[::-1]

    idx_upper = np.searchsorted(lev, trop_pressure)
    idx_lower = idx_upper - 1
    idx_lower = np.clip(idx_lower, 0, len(lev) - 2)
    idx_upper = np.clip(idx_upper, 1, len(lev) - 1)

    lower = int(np.asarray(idx_lower).ravel()[0])
    upper = int(np.asarray(idx_upper).ravel()[0])
    p1 = lev[lower]
    p2 = lev[upper]
    ratio = float(np.asarray((trop_pressure - p1) / (p2 - p1)).ravel()[0])
    return lower, upper, ratio


def pressure_level_weights(dataset) -> np.ndarray:
    lev = read_variable(dataset, "lev")
    if "ilev" in dataset.variables and len(read_variable(dataset, "ilev")) == len(lev) + 1:
        edges = read_variable(dataset, "ilev")
    else:
        mid_edges = 0.5 * (lev[:-1] + lev[1:])
        first_edge = lev[0] - (mid_edges[0] - lev[0])
        last_edge = lev[-1] + (lev[-1] - mid_edges[-1])
        edges = np.concatenate(([first_edge], mid_edges, [last_edge]))
    return np.abs(np.diff(edges))


def pressure_weighted_mean(values: np.ndarray, weights: np.ndarray) -> float:
    values = np.ma.masked_invalid(np.ma.array(values))
    weights = np.ma.masked_invalid(np.ma.array(weights))
    if values.ndim > weights.ndim:
        weights = np.reshape(weights, (1,) * (values.ndim - weights.ndim) + weights.shape)
    weights = np.broadcast_to(weights, values.shape)
    weights = np.ma.array(
        weights,
        mask=np.ma.getmaskarray(values) | np.ma.getmaskarray(weights),
    )
    return float(np.ma.sum(values * weights) / np.ma.sum(weights))


def fixed_layer_pressure_weighted_global_mean(
    response_data: np.ndarray,
    area_weights: np.ndarray,
    pressure_weights: np.ndarray,
) -> float:
    end_level = min(FALLBACK_STRAT_END_LEVEL, response_data.shape[1])
    lat_values = [
        pressure_weighted_mean(
            response_data[:, TEMP_START_LEVEL:end_level, lat_index],
            pressure_weights[TEMP_START_LEVEL:end_level],
        )
        for lat_index in range(response_data.shape[-1])
    ]
    return weighted_mean(np.asarray(lat_values), area_weights[:response_data.shape[-1]])


def aggregate_response(dataset, response: str, area_weights: np.ndarray) -> float:
    if response in ("FLNT", "FSNT", "FLNTC", "FSNTC"):
        return global_area_weighted_mean(read_variable(dataset, response), area_weights)

    if response == "O3":
        mass = read_variable(dataset, "MASS")[0, :, :]
        ozone = read_variable(dataset, "O3")[0, :, :]
        mol_air = mass * 1000.0 / AIR_MOLAR_MASS_G_PER_MOL
        mol_ozone = ozone * mol_air
        mol_ozone_per_cm2 = np.nansum(mol_ozone / area_weights, axis=0)
        dobson_units = mol_ozone_per_cm2 / DU_MOLECULES_PER_CM2
        return weighted_mean(dobson_units, area_weights)

    if response in ("T", "CLOUD", "CLDICE", "CLDLIQ"):
        response_data = read_variable(dataset, response)
        pressure_weights = pressure_level_weights(dataset)
        if "TROP_P" not in dataset.variables:
            return fixed_layer_pressure_weighted_global_mean(
                response_data, area_weights, pressure_weights
            )
        lat_values = []
        nlat = response_data.shape[-1]
        for lat_index in range(nlat):
            lower, upper, trop_ratio = tropopause_indices(dataset, lat_index)
            lower_mean = pressure_weighted_mean(
                response_data[:, TEMP_START_LEVEL:lower, lat_index],
                pressure_weights[TEMP_START_LEVEL:lower],
            )
            upper_mean = pressure_weighted_mean(
                response_data[:, TEMP_START_LEVEL:upper, lat_index],
                pressure_weights[TEMP_START_LEVEL:upper],
            )
            lat_values.append(lower_mean * (1 - trop_ratio) + upper_mean * trop_ratio)
        return weighted_mean(np.asarray(lat_values), area_weights[:nlat])

    raise ValueError(f"Unsupported response: {response}")


def monthly_response_anomalies(
    pairs: list[tuple[str, Path, Path]],
    area_weights: np.ndarray,
) -> dict[str, list[tuple[str, float]]]:
    output = {response: [] for response in BASE_RESPONSES}
    for date, reference_path, case_path in pairs:
        with open_dataset(reference_path) as reference, open_dataset(case_path) as case:
            reference_values = {
                response: aggregate_response(reference, response, area_weights)
                for response in BASE_RESPONSES
            }
            case_values = {
                response: aggregate_response(case, response, area_weights)
                for response in BASE_RESPONSES
            }
        for response in BASE_RESPONSES:
            output[response].append((date, case_values[response] - reference_values[response]))
    output["ERF"] = [
        (date, fsnt - flnt)
        for (date, fsnt), (_, flnt) in zip(output["FSNT"], output["FLNT"])
    ]
    output["ERFC"] = [
        (date, fsntc - flntc)
        for (date, fsntc), (_, flntc) in zip(output["FSNTC"], output["FLNTC"])
    ]
    return output


def annualize(
    monthly_values: list[tuple[str, float]],
    min_months_per_year: int,
) -> list[tuple[int, int, float]]:
    by_year: dict[int, list[float]] = {}
    for date, value in monthly_values:
        by_year.setdefault(int(date[:4]), []).append(value)

    annual_values = []
    for year in sorted(by_year):
        values = np.asarray(by_year[year], dtype=float)
        values = values[np.isfinite(values)]
        if len(values) >= min_months_per_year:
            annual_values.append((year, len(values), float(np.nanmean(values))))
    return annual_values


def collect_annual_responses(
    start_year: int,
    min_months_per_year: int,
) -> list[dict[str, object]]:
    area_weights = zonal_area()
    rows = []
    for case in CASES:
        print(f"Processing {case.case_id}...")
        pairs = matching_files(CONTROL_DIR, case.directory, start_year)
        monthly = monthly_response_anomalies(pairs, area_weights)
        for response in PLOT_RESPONSES:
            for year, months, value in annualize(monthly[response], min_months_per_year):
                rows.append({
                    "year": year,
                    "months": months,
                    "response": response,
                    "case_id": case.case_id,
                    "launch_date": case.date,
                    "launch_season": case.season,
                    "location": case.location,
                    "launch_latitude": case.latitude,
                    "anomaly": value,
                })
    return rows


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "year",
        "months",
        "response",
        "case_id",
        "launch_date",
        "launch_season",
        "location",
        "launch_latitude",
        "anomaly",
    ]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def plot_response(response: str, rows: list[dict[str, object]], fig_dir: Path) -> None:
    response_rows = [row for row in rows if row["response"] == response]
    fig, axes = plt.subplots(1, 3, figsize=(13, 4), sharex=True, sharey=True)

    for ax, season in zip(axes, ("all_year", "summer", "winter")):
        season_cases = [case for case in CASES if case.season == season]
        for case in season_cases:
            case_rows = [
                row for row in response_rows
                if row["case_id"] == case.case_id
            ]
            if not case_rows:
                continue
            case_rows.sort(key=lambda row: row["year"])
            years = [int(row["year"]) for row in case_rows]
            values = [float(row["anomaly"]) for row in case_rows]
            linestyle = "--" if case.location == "CC_BC_only" else "-"
            ax.plot(
                years,
                values,
                color=LOCATION_COLORS.get(case.location, COLORS["gray"]),
                linestyle=linestyle,
                marker="o",
                markersize=2.8,
                linewidth=1.2,
                label=case.label,
            )
        ax.axhline(0, color=COLORS["gray"], linewidth=0.8, alpha=0.7)
        ax.set_title(SEASON_TITLES[season])
        ax.set_xlabel("Year")
        ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.5)
        ax.legend(frameon=False, fontsize=8, loc="best")

    axes[0].set_ylabel(RESPONSE_LABELS[response])
    fig.tight_layout()
    fig_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(fig_dir / f"{response}_annual_timeseries.pdf")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot annual time series of WACCM response anomalies."
    )
    parser.add_argument(
        "--start-year",
        type=int,
        default=1999,
        help="First calendar year to include.",
    )
    parser.add_argument(
        "--min-months-per-year",
        type=int,
        default=12,
        choices=range(1, 13),
        metavar="{1..12}",
        help="Minimum number of monthly files required to keep an annual value.",
    )
    parser.add_argument(
        "--csv",
        type=Path,
        default=DATA_DIR / "annual_response_timeseries.csv",
        help="Output CSV path.",
    )
    parser.add_argument(
        "--fig-dir",
        type=Path,
        default=FIG_DIR,
        help="Output figure directory.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = collect_annual_responses(
        start_year=args.start_year,
        min_months_per_year=args.min_months_per_year,
    )
    write_csv(args.csv, rows)
    for response in PLOT_RESPONSES:
        plot_response(response, rows, args.fig_dir)

    years = sorted({int(row["year"]) for row in rows})
    print(f"Saved annual data: {args.csv}")
    print(f"Saved figures under: {args.fig_dir}")
    print(f"Year range plotted: {years[0]}-{years[-1]}")


if __name__ == "__main__":
    main()
