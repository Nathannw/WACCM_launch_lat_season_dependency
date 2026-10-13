#!/usr/bin/env python3
"""Plot control-case mean stratospheric meridional wind versus latitude."""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from netCDF4 import Dataset


DEFAULT_CONTROL_DIR = Path("results_cloud/lon/JUL_8_control")
DEFAULT_FIG_DIR = Path("fig/T_wind")
DEFAULT_DATA_DIR = Path("data_file")
EARTH_RADIUS_M = 6_371_000.0
SEASONS = (
    ("DJF", (12, 1, 2)),
    ("MAM", (3, 4, 5)),
    ("JJA", (6, 7, 8)),
    ("SON", (9, 10, 11)),
)


def list_control_files(control_dir: Path, skip_months: int) -> list[Path]:
    files = sorted(control_dir.glob("*.nc"), key=lambda path: path.name.lower())
    if not files:
        raise FileNotFoundError(f"No NetCDF files found in {control_dir}")
    if skip_months >= len(files):
        raise ValueError(
            f"skip_months={skip_months} removes all {len(files)} control files"
        )
    return files[skip_months:]


def pressure_layer_weights(
    dataset: Dataset, pressure_top_hpa: float, pressure_bottom_hpa: float
) -> tuple[np.ndarray, np.ndarray]:
    if pressure_top_hpa <= 0 or pressure_bottom_hpa <= 0:
        raise ValueError("Pressure bounds must be positive hPa values")
    if pressure_top_hpa >= pressure_bottom_hpa:
        raise ValueError("pressure_top_hpa must be less than pressure_bottom_hpa")

    lev = np.asarray(dataset.variables["lev"][:], dtype=float)
    if "ilev" in dataset.variables and len(dataset.variables["ilev"][:]) == len(lev) + 1:
        edges = np.asarray(dataset.variables["ilev"][:], dtype=float)
    else:
        mid_edges = 0.5 * (lev[:-1] + lev[1:])
        first_edge = lev[0] - (mid_edges[0] - lev[0])
        last_edge = lev[-1] + (lev[-1] - mid_edges[-1])
        edges = np.concatenate(([first_edge], mid_edges, [last_edge]))

    if edges[0] > edges[-1]:
        edges = edges[::-1]
        lev = lev[::-1]

    layer_top = np.minimum(edges[:-1], edges[1:])
    layer_bottom = np.maximum(edges[:-1], edges[1:])
    overlap = np.minimum(layer_bottom, pressure_bottom_hpa) - np.maximum(
        layer_top, pressure_top_hpa
    )
    weights = np.maximum(overlap, 0.0)
    selected = weights > 0

    if not np.any(selected):
        raise ValueError(
            "No model levels overlap the requested pressure range "
            f"{pressure_top_hpa:g}-{pressure_bottom_hpa:g} hPa"
        )

    return selected, weights[selected]


def read_monthly_meridional_wind(file_path: Path, selected_levels: np.ndarray) -> np.ma.MaskedArray:
    with Dataset(file_path) as dataset:
        variable = dataset.variables["V"]
        data = np.ma.asarray(variable[:])
        dims = variable.dimensions

    if "lev" not in dims or "lat" not in dims:
        raise ValueError(f"V in {file_path} must have lev and lat dimensions")

    lev_axis = dims.index("lev")
    data = np.ma.take(data, np.where(selected_levels)[0], axis=lev_axis)

    dims_after_level_slice = list(dims)
    lat_axis = dims_after_level_slice.index("lat")
    lev_axis = dims_after_level_slice.index("lev")

    axes_to_average = [
        axis
        for axis, dim in enumerate(dims_after_level_slice)
        if dim not in {"lev", "lat"}
    ]
    if axes_to_average:
        data = np.ma.mean(data, axis=tuple(axes_to_average))
        remaining_dims = [
            dim
            for dim in dims_after_level_slice
            if dim in {"lev", "lat"}
        ]
        lev_axis = remaining_dims.index("lev")
        lat_axis = remaining_dims.index("lat")

    if lev_axis != 0 or lat_axis != 1:
        data = np.ma.moveaxis(data, (lev_axis, lat_axis), (0, 1))

    return data


def month_from_filename(file_path: Path) -> int:
    match = re.search(r"\.(\d{4})-(\d{2})\.nc$", file_path.name)
    if not match:
        raise ValueError(f"Could not parse YYYY-MM from {file_path.name}")
    return int(match.group(2))


def weighted_monthly_profiles(
    files: list[Path], pressure_top_hpa: float, pressure_bottom_hpa: float
) -> tuple[np.ndarray, np.ndarray, list[tuple[Path, np.ma.MaskedArray]]]:
    with Dataset(files[0]) as dataset:
        lat = np.asarray(dataset.variables["lat"][:], dtype=float)
        selected_levels, level_weights = pressure_layer_weights(
            dataset, pressure_top_hpa, pressure_bottom_hpa
        )
        lev = np.asarray(dataset.variables["lev"][:], dtype=float)[selected_levels]

    monthly_profiles = []
    for file_path in files:
        monthly_v = read_monthly_meridional_wind(file_path, selected_levels)
        profile = np.ma.average(monthly_v, axis=0, weights=level_weights)
        monthly_profiles.append((file_path, profile))

    return lat, lev, monthly_profiles


def mean_profile(monthly_profiles: list[tuple[Path, np.ma.MaskedArray]]) -> np.ndarray:
    profiles = [profile for _, profile in monthly_profiles]
    return np.ma.mean(np.ma.stack(profiles), axis=0).filled(np.nan)


def seasonal_profiles(
    monthly_profiles: list[tuple[Path, np.ma.MaskedArray]]
) -> dict[str, tuple[int, np.ndarray]]:
    profiles_by_season = {}
    for season, months in SEASONS:
        profiles = [
            profile
            for file_path, profile in monthly_profiles
            if month_from_filename(file_path) in months
        ]
        if not profiles:
            raise ValueError(f"No files found for season {season}")
        profiles_by_season[season] = (
            len(profiles),
            np.ma.mean(np.ma.stack(profiles), axis=0).filled(np.nan),
        )
    return profiles_by_season


def save_profile_csv(lat: np.ndarray, mean_v: np.ndarray, output_csv: Path) -> None:
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    header = "latitude_deg_north,mean_stratospheric_meridional_wind_m_s"
    np.savetxt(
        output_csv,
        np.column_stack((lat, mean_v)),
        delimiter=",",
        header=header,
        comments="",
        fmt="%.8g",
    )


def save_seasonal_profile_csv(
    lat: np.ndarray,
    profiles_by_season: dict[str, tuple[int, np.ndarray]],
    output_csv: Path,
) -> None:
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    header = ["latitude_deg_north"]
    columns = [lat]
    for season, _ in SEASONS:
        files_used, profile = profiles_by_season[season]
        header.append(f"{season}_{files_used}_months_meridional_wind_m_s")
        columns.append(profile)

    np.savetxt(
        output_csv,
        np.column_stack(columns),
        delimiter=",",
        header=",".join(header),
        comments="",
        fmt="%.8g",
    )


def latitude_area_weights(lat: np.ndarray) -> np.ndarray:
    lat = np.asarray(lat, dtype=float)
    if lat.ndim != 1:
        raise ValueError("Latitude must be one-dimensional")
    if len(lat) < 2:
        raise ValueError("At least two latitude points are required")

    edges = np.empty(len(lat) + 1)
    edges[1:-1] = 0.5 * (lat[:-1] + lat[1:])
    edges[0] = max(-90.0, lat[0] - 0.5 * (lat[1] - lat[0]))
    edges[-1] = min(90.0, lat[-1] + 0.5 * (lat[-1] - lat[-2]))

    # Zonal-band area is proportional to sin(lat_north_edge) - sin(lat_south_edge).
    weights = np.sin(np.deg2rad(edges[1:])) - np.sin(np.deg2rad(edges[:-1]))
    weights = np.abs(weights)
    return weights / np.sum(weights)


def area_weighted_global_mean(lat: np.ndarray, values_by_lat: np.ndarray) -> float:
    weights = latitude_area_weights(lat)
    valid = np.isfinite(values_by_lat)
    if not np.any(valid):
        raise ValueError("No finite latitude values available for global mean")
    return float(np.average(values_by_lat[valid], weights=weights[valid]))


def meridional_convergence(lat: np.ndarray, meridional_wind: np.ndarray) -> np.ndarray:
    lat = np.asarray(lat, dtype=float)
    meridional_wind = np.asarray(meridional_wind, dtype=float)
    lat_rad = np.deg2rad(lat)
    cos_lat = np.cos(lat_rad)
    meridional_flux = meridional_wind * cos_lat
    flux_gradient = np.gradient(meridional_flux, lat_rad, edge_order=2)

    with np.errstate(divide="ignore", invalid="ignore"):
        convergence = -flux_gradient / (EARTH_RADIUS_M * cos_lat)

    convergence[np.abs(cos_lat) < 1.0e-8] = np.nan
    return convergence


def hemispheric_convergence_means(
    lat: np.ndarray, convergence: np.ndarray
) -> dict[str, float]:
    weights = latitude_area_weights(lat)
    hemispheres = {
        "SH": lat < 0.0,
        "NH": lat > 0.0,
    }
    means = {}
    for hemisphere, mask in hemispheres.items():
        valid = mask & np.isfinite(convergence)
        if not np.any(valid):
            means[hemisphere] = np.nan
        else:
            means[hemisphere] = float(
                np.average(convergence[valid], weights=weights[valid])
            )
    return means


def save_global_summary_csv(
    global_mean_v: float,
    pressure_top_hpa: float,
    pressure_bottom_hpa: float,
    files_used: int,
    output_csv: Path,
) -> None:
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(output_csv, "w") as file:
        file.write(
            "pressure_top_hpa,pressure_bottom_hpa,monthly_files_used,"
            "area_weighted_global_meridional_wind_m_s\n"
        )
        file.write(
            f"{pressure_top_hpa:g},{pressure_bottom_hpa:g},{files_used},"
            f"{global_mean_v:.8g}\n"
        )


def save_convergence_profile_csv(
    lat: np.ndarray,
    annual_convergence: np.ndarray,
    seasonal_convergence: dict[str, np.ndarray],
    output_csv: Path,
) -> None:
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    header = ["latitude_deg_north", "annual_convergence_s^-1"]
    columns = [lat, annual_convergence]
    for season, _ in SEASONS:
        header.append(f"{season}_convergence_s^-1")
        columns.append(seasonal_convergence[season])

    np.savetxt(
        output_csv,
        np.column_stack(columns),
        delimiter=",",
        header=",".join(header),
        comments="",
        fmt="%.8e",
    )


def save_hemispheric_convergence_csv(
    annual_means: dict[str, float],
    seasonal_means: dict[str, dict[str, float]],
    pressure_top_hpa: float,
    pressure_bottom_hpa: float,
    files_used: int,
    profiles_by_season: dict[str, tuple[int, np.ndarray]],
    output_csv: Path,
) -> None:
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(output_csv, "w") as file:
        file.write(
            "period,monthly_files_used,pressure_top_hpa,pressure_bottom_hpa,"
            "SH_area_weighted_convergence_s^-1,NH_area_weighted_convergence_s^-1\n"
        )
        file.write(
            f"Annual,{files_used},{pressure_top_hpa:g},{pressure_bottom_hpa:g},"
            f"{annual_means['SH']:.8e},{annual_means['NH']:.8e}\n"
        )
        for season, _ in SEASONS:
            season_files, _ = profiles_by_season[season]
            means = seasonal_means[season]
            file.write(
                f"{season},{season_files},{pressure_top_hpa:g},{pressure_bottom_hpa:g},"
                f"{means['SH']:.8e},{means['NH']:.8e}\n"
            )


def plot_profile(
    lat: np.ndarray,
    mean_v: np.ndarray,
    pressure_top_hpa: float,
    pressure_bottom_hpa: float,
    files_used: int,
    output_pdf: Path,
    output_png: Path,
) -> None:
    output_pdf.parent.mkdir(parents=True, exist_ok=True)
    output_png.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(lat, mean_v, color="#0072B2", linewidth=1.8)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xlim(-90, 90)
    ax.set_xlabel("Latitude (deg)")
    ax.set_ylabel("Meridional wind (m s$^{-1}$)")
    ax.set_title(
        "Control mean stratospheric meridional wind\n"
        f"{pressure_top_hpa:g}-{pressure_bottom_hpa:g} hPa, {files_used} monthly files"
    )
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_pdf)
    fig.savefig(output_png, dpi=300)
    plt.close(fig)


def plot_seasonal_profiles(
    lat: np.ndarray,
    profiles_by_season: dict[str, tuple[int, np.ndarray]],
    pressure_top_hpa: float,
    pressure_bottom_hpa: float,
    output_pdf: Path,
    output_png: Path,
) -> None:
    output_pdf.parent.mkdir(parents=True, exist_ok=True)
    output_png.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(2, 2, figsize=(10, 7), sharex=True, sharey=True)
    axes_flat = axes.ravel()

    all_values = np.concatenate(
        [profiles_by_season[season][1] for season, _ in SEASONS]
    )
    y_limit = np.nanmax(np.abs(all_values))
    y_limit = max(0.05, float(y_limit) * 1.1)

    for ax, (season, _) in zip(axes_flat, SEASONS):
        files_used, profile = profiles_by_season[season]
        global_mean_v = area_weighted_global_mean(lat, profile)
        ax.plot(lat, profile, color="#0072B2", linewidth=1.8)
        ax.axhline(0, color="black", linewidth=0.8)
        ax.set_xlim(-90, 90)
        ax.set_ylim(-y_limit, y_limit)
        ax.set_title(f"{season} ({files_used} months, global={global_mean_v:.3g} m s$^{{-1}}$)")
        ax.grid(True, alpha=0.25)

    fig.suptitle(
        "Control mean stratospheric meridional wind\n"
        f"{pressure_top_hpa:g}-{pressure_bottom_hpa:g} hPa"
    )
    fig.supxlabel("Latitude (deg)")
    fig.supylabel("Meridional wind (m s$^{-1}$)")
    fig.tight_layout()
    fig.savefig(output_pdf)
    fig.savefig(output_png, dpi=300)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Read the WACCM control case and plot pressure-weighted average "
            "stratospheric meridional wind over latitude."
        )
    )
    parser.add_argument(
        "--control-dir",
        type=Path,
        default=DEFAULT_CONTROL_DIR,
        help=f"Control-case NetCDF directory (default: {DEFAULT_CONTROL_DIR})",
    )
    parser.add_argument(
        "--skip-months",
        type=int,
        default=120,
        help="Number of sorted monthly files to skip as spin-up (default: 0)",
    )
    parser.add_argument(
        "--pressure-top-hpa",
        type=float,
        default= 10, #0.04, # 70km
        help="Top pressure bound for stratospheric average in hPa (default: 1)",
    )
    parser.add_argument(
        "--pressure-bottom-hpa",
        type=float,
        default=50, #3, # 40km
        help="Bottom pressure bound for stratospheric average in hPa (default: 100)",
    )
    parser.add_argument(
        "--output-pdf",
        type=Path,
        default=DEFAULT_FIG_DIR / "control_stratospheric_meridional_wind.pdf",
        help="Output PDF path",
    )
    parser.add_argument(
        "--output-png",
        type=Path,
        default=DEFAULT_FIG_DIR / "control_stratospheric_meridional_wind.png",
        help="Output PNG path",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=DEFAULT_DATA_DIR / "control_stratospheric_meridional_wind.csv",
        help="Output CSV path",
    )
    parser.add_argument(
        "--output-global-csv",
        type=Path,
        default=DEFAULT_DATA_DIR / "control_stratospheric_meridional_wind_global.csv",
        help="Output CSV path for the single area-weighted global value",
    )
    parser.add_argument(
        "--output-seasonal-pdf",
        type=Path,
        default=DEFAULT_FIG_DIR / "control_stratospheric_meridional_wind_seasonal.pdf",
        help="Output PDF path for DJF/MAM/JJA/SON subplots",
    )
    parser.add_argument(
        "--output-seasonal-png",
        type=Path,
        default=DEFAULT_FIG_DIR / "control_stratospheric_meridional_wind_seasonal.png",
        help="Output PNG path for DJF/MAM/JJA/SON subplots",
    )
    parser.add_argument(
        "--output-seasonal-csv",
        type=Path,
        default=DEFAULT_DATA_DIR / "control_stratospheric_meridional_wind_seasonal.csv",
        help="Output CSV path for DJF/MAM/JJA/SON latitude profiles",
    )
    parser.add_argument(
        "--output-convergence-csv",
        type=Path,
        default=DEFAULT_DATA_DIR / "control_stratospheric_meridional_wind_convergence.csv",
        help="Output CSV path for annual and seasonal convergence profiles",
    )
    parser.add_argument(
        "--output-hemispheric-convergence-csv",
        type=Path,
        default=DEFAULT_DATA_DIR
        / "control_stratospheric_meridional_wind_hemispheric_convergence.csv",
        help="Output CSV path for area-weighted SH/NH convergence means",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    files = list_control_files(args.control_dir, args.skip_months)
    lat, lev, monthly_profiles = weighted_monthly_profiles(
        files, args.pressure_top_hpa, args.pressure_bottom_hpa
    )
    mean_v = mean_profile(monthly_profiles)
    profiles_by_season = seasonal_profiles(monthly_profiles)
    global_mean_v = area_weighted_global_mean(lat, mean_v)
    annual_convergence = meridional_convergence(lat, mean_v)
    annual_convergence_means = hemispheric_convergence_means(lat, annual_convergence)
    seasonal_convergence = {
        season: meridional_convergence(lat, profiles_by_season[season][1])
        for season, _ in SEASONS
    }
    seasonal_convergence_means = {
        season: hemispheric_convergence_means(lat, seasonal_convergence[season])
        for season, _ in SEASONS
    }

    save_profile_csv(lat, mean_v, args.output_csv)
    save_seasonal_profile_csv(lat, profiles_by_season, args.output_seasonal_csv)
    save_convergence_profile_csv(
        lat,
        annual_convergence,
        seasonal_convergence,
        args.output_convergence_csv,
    )
    save_hemispheric_convergence_csv(
        annual_convergence_means,
        seasonal_convergence_means,
        args.pressure_top_hpa,
        args.pressure_bottom_hpa,
        len(files),
        profiles_by_season,
        args.output_hemispheric_convergence_csv,
    )
    save_global_summary_csv(
        global_mean_v,
        args.pressure_top_hpa,
        args.pressure_bottom_hpa,
        len(files),
        args.output_global_csv,
    )
    plot_profile(
        lat,
        mean_v,
        args.pressure_top_hpa,
        args.pressure_bottom_hpa,
        len(files),
        args.output_pdf,
        args.output_png,
    )
    plot_seasonal_profiles(
        lat,
        profiles_by_season,
        args.pressure_top_hpa,
        args.pressure_bottom_hpa,
        args.output_seasonal_pdf,
        args.output_seasonal_png,
    )

    print(f"Read {len(files)} files from {args.control_dir}")
    print(f"Used {len(lev)} pressure levels from {lev.min():.3g} to {lev.max():.3g} hPa")
    print(f"Area-weighted global mean: {global_mean_v:.8g} m s^-1")
    print(
        "Annual convergence: "
        f"SH {annual_convergence_means['SH']:.8e} s^-1, "
        f"NH {annual_convergence_means['NH']:.8e} s^-1"
    )
    for season, _ in SEASONS:
        files_used, profile = profiles_by_season[season]
        seasonal_global = area_weighted_global_mean(lat, profile)
        convergence_means = seasonal_convergence_means[season]
        print(
            f"{season}: {files_used} months, global mean {seasonal_global:.8g} m s^-1, "
            f"convergence SH {convergence_means['SH']:.8e} s^-1, "
            f"NH {convergence_means['NH']:.8e} s^-1"
        )
    print(f"Wrote {args.output_csv}")
    print(f"Wrote {args.output_global_csv}")
    print(f"Wrote {args.output_seasonal_csv}")
    print(f"Wrote {args.output_convergence_csv}")
    print(f"Wrote {args.output_hemispheric_convergence_csv}")
    print(f"Wrote {args.output_pdf}")
    print(f"Wrote {args.output_png}")
    print(f"Wrote {args.output_seasonal_pdf}")
    print(f"Wrote {args.output_seasonal_png}")


if __name__ == "__main__":
    main()
