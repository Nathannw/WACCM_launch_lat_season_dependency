from pathlib import Path

import numpy as np
from scipy import stats


PROJECT_DIR = Path(__file__).resolve().parent
DATA_DIR = PROJECT_DIR / "data_file"

CONTROL_FILE = PROJECT_DIR / (
    "FWma2000climo_default_case_fixed.cam.2000-2049.annual.concat_IDL_vars.nc"
)
CASE_FILE = PROJECT_DIR / (
    "FWma2000climo_f19_f19_mg17_BC_30Gg_case_fixed_num_err."
    "cam.2000-2049.annual.concat_IDL_vars.nc"
)
MONTHLY_CONTROL_DIR = PROJECT_DIR / "results_cloud" / "lon" / "JUL_8_control"
MONTHLY_CASE_DIR = PROJECT_DIR / "results_cloud" / "lon" / "JUL_8_30Gg_all_year_CC"
GEO_FILE = PROJECT_DIR / "geo_file" / "coords_1.9x2.5_L70.nc"
GW_FILE = PROJECT_DIR / "geo_file" / "coords_1.9x2.5_L88_c150828.nc"

LAST_N_YEARS = 40
MONTHLY_SPINUP_FILES = 120
# This top-to-143 hPa slice reproduces the reported paper-scale
# stratospheric temperature diagnostic much more closely than [25:52].
PAPER_T_LEVELS = slice(None, 51)
# The paper-scale O3 diagnostic matches total-column ozone over all levels.
PAPER_O3_LEVELS = slice(None)
RADIATIVE_FLUXES = ("FLNT", "FSNT", "FLNTC", "FSNTC")
G = 9.80665
MOLAR_MASS_DRY_AIR = 0.02897
DU_IN_MOL_PER_CM2 = 4.4615e-8
EARTH_RADIUS_CM = 637.8e6


class Dataset:
    def __init__(self, path):
        self.path = Path(path)
        self._dataset = None
        self._backend = None

    def __enter__(self):
        try:
            from scipy.io import netcdf_file

            self._dataset = netcdf_file(self.path, mode="r", mmap=False)
            self._backend = "scipy"
        except Exception:
            from netCDF4 import Dataset as NetCDFDataset

            self._dataset = NetCDFDataset(self.path)
            self._backend = "netCDF4"
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self._dataset.close()

    @property
    def variables(self):
        return self._dataset.variables


def read_variable(dataset, name, key):
    data = np.asarray(dataset.variables[name][key], dtype=np.float64)
    var = dataset.variables[name]
    for attr_name in ("_FillValue", "missing_value"):
        fill_value = getattr(var, attr_name, None)
        if fill_value is not None:
            data = np.where(data == fill_value, np.nan, data)
    return data


def time_slice_for_last_years(dataset, n_years):
    n_time = dataset.variables["time"].shape[0]
    if n_time < n_years:
        raise ValueError(f"{dataset.path.name} has only {n_time} time values.")
    return slice(n_time - n_years, n_time)


def years_for_slice(dataset, time_slice):
    if "date" not in dataset.variables:
        return None
    dates = np.asarray(dataset.variables["date"][time_slice])
    return dates.astype(int) // 10000


def confidence_interval_95(values):
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if values.size <= 1:
        return 0.0
    return float(stats.t.ppf(0.975, values.size - 1) * stats.sem(values))


def annual_average(monthly_values):
    monthly_values = np.asarray(monthly_values, dtype=np.float64)
    n_complete_months = (monthly_values.size // 12) * 12
    if n_complete_months == 0:
        raise ValueError("Need at least 12 monthly values for annual averages.")
    return np.nanmean(monthly_values[:n_complete_months].reshape(-1, 12), axis=1)


def paper_stratospheric_temperature_series(dataset, time_slice):
    temperature = read_variable(dataset, "T", (time_slice, PAPER_T_LEVELS, slice(None), slice(None)))
    return np.nanmean(temperature, axis=(1, 2, 3))


def paper_ozone_column_series(dataset, time_slice):
    values = []
    time_indices = range(time_slice.start, time_slice.stop)
    for time_index in time_indices:
        key = (time_index, PAPER_O3_LEVELS, slice(None), slice(None))
        ozone = read_variable(dataset, "O3", key)
        pdel = read_variable(dataset, "PDELDRY", key)
        mol_o3_per_m2 = ozone * pdel / (G * MOLAR_MASS_DRY_AIR)
        mol_o3_per_cm2 = np.nansum(mol_o3_per_m2, axis=0) / 1.0e4
        ozone_du = mol_o3_per_cm2 / DU_IN_MOL_PER_CM2
        values.append(np.nanmean(ozone_du))
    return np.array(values, dtype=np.float64)


def paper_flux_series(dataset, time_slice, name):
    flux = read_variable(dataset, name, (time_slice, slice(None), slice(None)))
    return np.nanmean(flux, axis=(1, 2))


def level_thickness_weights(dataset, level_slice):
    if "ilev" in dataset.variables:
        interfaces = read_variable(dataset, "ilev", slice(None))
    else:
        levels = read_variable(dataset, "lev", slice(None))
        mid_edges = 0.5 * (levels[:-1] + levels[1:])
        first_edge = levels[0] - (mid_edges[0] - levels[0])
        last_edge = levels[-1] + (levels[-1] - mid_edges[-1])
        interfaces = np.concatenate(([first_edge], mid_edges, [last_edge]))
    return np.abs(np.diff(interfaces))[level_slice]


def weighted_mean_series(values, weights):
    weights = np.broadcast_to(weights, values.shape)
    valid = np.isfinite(values) & np.isfinite(weights)
    numerator = np.sum(np.where(valid, values * weights, 0.0), axis=tuple(range(1, values.ndim)))
    denominator = np.sum(np.where(valid, weights, 0.0), axis=tuple(range(1, values.ndim)))
    return numerator / denominator


def pressure_area_weighted_temperature_series(dataset, time_slice, area_by_lat):
    temperature = read_variable(dataset, "T", (time_slice, PAPER_T_LEVELS, slice(None), slice(None)))
    level_weights = level_thickness_weights(dataset, PAPER_T_LEVELS)
    weights = level_weights.reshape(1, -1, 1, 1) * area_by_lat.reshape(1, 1, -1, 1)
    return weighted_mean_series(temperature, weights)


def area_weighted_ozone_column_series(dataset, time_slice, area_by_lat):
    values = []
    time_indices = range(time_slice.start, time_slice.stop)
    for time_index in time_indices:
        key = (time_index, PAPER_O3_LEVELS, slice(None), slice(None))
        ozone = read_variable(dataset, "O3", key)
        pdel = read_variable(dataset, "PDELDRY", key)
        mol_o3_per_m2 = ozone * pdel / (G * MOLAR_MASS_DRY_AIR)
        mol_o3_per_cm2 = np.nansum(mol_o3_per_m2, axis=0) / 1.0e4
        ozone_du = mol_o3_per_cm2 / DU_IN_MOL_PER_CM2
        values.append(weighted_mean_series(ozone_du.reshape(1, *ozone_du.shape), area_by_lat.reshape(1, -1, 1))[0])
    return np.array(values, dtype=np.float64)


def area_weighted_flux_series(dataset, time_slice, name, area_by_lat):
    flux = read_variable(dataset, name, (time_slice, slice(None), slice(None)))
    return weighted_mean_series(flux, area_by_lat.reshape(1, -1, 1))


def nc_files(case_dir):
    files = sorted(Path(case_dir).glob("*.nc"), key=lambda path: path.name.lower())
    if not files:
        raise FileNotFoundError(f"No .nc files found in {case_dir}")
    return files


def grid_cell_area_by_latitude():
    with Dataset(GEO_FILE) as dataset:
        nlon = read_variable(dataset, "lon", slice(None)).size
    with Dataset(GW_FILE) as dataset:
        gw = read_variable(dataset, "gw", slice(None))

    earth_area = 4.0 * np.pi * EARTH_RADIUS_CM**2
    return gw * earth_area / nlon / np.sum(gw)


def monthly_years_and_values(case_dir, value_func):
    files = nc_files(case_dir)[MONTHLY_SPINUP_FILES:]
    years = []
    values = []
    for path in files:
        with Dataset(path) as dataset:
            dates = np.asarray(dataset.variables["date"][:])
            years.append(int(dates[0]) // 10000)
            values.append(value_func(dataset))
    return np.array(years, dtype=int), np.array(values, dtype=np.float64)


def monthly_paper_temperature(dataset):
    temperature = read_variable(dataset, "T", (slice(None), PAPER_T_LEVELS, slice(None)))
    return float(np.nanmean(temperature))


def monthly_paper_ozone_column(dataset, area_by_lat):
    ozone = read_variable(dataset, "O3", (0, PAPER_O3_LEVELS, slice(None)))
    mass = read_variable(dataset, "MASS", (0, PAPER_O3_LEVELS, slice(None)))
    mol_air = mass / MOLAR_MASS_DRY_AIR
    mol_o3_per_cm2 = np.nansum(ozone * mol_air / area_by_lat.reshape(1, -1), axis=0)
    ozone_du = mol_o3_per_cm2 / DU_IN_MOL_PER_CM2
    return float(np.nanmean(ozone_du))


def monthly_paper_flux(dataset, name):
    return float(np.nanmean(read_variable(dataset, name, slice(None))))


def monthly_paper_case_values(case_dir, area_by_lat):
    files = nc_files(case_dir)[MONTHLY_SPINUP_FILES:]
    years = []
    values = {
        "temperature": [],
        "ozone_column": [],
        **{name: [] for name in RADIATIVE_FLUXES},
    }

    for path in files:
        with Dataset(path) as dataset:
            dates = np.asarray(dataset.variables["date"][:])
            years.append(int(dates[0]) // 10000)
            values["temperature"].append(monthly_paper_temperature(dataset))
            values["ozone_column"].append(monthly_paper_ozone_column(dataset, area_by_lat))
            for name in RADIATIVE_FLUXES:
                values[name].append(monthly_paper_flux(dataset, name))

    return np.array(years, dtype=int), {
        name: np.array(monthly_values, dtype=np.float64)
        for name, monthly_values in values.items()
    }


def monthly_case_summary(control_dir, case_dir):
    area_by_lat = grid_cell_area_by_latitude()

    control_years, control_monthly = monthly_paper_case_values(control_dir, area_by_lat)
    case_years, case_monthly = monthly_paper_case_values(case_dir, area_by_lat)

    n_months = min(
        *(values.size for values in control_monthly.values()),
        *(values.size for values in case_monthly.values()),
    )
    n_complete_months = (n_months // 12) * 12
    if n_complete_months == 0:
        raise ValueError("No complete annual pairs found for monthly case summary.")

    control_years = control_years[:n_complete_months]
    case_years = case_years[:n_complete_months]
    if not np.array_equal(control_years, case_years):
        print("Warning: monthly control and case year labels do not align exactly.")

    summary = {
        "years": control_years.reshape(-1, 12)[:, 0],
        "control_temperature_series_k": annual_average(control_monthly["temperature"][:n_complete_months]),
        "case_temperature_series_k": annual_average(case_monthly["temperature"][:n_complete_months]),
        "control_ozone_column_series_du": annual_average(control_monthly["ozone_column"][:n_complete_months]),
        "case_ozone_column_series_du": annual_average(case_monthly["ozone_column"][:n_complete_months]),
    }
    for name in RADIATIVE_FLUXES:
        summary[f"control_{name}_series_wm2"] = annual_average(control_monthly[name][:n_complete_months])
        summary[f"case_{name}_series_wm2"] = annual_average(case_monthly[name][:n_complete_months])
    return summary


def summarize(path):
    with Dataset(path) as dataset:
        time_slice = time_slice_for_last_years(dataset, LAST_N_YEARS)
        years = years_for_slice(dataset, time_slice)
        temperature_series = paper_stratospheric_temperature_series(dataset, time_slice)
        ozone_series = paper_ozone_column_series(dataset, time_slice)
        flux_series = {
            name: paper_flux_series(dataset, time_slice, name)
            for name in RADIATIVE_FLUXES
        }
        return {
            "path": path,
            "years": years,
            "temperature_series_k": temperature_series,
            "temperature_k": float(np.nanmean(temperature_series)),
            "temperature_ci95_k": confidence_interval_95(temperature_series),
            "ozone_column_series_du": ozone_series,
            "ozone_column_du": float(np.nanmean(ozone_series)),
            "ozone_column_ci95_du": confidence_interval_95(ozone_series),
            **{f"{name}_series_wm2": series for name, series in flux_series.items()},
        }


def weighted_summarize(path):
    area_by_lat = grid_cell_area_by_latitude()
    with Dataset(path) as dataset:
        time_slice = time_slice_for_last_years(dataset, LAST_N_YEARS)
        years = years_for_slice(dataset, time_slice)
        temperature_series = pressure_area_weighted_temperature_series(dataset, time_slice, area_by_lat)
        ozone_series = area_weighted_ozone_column_series(dataset, time_slice, area_by_lat)
        flux_series = {
            name: area_weighted_flux_series(dataset, time_slice, name, area_by_lat)
            for name in RADIATIVE_FLUXES
        }
        return {
            "path": path,
            "years": years,
            "temperature_series_k": temperature_series,
            "temperature_k": float(np.nanmean(temperature_series)),
            "temperature_ci95_k": confidence_interval_95(temperature_series),
            "ozone_column_series_du": ozone_series,
            "ozone_column_du": float(np.nanmean(ozone_series)),
            "ozone_column_ci95_du": confidence_interval_95(ozone_series),
            **{f"{name}_series_wm2": series for name, series in flux_series.items()},
        }


def summary_anomalies(control, case):
    temperature_diff = case["temperature_series_k"] - control["temperature_series_k"]
    ozone_diff = case["ozone_column_series_du"] - control["ozone_column_series_du"]
    erf_diff = (
        case["FSNT_series_wm2"]
        - control["FSNT_series_wm2"]
        - case["FLNT_series_wm2"]
        + control["FLNT_series_wm2"]
    )
    erfc_diff = (
        case["FSNTC_series_wm2"]
        - control["FSNTC_series_wm2"]
        - case["FLNTC_series_wm2"]
        + control["FLNTC_series_wm2"]
    )
    return {
        "temperature": (float(np.nanmean(temperature_diff)), confidence_interval_95(temperature_diff)),
        "ozone": (float(np.nanmean(ozone_diff)), confidence_interval_95(ozone_diff)),
        "erf": (float(np.nanmean(erf_diff)), confidence_interval_95(erf_diff)),
        "erfc": (float(np.nanmean(erfc_diff)), confidence_interval_95(erfc_diff)),
    }


def write_csv(
    control,
    case,
    temperature_anomaly,
    temperature_anomaly_ci95,
    ozone_column_anomaly,
    ozone_column_anomaly_ci95,
    erf_anomaly,
    erf_anomaly_ci95,
    erfc_anomaly,
    erfc_anomaly_ci95,
):
    DATA_DIR.mkdir(exist_ok=True)
    output_path = DATA_DIR / "last40_paper_match_stratosphere_anomalies.csv"
    control_erf = control["FSNT_series_wm2"] - control["FLNT_series_wm2"]
    case_erf = case["FSNT_series_wm2"] - case["FLNT_series_wm2"]
    control_erfc = control["FSNTC_series_wm2"] - control["FLNTC_series_wm2"]
    case_erfc = case["FSNTC_series_wm2"] - case["FLNTC_series_wm2"]
    with open(output_path, "w") as file:
        file.write("metric,control,control_ci95,case,case_minus_control,case_minus_control_ci95,units\n")
        file.write(
            "stratospheric_temperature,"
            f"{control['temperature_k']:.10g},"
            f"{control['temperature_ci95_k']:.10g},"
            f"{case['temperature_k']:.10g},"
            f"{temperature_anomaly:.10g},"
            f"{temperature_anomaly_ci95:.10g},"
            "K\n"
        )
        file.write(
            "ozone_column,"
            f"{control['ozone_column_du']:.10g},"
            f"{control['ozone_column_ci95_du']:.10g},"
            f"{case['ozone_column_du']:.10g},"
            f"{ozone_column_anomaly:.10g},"
            f"{ozone_column_anomaly_ci95:.10g},"
            "DU\n"
        )
        file.write(
            "ERF,"
            f"{np.nanmean(control_erf):.10g},"
            f"{confidence_interval_95(control_erf):.10g},"
            f"{np.nanmean(case_erf):.10g},"
            f"{erf_anomaly:.10g},"
            f"{erf_anomaly_ci95:.10g},"
            "W/m2\n"
        )
        file.write(
            "ERFC,"
            f"{np.nanmean(control_erfc):.10g},"
            f"{confidence_interval_95(control_erfc):.10g},"
            f"{np.nanmean(case_erfc):.10g},"
            f"{erfc_anomaly:.10g},"
            f"{erfc_anomaly_ci95:.10g},"
            "W/m2\n"
        )
    return output_path


def write_weighted_csv(control, case, anomalies):
    DATA_DIR.mkdir(exist_ok=True)
    output_path = DATA_DIR / "last40_pressure_area_weighted_anomalies.csv"
    control_erf = control["FSNT_series_wm2"] - control["FLNT_series_wm2"]
    case_erf = case["FSNT_series_wm2"] - case["FLNT_series_wm2"]
    control_erfc = control["FSNTC_series_wm2"] - control["FLNTC_series_wm2"]
    case_erfc = case["FSNTC_series_wm2"] - case["FLNTC_series_wm2"]
    rows = [
        (
            "stratospheric_temperature",
            control["temperature_series_k"],
            case["temperature_series_k"],
            anomalies["temperature"],
            "K",
        ),
        (
            "ozone_column",
            control["ozone_column_series_du"],
            case["ozone_column_series_du"],
            anomalies["ozone"],
            "DU",
        ),
        ("ERF", control_erf, case_erf, anomalies["erf"], "W/m2"),
        ("ERFC", control_erfc, case_erfc, anomalies["erfc"], "W/m2"),
    ]

    with open(output_path, "w") as file:
        file.write("metric,control,control_ci95,case,case_minus_control,case_minus_control_ci95,units\n")
        for metric, control_series, case_series, (anomaly, anomaly_ci95), units in rows:
            file.write(
                f"{metric},"
                f"{np.nanmean(control_series):.10g},"
                f"{confidence_interval_95(control_series):.10g},"
                f"{np.nanmean(case_series):.10g},"
                f"{anomaly:.10g},"
                f"{anomaly_ci95:.10g},"
                f"{units}\n"
            )
    return output_path


def print_weighted_summary(control, anomalies, output_path):
    years = control["years"]
    print()
    print("Pressure/area-weighted annual CONTROL_FILE/CASE_FILE version")
    if years is not None and len(years) > 0:
        print(f"Years used: {years[0]}-{years[-1]} ({LAST_N_YEARS} annual values)")
    else:
        print(f"Years used: last {LAST_N_YEARS} annual values")
    print("T weighting: pressure-layer thickness over [:51] and grid-cell area over lat/lon")
    print("O3 weighting: pressure-thickness column integration and grid-cell area over lat/lon")
    print("Flux weighting: grid-cell area over lat/lon")
    print()
    print(f"T anomaly:         {anomalies['temperature'][0]:.4f} +/- {anomalies['temperature'][1]:.4f} K")
    print(f"O3 column anomaly: {anomalies['ozone'][0]:.4f} +/- {anomalies['ozone'][1]:.4f} DU")
    print(f"ERF anomaly:       {anomalies['erf'][0]:.4f} +/- {anomalies['erf'][1]:.4f} W/m2")
    print(f"ERFC anomaly:      {anomalies['erfc'][0]:.4f} +/- {anomalies['erfc'][1]:.4f} W/m2")
    print()
    print(f"Saved: {output_path}")


def derived_flux_series(summary, prefix, clear_sky=False):
    if clear_sky:
        return summary[f"{prefix}_FSNTC_series_wm2"] - summary[f"{prefix}_FLNTC_series_wm2"]
    return summary[f"{prefix}_FSNT_series_wm2"] - summary[f"{prefix}_FLNT_series_wm2"]


def write_monthly_case_csv(
    summary,
    temperature_anomaly,
    temperature_anomaly_ci95,
    ozone_anomaly,
    ozone_anomaly_ci95,
    erf_anomaly,
    erf_anomaly_ci95,
    erfc_anomaly,
    erfc_anomaly_ci95,
):
    DATA_DIR.mkdir(exist_ok=True)
    output_path = DATA_DIR / "paper_match_JUL_8_30Gg_all_year_CC_vs_JUL_8_control.csv"
    control_erf = derived_flux_series(summary, "control")
    case_erf = derived_flux_series(summary, "case")
    control_erfc = derived_flux_series(summary, "control", clear_sky=True)
    case_erfc = derived_flux_series(summary, "case", clear_sky=True)
    with open(output_path, "w") as file:
        file.write("metric,control,control_ci95,case,case_minus_control,case_minus_control_ci95,units\n")
        file.write(
            "stratospheric_temperature,"
            f"{np.nanmean(summary['control_temperature_series_k']):.10g},"
            f"{confidence_interval_95(summary['control_temperature_series_k']):.10g},"
            f"{np.nanmean(summary['case_temperature_series_k']):.10g},"
            f"{temperature_anomaly:.10g},"
            f"{temperature_anomaly_ci95:.10g},"
            "K\n"
        )
        file.write(
            "ozone_column,"
            f"{np.nanmean(summary['control_ozone_column_series_du']):.10g},"
            f"{confidence_interval_95(summary['control_ozone_column_series_du']):.10g},"
            f"{np.nanmean(summary['case_ozone_column_series_du']):.10g},"
            f"{ozone_anomaly:.10g},"
            f"{ozone_anomaly_ci95:.10g},"
            "DU\n"
        )
        file.write(
            "ERF,"
            f"{np.nanmean(control_erf):.10g},"
            f"{confidence_interval_95(control_erf):.10g},"
            f"{np.nanmean(case_erf):.10g},"
            f"{erf_anomaly:.10g},"
            f"{erf_anomaly_ci95:.10g},"
            "W/m2\n"
        )
        file.write(
            "ERFC,"
            f"{np.nanmean(control_erfc):.10g},"
            f"{confidence_interval_95(control_erfc):.10g},"
            f"{np.nanmean(case_erfc):.10g},"
            f"{erfc_anomaly:.10g},"
            f"{erfc_anomaly_ci95:.10g},"
            "W/m2\n"
        )
    return output_path


def print_monthly_case_summary(
    summary,
    temperature_anomaly,
    temperature_anomaly_ci95,
    ozone_anomaly,
    ozone_anomaly_ci95,
    erf_anomaly,
    erf_anomaly_ci95,
    erfc_anomaly,
    erfc_anomaly_ci95,
    output_path,
):
    years = summary["years"]
    print()
    print(f"Monthly case: {MONTHLY_CASE_DIR.name} minus {MONTHLY_CONTROL_DIR.name}")
    print(f"Years used: {years[0]}-{years[-1]} ({len(years)} annual values after {MONTHLY_SPINUP_FILES} spinup files)")
    print("Paper-match T level slice: [:51] -> Python levels 0 through 50")
    print("Paper-match O3 level slice: all levels")
    print("Paper-match weighting: arithmetic means after the existing lon-mean preprocessing")
    print()
    # print(
    #     "Control T mean: "
    #     f"{np.nanmean(summary['control_temperature_series_k']):.6f} +/- "
    #     f"{confidence_interval_95(summary['control_temperature_series_k']):.6f} K"
    # )
    # print(f"Case T mean:    {np.nanmean(summary['case_temperature_series_k']):.6f} K")
    print(f"T anomaly:      {temperature_anomaly:.4f} +/- {temperature_anomaly_ci95:.4f} K")
    # print()
    # print(
    #     "Control O3 column mean: "
    #     f"{np.nanmean(summary['control_ozone_column_series_du']):.6f} +/- "
    #     f"{confidence_interval_95(summary['control_ozone_column_series_du']):.6f} DU"
    # )
    # print(f"Case O3 column mean:    {np.nanmean(summary['case_ozone_column_series_du']):.6f} DU")
    print(f"O3 column anomaly:      {ozone_anomaly:.4f} +/- {ozone_anomaly_ci95:.4f} DU")
    print(f"ERF anomaly:            {erf_anomaly:.4f} +/- {erf_anomaly_ci95:.4f} W/m2")
    print(f"ERFC anomaly:           {erfc_anomaly:.4f} +/- {erfc_anomaly_ci95:.4f} W/m2")
    print()
    print(f"Saved: {output_path}")


def print_summary(
    control,
    case,
    temperature_anomaly,
    temperature_anomaly_ci95,
    ozone_column_anomaly,
    ozone_column_anomaly_ci95,
    erf_anomaly,
    erf_anomaly_ci95,
    erfc_anomaly,
    erfc_anomaly_ci95,
    output_path,
):
    years = control["years"]
    if years is not None and len(years) > 0:
        print(f"Years used: {years[0]}-{years[-1]} ({LAST_N_YEARS} annual values)")
    else:
        print(f"Years used: last {LAST_N_YEARS} annual values")
    print("Paper-match T level slice: [:51] -> Python levels 0 through 50")
    print("Paper-match T weighting: arithmetic mean over time, level, latitude, longitude")
    print("Paper-match O3 level slice: all levels")
    print("Paper-match O3 weighting: arithmetic mean over latitude and longitude after column integration")
    print()
    # print(f"Control T mean: {control['temperature_k']:.6f} +/- {control['temperature_ci95_k']:.6f} K")
    # print(f"Case T mean:    {case['temperature_k']:.6f} K")
    print(f"T anomaly:      {temperature_anomaly:.4f} +/- {temperature_anomaly_ci95:.4f} K")
    # print()
    # print(f"Control O3 column mean: {control['ozone_column_du']:.6f} +/- {control['ozone_column_ci95_du']:.6f} DU")
    # print(f"Case O3 column mean:    {case['ozone_column_du']:.6f} DU")
    print(f"O3 column anomaly:      {ozone_column_anomaly:.4f} +/- {ozone_column_anomaly_ci95:.4f} DU")
    print(f"ERF anomaly:            {erf_anomaly:.4f} +/- {erf_anomaly_ci95:.4f} W/m2")
    print(f"ERFC anomaly:           {erfc_anomaly:.4f} +/- {erfc_anomaly_ci95:.4f} W/m2")
    print()
    print(f"Saved: {output_path}")


def main():
    control = summarize(CONTROL_FILE)
    case = summarize(CASE_FILE)

    temperature_diff_series = case["temperature_series_k"] - control["temperature_series_k"]
    temperature_anomaly = float(np.nanmean(temperature_diff_series))
    temperature_anomaly_ci95 = confidence_interval_95(temperature_diff_series)
    ozone_column_diff_series = case["ozone_column_series_du"] - control["ozone_column_series_du"]
    ozone_column_anomaly = float(np.nanmean(ozone_column_diff_series))
    ozone_column_anomaly_ci95 = confidence_interval_95(ozone_column_diff_series)
    erf_diff_series = (
        case["FSNT_series_wm2"]
        - control["FSNT_series_wm2"]
        - case["FLNT_series_wm2"]
        + control["FLNT_series_wm2"]
    )
    erfc_diff_series = (
        case["FSNTC_series_wm2"]
        - control["FSNTC_series_wm2"]
        - case["FLNTC_series_wm2"]
        + control["FLNTC_series_wm2"]
    )
    erf_anomaly = float(np.nanmean(erf_diff_series))
    erf_anomaly_ci95 = confidence_interval_95(erf_diff_series)
    erfc_anomaly = float(np.nanmean(erfc_diff_series))
    erfc_anomaly_ci95 = confidence_interval_95(erfc_diff_series)

    output_path = write_csv(
        control,
        case,
        temperature_anomaly,
        temperature_anomaly_ci95,
        ozone_column_anomaly,
        ozone_column_anomaly_ci95,
        erf_anomaly,
        erf_anomaly_ci95,
        erfc_anomaly,
        erfc_anomaly_ci95,
    )
    print_summary(
        control,
        case,
        temperature_anomaly,
        temperature_anomaly_ci95,
        ozone_column_anomaly,
        ozone_column_anomaly_ci95,
        erf_anomaly,
        erf_anomaly_ci95,
        erfc_anomaly,
        erfc_anomaly_ci95,
        output_path,
    )

    weighted_control = weighted_summarize(CONTROL_FILE)
    weighted_case = weighted_summarize(CASE_FILE)
    weighted_anomalies = summary_anomalies(weighted_control, weighted_case)
    weighted_output_path = write_weighted_csv(weighted_control, weighted_case, weighted_anomalies)
    print_weighted_summary(weighted_control, weighted_anomalies, weighted_output_path)

    monthly_summary = monthly_case_summary(MONTHLY_CONTROL_DIR, MONTHLY_CASE_DIR)
    monthly_temperature_diff = (
        monthly_summary["case_temperature_series_k"]
        - monthly_summary["control_temperature_series_k"]
    )
    monthly_ozone_diff = (
        monthly_summary["case_ozone_column_series_du"]
        - monthly_summary["control_ozone_column_series_du"]
    )
    monthly_temperature_anomaly = float(np.nanmean(monthly_temperature_diff))
    monthly_temperature_anomaly_ci95 = confidence_interval_95(monthly_temperature_diff)
    monthly_ozone_anomaly = float(np.nanmean(monthly_ozone_diff))
    monthly_ozone_anomaly_ci95 = confidence_interval_95(monthly_ozone_diff)
    monthly_erf_diff = (
        monthly_summary["case_FSNT_series_wm2"]
        - monthly_summary["control_FSNT_series_wm2"]
        - monthly_summary["case_FLNT_series_wm2"]
        + monthly_summary["control_FLNT_series_wm2"]
    )
    monthly_erfc_diff = (
        monthly_summary["case_FSNTC_series_wm2"]
        - monthly_summary["control_FSNTC_series_wm2"]
        - monthly_summary["case_FLNTC_series_wm2"]
        + monthly_summary["control_FLNTC_series_wm2"]
    )
    monthly_erf_anomaly = float(np.nanmean(monthly_erf_diff))
    monthly_erf_anomaly_ci95 = confidence_interval_95(monthly_erf_diff)
    monthly_erfc_anomaly = float(np.nanmean(monthly_erfc_diff))
    monthly_erfc_anomaly_ci95 = confidence_interval_95(monthly_erfc_diff)
    monthly_output_path = write_monthly_case_csv(
        monthly_summary,
        monthly_temperature_anomaly,
        monthly_temperature_anomaly_ci95,
        monthly_ozone_anomaly,
        monthly_ozone_anomaly_ci95,
        monthly_erf_anomaly,
        monthly_erf_anomaly_ci95,
        monthly_erfc_anomaly,
        monthly_erfc_anomaly_ci95,
    )
    print_monthly_case_summary(
        monthly_summary,
        monthly_temperature_anomaly,
        monthly_temperature_anomaly_ci95,
        monthly_ozone_anomaly,
        monthly_ozone_anomaly_ci95,
        monthly_erf_anomaly,
        monthly_erf_anomaly_ci95,
        monthly_erfc_anomaly,
        monthly_erfc_anomaly_ci95,
        monthly_output_path,
    )


if __name__ == "__main__":
    main()
