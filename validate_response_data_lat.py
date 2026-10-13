from analysis.emission_funcs import col_area
from netCDF4 import Dataset as NetCDFFile
import csv
import os
from pathlib import Path

import numpy as np
import scipy.stats as stats


# Validate direct global averages from lon files against precomputed
# latitude-weighted files in results_cloud/lat.

PROJECT_DIR = Path(__file__).resolve().parent
LON_ROOT = PROJECT_DIR / "results_cloud" / "lon"
LAT_ROOT = PROJECT_DIR / "results_cloud" / "lat"
OUTPUT_PATH = PROJECT_DIR / "data_file" / "response_data_lat_validation.csv"

LON_SPINUP_MONTHS = 120
LAT_SPINUP_MONTHS = 60
CONTROL_CASE = "JUL_8_control"
RESPONSES = ["T", "O3", "FLNT", "FSNT", "FLNTC", "FSNTC"]

CASE_LIST_INITIAL = [
    "all_year", "all_year", "summer", "winter", "all_year",
    "all_year", "summer", "winter", "all_year", "summer", "winter",
    "all_year", "all_year", "summer", "winter",
]
LOCATION_LIST = [
    "CC_BC_only", "SA", "SA", "SA", "AU", "EQ", "EQ", "EQ",
    "CC", "CC", "CC", "UK", "RU", "RU", "RU",
]
LAT_LIST = [29, -55, -55, -55, -29, 0, 0, 0, 29, 29, 29, 55, 70, 70, 70]
DATE_LIST = [
    "AUG_13", "JUL_8", "JUL_8", "JUL_8", "AUG_13",
    "AUG_13", "AUG_13", "AUG_13", "JUL_8", "JUL_8", "JUL_8",
    "AUG_13", "JUL_8", "JUL_8", "JUL_8",
]

# Lat-averaged BC-only files were generated with a different historical name.
LAT_CASE_OVERRIDES = {
    "AUG_13_30Gg_all_year_CC_BC_only": "AUG_13_CC_all_year_30Gg_BC_only",
}

ABS_TOL = 1e-5
REL_TOL = 1e-4


def latitude_area_weights():
    area, _, _ = col_area()
    return np.mean(area, axis=1)


def global_area_weighted_mean(values):
    values = np.ma.masked_invalid(np.ma.array(values))
    zonal_area = latitude_area_weights()

    if values.shape[-1] == len(zonal_area):
        weights = zonal_area
    elif values.ndim >= 2 and values.shape[-2] == len(zonal_area):
        weights = zonal_area[:, np.newaxis]
    else:
        return np.ma.mean(values)

    weights = np.broadcast_to(weights, values.shape)
    weights = np.ma.array(weights, mask=np.ma.getmaskarray(values))
    return np.ma.sum(values * weights) / np.ma.sum(weights)


def effective_dof(data, min_dof=2.0):
    data = np.asarray(data, dtype=float)
    data = data[np.isfinite(data)]
    n = len(data)
    if n <= 1:
        return None
    if n <= min_dof + 1 or np.isclose(np.std(data), 0):
        return n - 1

    rho1 = np.corrcoef(data[1:], data[:-1])[0, 1]
    if not np.isfinite(rho1) or rho1 <= 0:
        return n - 1

    neff = n * (1 - rho1) / (1 + rho1)
    dof = neff - 1
    return min(max(dof, min_dof), n - 1)


def confidence_error(data, dof):
    data = np.asarray(data, dtype=float)
    data = data[np.isfinite(data)]
    if dof is None or len(data) <= 1:
        return 0.0

    sem = stats.sem(data)
    if not np.isfinite(sem):
        return 0.0

    return float(stats.t.ppf(0.975, df=dof) * sem)


def annual_average(monthly_data):
    return [
        sum(monthly_data[i:i + 12]) / 12
        for i in range(0, len(monthly_data) - len(monthly_data) % 12, 12)
    ]


def nc_files(case_dir, spinup_months):
    files = []
    for root, _, filenames in os.walk(case_dir):
        for filename in filenames:
            if filename.endswith(".nc"):
                files.append(Path(root) / filename)
    files.sort(key=lambda path: path.name.lower())
    return files[spinup_months:]


def monthly_lon_value(dataset, response):
    return global_area_weighted_mean(dataset.variables[response][:])


def monthly_lat_value(dataset, response):
    return np.ma.mean(np.ma.masked_invalid(dataset.variables[response][:]))


def avg_case_data(case_dir, response, source):
    spinup_months = LON_SPINUP_MONTHS if source == "lon" else LAT_SPINUP_MONTHS
    value_func = monthly_lon_value if source == "lon" else monthly_lat_value
    monthly_values = []

    files = nc_files(case_dir, spinup_months)
    if not files:
        raise FileNotFoundError(f"No post-spin-up .nc files found in {case_dir}")

    for case_file in files:
        dataset = NetCDFFile(str(case_file))
        try:
            monthly_values.append(float(np.ma.filled(value_func(dataset, response), np.nan)))
        finally:
            dataset.close()

    return annual_average(monthly_values), monthly_values


def processing(control_dir, case_dir, response, source):
    control_annual, _ = avg_case_data(control_dir, response, source)
    case_annual, _ = avg_case_data(case_dir, response, source)

    n_years = min(len(control_annual), len(case_annual))
    diff = np.array(case_annual[:n_years]) - np.array(control_annual[:n_years])
    mean = float(np.mean(diff))
    error = confidence_error(diff, effective_dof(diff))
    return mean, error


def lon_case_name(date, season, location):
    return f"{date}_30Gg_{season}_{location}"


def lat_case_name(date, season, location):
    name = lon_case_name(date, season, location)
    return LAT_CASE_OVERRIDES.get(name, name)


def validation_status(abs_diff, rel_diff):
    if abs_diff <= ABS_TOL or rel_diff <= REL_TOL:
        return "PASS"
    return "REVIEW"


def add_base_rows(rows, results, errors):
    for i, season in enumerate(CASE_LIST_INITIAL):
        location = LOCATION_LIST[i]
        date = DATE_LIST[i]
        launch_lat = LAT_LIST[i]
        lon_name = lon_case_name(date, season, location)
        lat_name = lat_case_name(date, season, location)

        lon_dir = LON_ROOT / lon_name
        lat_dir = LAT_ROOT / lat_name
        lon_control_dir = LON_ROOT / CONTROL_CASE
        lat_control_dir = LAT_ROOT / CONTROL_CASE

        if not lon_dir.exists() or not lat_dir.exists():
            print(f"Skipping missing case: lon={lon_dir.exists()} lat={lat_dir.exists()} {lon_name}")
            continue

        for response in RESPONSES:
            lon_mean, lon_error = processing(lon_control_dir, lon_dir, response, "lon")
            lat_mean, lat_error = processing(lat_control_dir, lat_dir, response, "lat")

            results["lon"][response].append(lon_mean)
            results["lat"][response].append(lat_mean)
            errors["lon"][response].append(lon_error)
            errors["lat"][response].append(lat_error)

            abs_diff = abs(lon_mean - lat_mean)
            rel_diff = abs_diff / max(abs(lat_mean), 1e-12)
            rows.append({
                "response": response,
                "case": lon_name,
                "lat_case": lat_name,
                "launch_lat": launch_lat,
                "season": season,
                "location": location,
                "lon_mean": lon_mean,
                "lat_mean": lat_mean,
                "abs_diff": abs_diff,
                "rel_diff": rel_diff,
                "lon_error": lon_error,
                "lat_error": lat_error,
                "status": validation_status(abs_diff, rel_diff),
            })


def write_rows(rows):
    OUTPUT_PATH.parent.mkdir(exist_ok=True)
    fieldnames = [
        "response", "case", "lat_case", "launch_lat", "season", "location",
        "lon_mean", "lat_mean", "abs_diff", "rel_diff",
        "lon_error", "lat_error", "status",
    ]
    with open(OUTPUT_PATH, "w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def print_summary(rows):
    for response in RESPONSES:
        response_rows = [row for row in rows if row["response"] == response]
        if not response_rows:
            continue
        max_abs = max(row["abs_diff"] for row in response_rows)
        max_rel = max(row["rel_diff"] for row in response_rows)
        n_review = sum(row["status"] == "REVIEW" for row in response_rows)
        print(
            f"{response}: max_abs={max_abs:.6g}, "
            f"max_rel={max_rel:.6g}, review={n_review}/{len(response_rows)}"
        )


def main():
    os.chdir(PROJECT_DIR)
    rows = []
    results = {source: {response: [] for response in RESPONSES} for source in ["lon", "lat"]}
    errors = {source: {response: [] for response in RESPONSES} for source in ["lon", "lat"]}

    add_base_rows(rows, results, errors)
    write_rows(rows)
    print_summary(rows)
    print(f"Saved validation table: {OUTPUT_PATH}")
    print("Validation uses direct global averages only; no tropopause or column transformations are applied.")


if __name__ == "__main__":
    main()
