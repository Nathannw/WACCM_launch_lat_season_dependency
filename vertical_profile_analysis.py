from analysis.emission_funcs import col_area
from analysis.func import find_trop_func, pressure_level_weights, pressure_weighted_mean
from analysis.figure_style import COLORS, panel_label
from netCDF4 import Dataset as NetCDFFile
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import os
import scipy.stats as stats


# To generate and plot climate responses for equatorial vertical-profile cases.

PROJECT_DIR = Path(__file__).resolve().parent
DATA_DIR = PROJECT_DIR / "data_file"
FIG_DIR = PROJECT_DIR / "fig" / "response"
CONTROL_DIR = PROJECT_DIR / "results_cloud" / "lon" / "JUL_8_control"

CASES = [
    ("Profile", PROJECT_DIR / "results_cloud" / "lon" / "AUG_13_30Gg_all_year_EQ"),
    ("20 km", PROJECT_DIR / "results_cloud" / "lon" / "AUG_13_30Gg_all_year_EQ_20km"),
    ("30 km", PROJECT_DIR / "results_cloud" / "lon" / "AUG_13_30Gg_all_year_EQ_30km"),
    ("70 km", PROJECT_DIR / "results_cloud" / "lon" / "AUG_13_30Gg_all_year_EQ_70km"),
]

BASE_RESPONSES = ["T", "O3", "FLNT", "FSNT", "FLNTC", "FSNTC"]
PLOT_RESPONSES = ["T", "O3", "ERF", "ERFC", "FSNT", "FLNT", "FSNTC", "FLNTC"]
LABELS = [
    "Stratospheric temperature anomaly (K)",
    "Ozone anomaly (DU)",
    r"All-sky ERF ($\mathrm{W}\ \mathrm{m}^{-2}$)",
    r"Clear-sky ERF ($\mathrm{W}\ \mathrm{m}^{-2}$)",
    r"All-sky SW anomaly ($\mathrm{W}\ \mathrm{m}^{-2}$)",
    r"All-sky LW anomaly ($\mathrm{W}\ \mathrm{m}^{-2}$)",
    r"Clear-sky SW anomaly ($\mathrm{W}\ \mathrm{m}^{-2}$)",
    r"Clear-sky LW anomaly ($\mathrm{W}\ \mathrm{m}^{-2}$)",
]
TAGS = [panel_label(letter) for letter in ["a", "b", "c", "d", "e", "f", "g", "h"]]


def area_weighted_mean(values, weights):
    values = np.ma.masked_invalid(np.ma.array(values))
    weights = np.ma.array(weights, mask=np.ma.getmaskarray(values))
    return np.ma.sum(values * weights) / np.ma.sum(weights)


def global_area_weighted_mean(values):
    values = np.ma.masked_invalid(np.ma.array(values))
    area, _, _ = col_area()
    zonal_area = np.mean(area, axis=1)

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


def nc_files(case_dir):
    files = []
    for root, _, filenames in os.walk(case_dir):
        for filename in filenames:
            if filename.endswith(".nc"):
                files.append(Path(root) / filename)
    files.sort(key=lambda path: path.name.lower())
    return files


def annual_average(monthly_data):
    return [
        sum(monthly_data[i:i + 12]) / 12
        for i in range(0, len(monthly_data) - len(monthly_data) % 12, 12)
    ]


def avg_case_data(case_dir, response):
    case_files = nc_files(case_dir)[120:]
    data_list = []

    for case_file in case_files:
        dataset = NetCDFFile(str(case_file))
        try:
            if response in ["FLNT", "FSNT", "FLNTC", "FSNTC", "FLN200", "FLN200C", "FSN200", "FSN200C"]:
                aggregated_data = global_area_weighted_mean(dataset.variables[response][:])
            elif response == "O3":
                mol_air = dataset.variables["MASS"][0, :, :] * 1000 / 28.97
                ozone = dataset.variables["O3"][0, :, :]
                mol_ozone = ozone * mol_air
                area, _, _ = col_area()
                zonal_area = np.mean(area, axis=1)
                mol_ozone_per_cm2 = np.sum(mol_ozone / zonal_area, axis=0)
                du = mol_ozone_per_cm2 / 4.4615e-8
                aggregated_data = area_weighted_mean(du, zonal_area)
            elif response in ["CLOUD", "CLDICE", "CLDLIQ", "T"]:
                aggregated_data_temp = []
                response_data = dataset.variables[response][:]
                pressure_weights = pressure_level_weights(dataset)
                nlat = response_data.shape[-1]
                for lat_idx in range(nlat):
                    idx_lower, idx_upper, ratio = find_trop_func(dataset, lat_idx)
                    x1 = pressure_weighted_mean(
                        response_data[:, 25:idx_lower[0], lat_idx],
                        pressure_weights[25:idx_lower[0]],
                    )
                    x2 = pressure_weighted_mean(
                        response_data[:, 25:idx_upper[0], lat_idx],
                        pressure_weights[25:idx_upper[0]],
                    )
                    aggregated_data_1 = x1 * (1 - ratio[0]) + x2 * ratio[0]
                    aggregated_data_temp.append(aggregated_data_1)

                area, _, _ = col_area()
                zonal_area = np.mean(area, axis=1)
                aggregated_data = area_weighted_mean(aggregated_data_temp, zonal_area[:nlat])
            else:
                raise ValueError(f"Unsupported response: {response}")
        finally:
            dataset.close()

        data_list.append(float(np.ma.filled(aggregated_data, np.nan)))

    return annual_average(data_list), data_list


def processing(dir_control, dir_case, response):
    data_list_control, _ = avg_case_data(dir_control, response)
    data_list, _ = avg_case_data(dir_case, response)

    n_years = min(len(data_list_control), len(data_list))
    if n_years == 0:
        raise ValueError(f"No post-spin-up annual data for {dir_case}")
    if len(data_list_control) != len(data_list):
        print(
            f"Warning: {dir_case.name} has {len(data_list)} annual values; "
            f"control has {len(data_list_control)}. Using {n_years} paired years."
        )

    diff = np.array(data_list[:n_years]) - np.array(data_list_control[:n_years])
    mean = np.mean(diff)
    dof_eff = effective_dof(diff)
    error = confidence_error(diff, dof_eff)
    print(f"{dir_case.name} {response}: {mean:.3f} +/- {error:.2f}")
    return float(mean), float(error)


def write_to_file(filename, values):
    DATA_DIR.mkdir(exist_ok=True)
    labels = [label for label, _ in CASES]
    directories = [path.name for _, path in CASES]
    with open(DATA_DIR / filename, "w") as file:
        file.write(f"{labels}\n")
        file.write(f"{directories}\n")
        file.write(f"{values}")
    print(f"Saved: {DATA_DIR / filename}")


def read_profile_file(filename):
    import ast

    with open(DATA_DIR / filename, "r") as file:
        lines = [ast.literal_eval(line.strip()) for line in file]
    return lines[0], lines[1], np.array(lines[2], dtype=float)


def plot_response(case_labels, data, data_error, label, alphabet, response, legend):
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    x = np.arange(len(case_labels))

    plt.figure(figsize=(6, 4))
    plt.errorbar(
        x,
        data,
        yerr=data_error,
        color=COLORS["green"],
        marker="o",
        markersize=7,
        linestyle="none",
        capsize=3,
        elinewidth=1,
        label="Year-round",
        alpha=0.75,
    )

    plt.xlabel("Vertical emission case", fontsize=15)
    plt.ylabel(label, fontsize=15)
    plt.xticks(x, case_labels, fontsize=15)
    plt.yticks(fontsize=15)

    if legend:
        plt.legend(frameon=True, fontsize=15)

    for spine in plt.gca().spines.values():
        spine.set_linewidth(1.0)

    plt.text(
        0.02,
        1.05,
        f"{alphabet}",
        transform=plt.gca().transAxes,
        fontsize=15,
        fontweight="bold",
    )
    plt.tight_layout()
    plt.savefig(FIG_DIR / f"vertical_profile_{response}.pdf")
    plt.close()


def generate_data():
    results_by_response = {}
    errors_by_response = {}

    for response in BASE_RESPONSES:
        results = []
        errors = []
        for _, dir_case in CASES:
            if not nc_files(CONTROL_DIR):
                raise FileNotFoundError(f"No .nc files found in {CONTROL_DIR}")
            if not nc_files(dir_case):
                raise FileNotFoundError(f"No .nc files found in {dir_case}")
            mean, error = processing(CONTROL_DIR, dir_case, response)
            results.append(mean)
            errors.append(error)

        results_by_response[response] = results
        errors_by_response[response] = errors
        write_to_file(f"vertical_profile_{response}.txt", results)
        write_to_file(f"vertical_profile_{response}_error.txt", errors)

    erf = (np.array(results_by_response["FSNT"]) - np.array(results_by_response["FLNT"])).tolist()
    erf_error = np.sqrt(
        np.array(errors_by_response["FSNT"]) ** 2 + np.array(errors_by_response["FLNT"]) ** 2
    ).tolist()
    erfc = (np.array(results_by_response["FSNTC"]) - np.array(results_by_response["FLNTC"])).tolist()
    erfc_error = np.sqrt(
        np.array(errors_by_response["FSNTC"]) ** 2 + np.array(errors_by_response["FLNTC"]) ** 2
    ).tolist()

    write_to_file("vertical_profile_ERF.txt", erf)
    write_to_file("vertical_profile_ERF_error.txt", erf_error)
    write_to_file("vertical_profile_ERFC.txt", erfc)
    write_to_file("vertical_profile_ERFC_error.txt", erfc_error)


def generate_plots():
    for k, response in enumerate(PLOT_RESPONSES):
        case_labels, _, data = read_profile_file(f"vertical_profile_{response}.txt")
        _, _, data_error = read_profile_file(f"vertical_profile_{response}_error.txt")
        plot_response(
            case_labels,
            data,
            data_error,
            LABELS[k],
            TAGS[k],
            response,
            legend=(k == 0),
        )


def main():
    os.chdir(PROJECT_DIR)
    generate_data()
    generate_plots()
    print("Vertical profile analysis complete!")


if __name__ == "__main__":
    main()
