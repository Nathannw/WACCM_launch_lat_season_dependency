from functools import lru_cache
from pathlib import Path
import csv
import math
import os
import sys

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
import numpy as np
import scipy.stats as stats

from analysis.figure_style import apply_journal_style
from analysis.response_data import avg_case_data, effective_dof


# Pairwise comparison of year-round launch-latitude cases.
#
# Differences are computed from annual global response anomalies:
#   (case A - control) - (case B - control)
# which is equivalent to case A - case B for cases sharing the same control.

PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_DIR / "data_file"
OUTPUT_DIR = DATA_DIR / "year_round_pairwise_matrices"
PLOT_DIR = PROJECT_DIR / "fig" / "year_round_pairwise_matrices"
CONTROL_DIR = PROJECT_DIR / "results_cloud" / "lon" / "JUL_8_control"

YEAR_ROUND_CASES = [
    {"label": "SA", "latitude": -55, "directory": PROJECT_DIR / "results_cloud" / "lon" / "JUL_8_30Gg_all_year_SA"},
    {"label": "AU", "latitude": -29, "directory": PROJECT_DIR / "results_cloud" / "lon" / "AUG_13_30Gg_all_year_AU"},
    {"label": "EQ", "latitude": 0, "directory": PROJECT_DIR / "results_cloud" / "lon" / "AUG_13_30Gg_all_year_EQ"},
    {"label": "CC", "latitude": 29, "directory": PROJECT_DIR / "results_cloud" / "lon" / "JUL_8_30Gg_all_year_CC"},
    {"label": "UK", "latitude": 55, "directory": PROJECT_DIR / "results_cloud" / "lon" / "AUG_13_30Gg_all_year_UK"},
    {"label": "RU", "latitude": 70, "directory": PROJECT_DIR / "results_cloud" / "lon" / "JUL_8_30Gg_all_year_RU"},
]

BASE_RESPONSES = ["T", "O3", "FLNT", "FSNT", "FLNTC", "FSNTC"]
DERIVED_RESPONSES = {
    "ERF": ("FSNT", "FLNT"),
    "ERFC": ("FSNTC", "FLNTC"),
}
RESPONSES = ["T", "O3", "ERF", "ERFC", "FSNT", "FLNT", "FSNTC", "FLNTC"]
PANEL_LABELS = {
    "T": "(b)",
    "O3": "(d)",
    "ERF": "(a)",
    "ERFC": "(b)",
    "FSNT": "",
    "FLNT": "",
    "FSNTC": "",
    "FLNTC": "",
}
RESPONSE_LABELS = {
    "T": "Stratospheric temperature anomaly (K)",
    "O3": "Ozone anomaly (DU)",
    "ERF": "All-sky ERF (W m-2)",
    "ERFC": "Clear-sky ERF (W m-2)",
    "FSNT": "All-sky SW anomaly (W m-2)",
    "FLNT": "All-sky LW anomaly (W m-2)",
    "FSNTC": "Clear-sky SW anomaly (W m-2)",
    "FLNTC": "Clear-sky LW anomaly (W m-2)",
}


def nc_files(case_dir):
    files = []
    for root, _, filenames in os.walk(case_dir):
        for filename in filenames:
            if filename.endswith(".nc"):
                files.append(Path(root) / filename)
    return sorted(files, key=lambda path: path.name.lower())


def validate_inputs():
    missing = []
    if not nc_files(CONTROL_DIR):
        missing.append(str(CONTROL_DIR))
    for case in YEAR_ROUND_CASES:
        if not nc_files(case["directory"]):
            missing.append(str(case["directory"]))

    if missing:
        joined = "\n  ".join(missing)
        raise FileNotFoundError(f"Missing NetCDF files in:\n  {joined}")


@lru_cache(maxsize=None)
def annual_values(directory, response):
    values, _ = avg_case_data(Path(directory), response)
    return np.array(values, dtype=float)


@lru_cache(maxsize=None)
def annual_anomaly(case_label, response):
    case = next(item for item in YEAR_ROUND_CASES if item["label"] == case_label)

    if response in DERIVED_RESPONSES:
        positive_response, negative_response = DERIVED_RESPONSES[response]
        return (
            annual_anomaly(case_label, positive_response)
            - annual_anomaly(case_label, negative_response)
        )

    case_values = annual_values(str(case["directory"]), response)
    control_values = annual_values(str(CONTROL_DIR), response)
    n_years = min(len(case_values), len(control_values))
    if n_years == 0:
        raise ValueError(f"No post-spin-up annual data for {case['label']} {response}")
    if len(case_values) != len(control_values):
        print(
            f"Warning: {case['label']} {response} has {len(case_values)} annual values; "
            f"control has {len(control_values)}. Using {n_years} paired years."
        )

    return case_values[:n_years] - control_values[:n_years]


def two_sided_p_value(diff_series, dof):
    diff_series = np.asarray(diff_series, dtype=float)
    diff_series = diff_series[np.isfinite(diff_series)]
    if len(diff_series) <= 1:
        return math.nan

    sem = stats.sem(diff_series)
    if not np.isfinite(sem) or np.isclose(sem, 0.0):
        return 1.0 if np.isclose(np.mean(diff_series), 0.0) else 0.0

    if dof is None:
        return math.nan

    t_stat = np.mean(diff_series) / sem
    return float(2 * stats.t.sf(abs(t_stat), dof))


def confidence_interval_95(diff_series, dof):
    diff_series = np.asarray(diff_series, dtype=float)
    diff_series = diff_series[np.isfinite(diff_series)]
    if len(diff_series) <= 1 or dof is None:
        return math.nan

    sem = stats.sem(diff_series)
    if not np.isfinite(sem):
        return math.nan

    return float(stats.t.ppf(0.975, df=dof) * sem)


def pairwise_result(response, case_a, case_b):
    series_a = annual_anomaly(case_a["label"], response)
    series_b = annual_anomaly(case_b["label"], response)
    n_years = min(len(series_a), len(series_b))
    diff_series = series_a[:n_years] - series_b[:n_years]
    dof = effective_dof(diff_series)
    return {
        "response": response,
        "case_a": case_a["label"],
        "latitude_a": case_a["latitude"],
        "case_b": case_b["label"],
        "latitude_b": case_b["latitude"],
        "difference_a_minus_b": float(np.mean(diff_series)),
        "difference_ci95": confidence_interval_95(diff_series, dof),
        "p_value": two_sided_p_value(diff_series, dof),
        "effective_dof": math.nan if dof is None else float(dof),
        "n_years": n_years,
    }


def format_number(value, precision=6):
    if value is None or not np.isfinite(value):
        return ""
    return f"{value:.{precision}f}"


def format_p_value(value):
    if value is None or not np.isfinite(value):
        return "p=NA"
    if value < 0.05:
        return "p<0.05"
    return f"p={value:.3f}"


def matrix_arrays(rows):
    labels = [case["label"] for case in YEAR_ROUND_CASES]
    lookup = {(row["case_a"], row["case_b"]): row for row in rows}
    difference = np.array(
        [
            [lookup[(label_a, label_b)]["difference_a_minus_b"] for label_b in labels]
            for label_a in labels
        ],
        dtype=float,
    )
    ci95 = np.array(
        [
            [lookup[(label_a, label_b)]["difference_ci95"] for label_b in labels]
            for label_a in labels
        ],
        dtype=float,
    )
    p_value = np.array(
        [
            [lookup[(label_a, label_b)]["p_value"] for label_b in labels]
            for label_a in labels
        ],
        dtype=float,
    )
    return labels, difference, ci95, p_value


def plot_response_matrix(response, rows, panel_label=None):
    PLOT_DIR.mkdir(parents=True, exist_ok=True)
    apply_journal_style()

    labels, difference, ci95, p_value = matrix_arrays(rows)
    tick_labels = ["55° S", "29° S", "0° N", "29° N", "55° N", "70° N"]
    max_abs = np.nanmax(np.abs(difference))
    if not np.isfinite(max_abs) or np.isclose(max_abs, 0.0):
        max_abs = 1.0

    fig, ax = plt.subplots(figsize=(7.6, 6.0))
    norm = TwoSlopeNorm(vmin=-max_abs, vcenter=0.0, vmax=max_abs)
    cmap_name = "RdBu" if response == "O3" else "RdBu_r"
    image = ax.imshow(difference, cmap=cmap_name, norm=norm)
    cbar = fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label(f"{RESPONSE_LABELS[response]}\ncase A - case B")

    ax.set_xticks(np.arange(len(labels)))
    ax.set_yticks(np.arange(len(labels)))
    ax.set_xticklabels(tick_labels)
    ax.set_yticklabels(tick_labels)
    ax.set_xlabel("Launch Latitude: Case B")
    ax.set_ylabel("Launch Latitude: Case A")
    # ax.set_title(f"{response}: pairwise differences (mean ± 95% CI)")
    if panel_label:
        ax.text(
            -0.14,
            1.08,
            panel_label,
            transform=ax.transAxes,
            ha="left",
            va="bottom",
            fontweight="bold",
            clip_on=False,
        )
    ax.invert_yaxis()

    ax.set_xticks(np.arange(-0.5, len(labels), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(labels), 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=1.0)
    ax.tick_params(which="minor", bottom=False, left=False)

    threshold = max_abs * 0.55
    for i in range(len(labels)):
        for j in range(len(labels)):
            value = difference[i, j]
            text_color = "white" if abs(value) > threshold else "black"
            significance = "*" if p_value[i, j] < 0.05 and i != j else ""
            ax.text(
                j,
                i,
                f"{value:.2f}{significance}\n± {ci95[i, j]:.2f}\n{format_p_value(p_value[i, j])}",
                ha="center",
                va="center",
                color=text_color,
                fontsize=8,
            )

    fig.tight_layout()
    for extension in ["pdf"]:
        output_file = PLOT_DIR / f"{response}_difference_p_value_matrix.{extension}"
        fig.savefig(output_file)
        print(f"Saved: {output_file}")
    plt.close(fig)


def write_long_form(rows):
    DATA_DIR.mkdir(exist_ok=True)
    output_file = DATA_DIR / "year_round_pairwise_response_comparison.csv"
    fieldnames = [
        "response",
        "case_a",
        "latitude_a",
        "case_b",
        "latitude_b",
        "difference_a_minus_b",
        "difference_ci95",
        "p_value",
        "effective_dof",
        "n_years",
    ]
    with open(output_file, "w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved: {output_file}")


def write_response_matrices(response, rows):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    labels = [case["label"] for case in YEAR_ROUND_CASES]
    lookup = {(row["case_a"], row["case_b"]): row for row in rows}

    matrix_specs = [
        ("difference", "difference_a_minus_b"),
        ("ci95", "difference_ci95"),
        ("p_value", "p_value"),
    ]
    for suffix, key in matrix_specs:
        output_file = OUTPUT_DIR / f"{response}_{suffix}.csv"
        with open(output_file, "w", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(["case_a_minus_case_b"] + labels)
            for label_a in labels:
                writer.writerow(
                    [label_a]
                    + [format_number(lookup[(label_a, label_b)][key]) for label_b in labels]
                )
        print(f"Saved: {output_file}")

    output_file = OUTPUT_DIR / f"{response}_difference_p_value.csv"
    with open(output_file, "w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["case_a_minus_case_b"] + labels)
        for label_a in labels:
            row = [label_a]
            for label_b in labels:
                item = lookup[(label_a, label_b)]
                diff = format_number(item["difference_a_minus_b"])
                ci95 = format_number(item["difference_ci95"])
                p_value = format_number(item["p_value"])
                row.append(f"{diff} ± {ci95} (p={p_value})")
            writer.writerow(row)
    print(f"Saved: {output_file}")


def main():
    os.chdir(PROJECT_DIR)
    validate_inputs()

    all_rows = []
    for response in RESPONSES:
        response_rows = []
        print(f"Processing {response}...")
        for case_a in YEAR_ROUND_CASES:
            for case_b in YEAR_ROUND_CASES:
                row = pairwise_result(response, case_a, case_b)
                response_rows.append(row)
                all_rows.append(row)
        write_response_matrices(response, response_rows)
        plot_response_matrix(
            response,
            response_rows,
            panel_label=PANEL_LABELS.get(response),
        )

    write_long_form(all_rows)
    print("Year-round pairwise comparison complete!")


if __name__ == "__main__":
    main()
