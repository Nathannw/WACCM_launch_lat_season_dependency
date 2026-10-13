import ast
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.interpolate import PchipInterpolator
from analysis.func import *
from analysis.figure_style import COLORS, panel_label

# To plot emission burdens and BC distribution.

latitudes = np.array([-55, -29, 0, 29, 55, 70])
output_dir = Path("fig/burdens_distribution")
distribution_case_dir = output_dir / "bc_distribution_cases"

season_titles = {
    "all_year": "Year-round",
    "summer": "Boreal summer-only",
    "winter": "Boreal winter-only",
}

season_order = ["all_year", "summer", "winter"]

launch_lat_colors = {
    -55: COLORS["sky_blue"],
    -29: COLORS["vermillion"],
    0: COLORS["purple"],
    29: COLORS["green"],
    55: COLORS["orange"],
    70: 'black',
}

season_line_styles = {
    "all_year": "-",
    "summer": "-",
    "winter": "-",
}

def plot(year_round, summer, winter, xlabel, ylabel, lim, label, savename, legend,
         year_round_error=None, summer_error=None, winter_error=None):
    # Create figure
    plt.figure(figsize=(6, 4))

    # Scatter plots (journal style: distinct but not flashy)
    plot_points(latitudes, year_round, year_round_error, COLORS["green"], "o", "Year-round")
    plot_points(latitudes, summer, summer_error, COLORS["orange"], "s", "Boreal summer-only")
    plot_points(latitudes, winter, winter_error, COLORS["blue"], "^", "Boreal winter-only")
    x_range = np.linspace(latitudes.min(), latitudes.max(), 400)

    f = PchipInterpolator(latitudes, year_round)
    y_pchip = f(x_range)

    # plt.plot(x_range, y_pchip, linestyle = '--', color = 'black')

    # Axis labels and title
    plt.ylabel(xlabel, fontsize=13)
    plt.xlabel(ylabel, fontsize=13)
    plt.xlim(lim)

    # Ticks formatting
    plt.xticks(fontsize=13)
    plt.yticks(fontsize=13)

    if legend:
        # Legend formatting
        plt.legend(frameon=True, fontsize=13)

    # Axis border (journal look: all sides visible, no grid)
    for spine in plt.gca().spines.values():
        spine.set_linewidth(1.0)

    plt.text(0.02, 1.05, label, transform=plt.gca().transAxes, fontsize=13, fontweight='bold')

    # Tight layout for publication
    plt.tight_layout()

    # Save high-resolution version (for journal submission)
    output_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_dir / savename)

    # # Show plot
    # plt.show()

def plot_points(x, y, xerr, color, marker, label):
    if xerr is None:
        plt.scatter(y, x, color=color, marker=marker, s=50, label=label, alpha=0.75)
    else:
        plt.errorbar(
            y, x, xerr=xerr,
            color=color, marker=marker, markersize=7,
            linestyle="none", capsize=3, elinewidth=1,
            label=label, alpha=0.75
        )

def to_float_array(values):
    return np.asarray(values, dtype=float)

def read_distribution_file(filename):
    with open(filename, "r") as file:
        read_data = [ast.literal_eval(line.strip()) for line in file]

    case_lats = read_data[0]
    seasons = read_data[1]
    locations = read_data[2]
    distributions = read_data[3]

    records = []
    for launch_lat, season, location, distribution in zip(case_lats, seasons, locations, distributions):
        records.append({
            "launch_lat": launch_lat,
            "season": season,
            "location": location,
            "distribution": distribution,
        })
    return records

def read_scalar_records(filename):
    with open(filename, "r") as file:
        read_data = [ast.literal_eval(line.strip()) for line in file]

    case_lats = read_data[0]
    seasons = read_data[1]
    locations = read_data[2]
    values = read_data[3]
    return {
        (launch_lat, season, location): value
        for launch_lat, season, location, value in zip(case_lats, seasons, locations, values)
    }

def case_label(record):
    location = record["location"]
    launch_lat = record["launch_lat"]
    if location == "CC_BC_only":
        location_plot =  "BC-only"
    else:
        location_plot = ""
    hemisphere = "N" if launch_lat >= 0 else "S"
    return f"{abs(launch_lat)}° {hemisphere} {location_plot}"

def case_color(record):
    if record["location"] == "CC_BC_only":
        return COLORS["green"]
    return launch_lat_colors.get(record["launch_lat"], COLORS["green"])

def case_line_style(record):
    if record["location"] == "CC_BC_only":
        return "-."
    return season_line_styles.get(record["season"], "-")

def style_distribution_axis(ax):
    for spine in ax.spines.values():
        spine.set_linewidth(1.0)
    ax.tick_params(labelsize=10)

def add_legend_if_needed(ax):
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        ax.legend(handles, labels, frameon=False, fontsize=8)

def step_values(edges, heights):
    return np.repeat(edges, 2)[1:-1], np.repeat(heights, 2)

def plot_latitude_distribution(ax, record):
    distribution = record["distribution"]
    latitude_edges = to_float_array(distribution["latitude_edges_deg"])
    bc_density = to_float_array(distribution["bc_mass_density_kg_per_degree"]) / 1e6
    x_step, y_step = step_values(latitude_edges, bc_density)
    ax.plot(
        x_step,
        y_step,
        color=case_color(record),
        linestyle=case_line_style(record),
        linewidth=1.5,
        alpha=0.75,
        label=case_label(record),
    )

def plot_altitude_distribution(ax, record):
    distribution = record["distribution"]
    altitude_edges = to_float_array(distribution["altitude_edges_km"])
    bc_density = to_float_array(distribution["bc_mass_density_kg_per_km"]) / 1e6
    x_step, y_step = step_values(altitude_edges, bc_density)
    valid = np.isfinite(x_step) & np.isfinite(y_step)
    ax.plot(
        y_step[valid],
        x_step[valid],
        color=case_color(record),
        linestyle=case_line_style(record),
        linewidth=1.5,
        alpha=0.75,
        label=case_label(record),
    )

def record_key(record):
    return record["launch_lat"], record["season"], record["location"]

def without_bc_only(records):
    return [record for record in records if record["location"] != "CC_BC_only"]

def latitude_area_gg(record):
    distribution = record["distribution"]
    latitude_edges = to_float_array(distribution["latitude_edges_deg"])
    bc_density = to_float_array(distribution["bc_mass_density_kg_per_degree"])
    return float(np.sum(bc_density * np.diff(latitude_edges)) / 1e6)

def altitude_area_gg(record):
    distribution = record["distribution"]
    altitude_edges = to_float_array(distribution["altitude_edges_km"])
    bc_density = to_float_array(distribution["bc_mass_density_kg_per_km"])
    return float(np.sum(bc_density * np.diff(altitude_edges)) / 1e6)

def verify_distribution_areas(lat_records, alt_records, bc_burdens):
    alt_by_key = {record_key(record): record for record in alt_records}
    rows = []

    for lat_record in lat_records:
        key = record_key(lat_record)
        alt_record = alt_by_key.get(key)
        bc_burden = bc_burdens.get(key)
        if alt_record is None or bc_burden is None:
            continue

        lat_area = latitude_area_gg(lat_record)
        alt_area = altitude_area_gg(alt_record)
        rows.append({
            "launch_lat": key[0],
            "season": key[1],
            "location": key[2],
            "bc_burden_gg": bc_burden,
            "latitude_area_gg": lat_area,
            "altitude_area_gg": alt_area,
            "latitude_minus_burden_gg": lat_area - bc_burden,
            "altitude_minus_burden_gg": alt_area - bc_burden,
        })

    if not rows:
        print("BC distribution area check skipped: no matching burden records.")
        return

    max_lat_error = max(abs(row["latitude_minus_burden_gg"]) for row in rows)
    max_alt_error = max(abs(row["altitude_minus_burden_gg"]) for row in rows)
    print(
        "BC distribution area check: "
        f"max |latitude area - burden| = {max_lat_error:.3e} Gg; "
        f"max |altitude area - burden| = {max_alt_error:.3e} Gg."
    )

    check_path = Path("data_file/BC_distribution_area_check.csv")
    with open(check_path, "w") as file:
        file.write(
            "launch_lat,season,location,bc_burden_gg,latitude_area_gg,"
            "altitude_area_gg,latitude_minus_burden_gg,altitude_minus_burden_gg\n"
        )
        for row in rows:
            file.write(
                f"{row['launch_lat']},{row['season']},{row['location']},"
                f"{row['bc_burden_gg']:.12g},{row['latitude_area_gg']:.12g},"
                f"{row['altitude_area_gg']:.12g},{row['latitude_minus_burden_gg']:.12g},"
                f"{row['altitude_minus_burden_gg']:.12g}\n"
            )

def sort_records_north_to_south(records):
    return sorted(records, key=lambda record: (
        -record["launch_lat"],
        record["location"] == "CC_BC_only",
        record["location"],
    ))

def plot_bc_distribution_overview(lat_records, alt_records):
    labels = ["g", "h", "i"]
    output_dir.mkdir(parents=True, exist_ok=True)

    for col, season in enumerate(season_order):
        fig, lat_ax = plt.subplots(figsize=(4.6, 3.8))
        matching_lat = sort_records_north_to_south([record for record in lat_records if record["season"] == season])

        for record in matching_lat:
            plot_latitude_distribution(lat_ax, record)

        lat_ax.set_xlabel("Latitude (°)")
        lat_ax.set_ylabel("BC burden density (Gg degree$^{-1}$)")
        lat_ax.set_xlim(-90, 90)
        lat_ax.axhline(0, color=COLORS["black"], linewidth=0.7, alpha=0.35)
        lat_ax.text(0.02, 1.08, panel_label(labels[col]), transform=lat_ax.transAxes, fontsize=13, fontweight="bold")

        add_legend_if_needed(lat_ax)
        style_distribution_axis(lat_ax)

        fig.tight_layout()
        fig.savefig(output_dir / f"bc_distribution_{season}.pdf")
        plt.close(fig)

def safe_case_filename(record):
    season = record["season"]
    location = record["location"]
    launch_lat = record["launch_lat"]
    return f"bc_distribution_{season}_{location}_{launch_lat}.pdf".replace("-", "m")

def plot_bc_distribution_case_files(lat_records, alt_records):
    distribution_case_dir.mkdir(parents=True, exist_ok=True)
    alt_by_key = {
        (record["launch_lat"], record["season"], record["location"]): record
        for record in alt_records
    }

    for lat_record in lat_records:
        key = (lat_record["launch_lat"], lat_record["season"], lat_record["location"])
        alt_record = alt_by_key.get(key)
        if alt_record is None:
            continue

        fig, axes = plt.subplots(1, 2, figsize=(8, 3.6))
        plot_latitude_distribution(axes[0], lat_record)
        plot_altitude_distribution(axes[1], alt_record)

        axes[0].set_xlabel("Latitude (°)")
        axes[0].set_ylabel("BC burden density (Gg degree$^{-1}$)")
        axes[0].set_xlim(-90, 90)
        axes[0].axhline(0, color=COLORS["black"], linewidth=0.7, alpha=0.35)

        axes[1].set_xlabel("BC burden density (Gg km$^{-1}$)")
        axes[1].set_ylabel("Altitude (km)")
        axes[1].set_ylim(bottom=10)
        axes[1].axvline(0, color=COLORS["black"], linewidth=0.7, alpha=0.35)

        title = f"{case_label(lat_record)} - {season_titles[lat_record['season']]}"
        fig.suptitle(title, fontsize=12)

        for ax in axes:
            style_distribution_axis(ax)

        fig.tight_layout()
        fig.savefig(distribution_case_dir / safe_case_filename(lat_record))
        plt.close(fig)

def plot_bc_distributions():
    lat_path = Path("data_file/BC_latitude_distribution.txt")
    alt_path = Path("data_file/BC_altitude_distribution.txt")
    if not lat_path.exists() or not alt_path.exists():
        print(
            "Skipping BC distribution plots. "
            "Run analysis/burden_distribution_data.py first to generate "
            "BC_latitude_distribution.txt and BC_altitude_distribution.txt."
        )
        return

    lat_records = read_distribution_file(lat_path)
    alt_records = read_distribution_file(alt_path)
    lat_records = without_bc_only(lat_records)
    alt_records = without_bc_only(alt_records)
    bc_burdens = read_scalar_records("data_file/BC_burden.txt")
    verify_distribution_areas(lat_records, alt_records, bc_burdens)
    plot_bc_distribution_overview(lat_records, alt_records)
    plot_bc_distribution_case_files(lat_records, alt_records)

def main():
    bc_loading, bc_loading_summer, bc_loading_winter = read_file('data_file/BC_burden.txt')
    bc_loading_error, bc_loading_summer_error, bc_loading_winter_error = read_file('data_file/BC_burden_error.txt')
    plot(bc_loading, bc_loading_summer, bc_loading_winter, "Launch latitude (°)", "BC burden (Gg)", None, panel_label('a'), "bc_loading_vs_launch_latitude.pdf", legend=True, year_round_error=bc_loading_error, summer_error=bc_loading_summer_error, winter_error=bc_loading_winter_error)

    latitude_BC_mean, latitude_BC_mean_summer, latitude_BC_mean_winter = read_file('data_file/mean_BC_latitude.txt')
    latitude_BC_mean_error, latitude_BC_mean_summer_error, latitude_BC_mean_winter_error = read_file('data_file/mean_BC_latitude_error.txt')
    plot(latitude_BC_mean, latitude_BC_mean_summer, latitude_BC_mean_winter, "Launch latitude (°)", "Weighted mean latitude of BC (°)", None, panel_label('b'), "mean_bc_vs_launch_latitude.pdf", legend=False, year_round_error=latitude_BC_mean_error, summer_error=latitude_BC_mean_summer_error, winter_error=latitude_BC_mean_winter_error)

    # altitude_BC_mean, altitude_BC_mean_summer, altitude_BC_mean_winter = read_file('data_file/mean_BC_altitude.txt')
    # altitude_BC_mean_error, altitude_BC_mean_summer_error, altitude_BC_mean_winter_error = read_file('data_file/mean_BC_altitude_error.txt')
    # plot(altitude_BC_mean, altitude_BC_mean_summer, altitude_BC_mean_winter, "Launch latitude (°)", "Weighted mean altitude of BC (km)", None, panel_label('c'), "mean_altitude_bc_vs_launch_latitude.pdf", legend=False, year_round_error=altitude_BC_mean_error, summer_error=altitude_BC_mean_summer_error, winter_error=altitude_BC_mean_winter_error)

    h2o_loading, h2o_loading_summer, h2o_loading_winter = read_file('data_file/H2O_burden.txt')
    h2o_loading_error, h2o_loading_summer_error, h2o_loading_winter_error = read_file('data_file/H2O_burden_error.txt')
    plot(h2o_loading, h2o_loading_summer, h2o_loading_winter, "Launch latitude (°)", "H2O burden (Tg)", None, panel_label('c'), "H2O_loading_vs_launch_latitude.pdf", legend=False, year_round_error=h2o_loading_error, summer_error=h2o_loading_summer_error, winter_error=h2o_loading_winter_error)

    nox_loading, nox_loading_summer, nox_loading_winter = read_file('data_file/NOX_burden.txt')
    nox_loading_error, nox_loading_summer_error, nox_loading_winter_error = read_file('data_file/NOX_burden_error.txt')
    plot(nox_loading, nox_loading_summer, nox_loading_winter, "Launch latitude (°)", "NOx burden (Gg)", None, panel_label('d'), "NOx_loading_vs_launch_latitude.pdf", legend=False, year_round_error=nox_loading_error, summer_error=nox_loading_summer_error, winter_error=nox_loading_winter_error)

    plot_bc_distributions()

if __name__ == "__main__":
    main()
