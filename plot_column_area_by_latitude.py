from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.io import netcdf_file

from analysis.figure_style import COLORS


# Visualize the latitude weights used by analysis.emission_funcs.col_area().

PROJECT_DIR = Path(__file__).resolve().parent
LAT_FILE = PROJECT_DIR / "geo_file" / "coords_1.9x2.5_L70.nc"
GW_FILE = PROJECT_DIR / "geo_file" / "coords_1.9x2.5_L88_c150828.nc"
FIG_DIR = PROJECT_DIR / "fig" / "grid"
DATA_DIR = PROJECT_DIR / "data_file"

EARTH_RADIUS_CM = 637.8e6


def read_variable(path, name):
    with netcdf_file(path, mode="r", mmap=False) as dataset:
        return np.array(dataset.variables[name].data, dtype=float).copy()


def column_area_by_latitude():
    lat = read_variable(LAT_FILE, "lat")
    lon = read_variable(LAT_FILE, "lon")
    gw = read_variable(GW_FILE, "gw")

    earth_area_cm2 = 4.0 * np.pi * EARTH_RADIUS_CM**2
    column_area_cm2 = gw * earth_area_cm2 / len(lon) / np.sum(gw)
    zonal_band_area_cm2 = column_area_cm2 * len(lon)
    area_fraction = zonal_band_area_cm2 / earth_area_cm2

    return lat, gw, column_area_cm2, zonal_band_area_cm2, area_fraction


def save_csv(lat, gw, column_area_cm2, zonal_band_area_cm2, area_fraction):
    DATA_DIR.mkdir(exist_ok=True)
    output_path = DATA_DIR / "column_area_by_latitude.csv"
    with open(output_path, "w") as file:
        file.write(
            "latitude_deg,gaussian_weight,column_area_cm2,"
            "zonal_band_area_cm2,area_fraction_percent\n"
        )
        for row in zip(lat, gw, column_area_cm2, zonal_band_area_cm2, area_fraction):
            file.write(
                f"{row[0]:.10g},{row[1]:.10g},{row[2]:.10g},"
                f"{row[3]:.10g},{row[4] * 100:.10g}\n"
            )
    return output_path


def plot(lat, column_area_cm2, area_fraction):
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(
        lat,
        column_area_cm2 / 1e16,
        color=COLORS["blue"],
        linewidth=1.5,
        marker="o",
        markersize=3,
        alpha=0.85,
        label="Grid-column area",
    )
    ax.set_xlabel("Latitude (deg)")
    ax.set_ylabel(r"Area per lon cell ($10^{16}$ cm$^2$)")
    ax.set_xlim(-90, 90)
    ax.tick_params(axis="both", labelsize=11)

    ax2 = ax.twinx()
    ax2.plot(
        lat,
        area_fraction * 100,
        color=COLORS["vermillion"],
        linewidth=1.2,
        linestyle="--",
        alpha=0.85,
        label="Zonal band share",
    )
    ax2.set_ylabel("Zonal band share of Earth area (%)")
    ax2.tick_params(axis="y", labelsize=11)

    handles, labels = ax.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(handles + handles2, labels + labels2, frameon=False, loc="upper left")

    for spine in ax.spines.values():
        spine.set_linewidth(1.0)
    for spine in ax2.spines.values():
        spine.set_linewidth(1.0)

    plt.tight_layout()
    pdf_path = FIG_DIR / "column_area_by_latitude.pdf"
    png_path = FIG_DIR / "column_area_by_latitude.png"
    plt.savefig(pdf_path)
    plt.savefig(png_path, dpi=300)
    plt.close(fig)
    return pdf_path, png_path


def main():
    lat, gw, column_area_cm2, zonal_band_area_cm2, area_fraction = column_area_by_latitude()
    csv_path = save_csv(lat, gw, column_area_cm2, zonal_band_area_cm2, area_fraction)
    pdf_path, png_path = plot(lat, column_area_cm2, area_fraction)

    equator_idx = int(np.argmin(np.abs(lat)))
    pole_idx = int(np.argmax(np.abs(lat)))
    ratio = column_area_cm2[equator_idx] / column_area_cm2[pole_idx]

    print(f"Saved: {pdf_path}")
    print(f"Saved: {png_path}")
    print(f"Saved: {csv_path}")
    print(
        "Equator-to-pole column-area ratio: "
        f"{ratio:.2f} ({lat[equator_idx]:.2f} deg vs {lat[pole_idx]:.2f} deg)"
    )


if __name__ == "__main__":
    main()
