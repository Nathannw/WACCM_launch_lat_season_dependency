import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import PchipInterpolator
from analysis.func import *

# To plot emission burdens and BC distribution.

latitudes = np.array([-55, -29, 0, 29, 55, 70])

def plot(year_round, summer, winter, xlabel, ylabel, lim, label, savename, legend):
    # Create figure
    plt.figure(figsize=(6, 4))

    # Scatter plots (journal style: distinct but not flashy)
    plt.scatter(latitudes, year_round, 
                color="forestgreen", marker="o", s=50, label="Year-round", alpha=0.75)
    plt.scatter(latitudes, summer, 
                color="darkorange", marker="s", s=50, label="Boreal Summer-only", alpha=0.75)
    plt.scatter(latitudes, winter, 
                color="navy", marker="^", s=50, label="Boreal Winter-only", alpha=0.75)
    x_range = np.linspace(latitudes.min(), latitudes.max(), 400)

    f = PchipInterpolator(latitudes, year_round)
    y_pchip = f(x_range)

    plt.plot(x_range, y_pchip, linestyle = '--', color = 'black')

    # Axis labels and title
    plt.xlabel(xlabel, fontsize=12)
    plt.ylabel(ylabel, fontsize=12)
    plt.ylim(lim)

    # Ticks formatting
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)

    if legend:
        # Legend formatting
        plt.legend(frameon=True, fontsize=12)

    # Axis border (journal look: all sides visible, no grid)
    for spine in plt.gca().spines.values():
        spine.set_linewidth(1.0)

    plt.text(0.02, 1.05, label, transform=plt.gca().transAxes, fontsize=12, fontweight='bold')

    # Tight layout for publication
    plt.tight_layout()

    # Save high-resolution version (for journal submission)
    plt.savefig(f"fig/burdens_distribution/{savename}")

    # # Show plot
    # plt.show()

def main():
    bc_loading, bc_loading_summer, bc_loading_winter = read_file('data_file/BC_burden.txt')
    plot(bc_loading, bc_loading_summer, bc_loading_winter, "Launch Latitude (°)", "BC Burden (Gg)", [60, 100], '(a)', "bc_loading_vs_launch_latitude.pdf", legend=True)

    latitude_BC_mean, latitude_BC_mean_summer, latitude_BC_mean_winter = read_file('data_file/mean_BC_latitude.txt')
    plot(latitude_BC_mean, latitude_BC_mean_summer, latitude_BC_mean_winter, "Launch Latitude (°)", "Weighted Mean Latitude of BC (°)", None, '(b)', "mean_bc_vs_launch_latitude.pdf", legend=False)

    altitude_BC_mean, altitude_BC_mean_summer, altitude_BC_mean_winter = read_file('data_file/mean_BC_altitude.txt')
    plot(altitude_BC_mean, altitude_BC_mean_summer, altitude_BC_mean_winter, "Launch Latitude (°)", "Weighted Mean Altitude of BC (km)", None, '(c)', "mean_altitude_bc_vs_launch_latitude.pdf", legend=False)

    h2o_loading, h2o_loading_summer, h2o_loading_winter = read_file('data_file/H2O_burden.txt')
    plot(h2o_loading, h2o_loading_summer, h2o_loading_winter, "Launch Latitude (°)", "H2O Burden (Gg)", None, '(d)', "H2O_loading_vs_launch_latitude.pdf", legend=False)

    nox_loading, nox_loading_summer, nox_loading_winter = read_file('data_file/NOX_burden.txt')
    plot(nox_loading, nox_loading_summer, nox_loading_winter, "Launch Latitude (°)", "NOx Burden (Gg)", None, '(e)', "NOx_loading_vs_launch_latitude.pdf", legend=False)

if __name__ == "__main__":
    main()