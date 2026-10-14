import matplotlib.pyplot as plt
from pathlib import Path
from scipy.stats import pearsonr
import numpy as np
from analysis.func import *
from analysis.figure_style import COLORS, panel_label

# To plot correlation

def plot_points(x, y, xerr, yerr, color, marker, label):
    if xerr is None and yerr is None:
        plt.scatter(x, y, color=color, marker=marker, s=50, label=label, alpha=0.75)
    else:
        plt.errorbar(
            x, y, xerr=xerr, yerr=yerr,
            color=color, marker=marker, markersize=7,
            linestyle="none", capsize=3, elinewidth=1,
            label=label, alpha=0.75
        )

def plotting(
    x, x_summer, x_winter, y, y_summer, y_winter,
    x_label, y_label, char, fig_name, legend,
    x_error=None, x_summer_error=None, x_winter_error=None,
    y_error=None, y_summer_error=None, y_winter_error=None,
    xlim=None
):
    x_all = np.hstack((x, x_summer, x_winter))
    y_all = np.hstack((y, y_summer, y_winter))
    valid = ~np.isnan(x_all) & ~np.isnan(y_all)
    r, p = pearsonr(x_all[valid], y_all[valid])
    # Create figure
    plt.figure(figsize=(6, 4))

    # Scatter plots (journal style: distinct but not flashy)
    plot_points(x, y, x_error, y_error, COLORS["green"], "o", "Year-round")
    plot_points(x_summer, y_summer, x_summer_error, y_summer_error, COLORS["orange"], "s", "Boreal summer-only")
    plot_points(x_winter, y_winter, x_winter_error, y_winter_error, COLORS["blue"], "^", "Boreal winter-only")
    
    plt.text(
        0.95, 0.95,
        f"r = {r:.2f}\np = {p:.2e}",
        transform=plt.gca().transAxes,
        ha="right",
        va="top",
        bbox=dict(facecolor="white", alpha=0.5, edgecolor="none")
    )


    # Axis labels and title
    plt.ylabel(y_label, fontsize=13)
    plt.xlabel(x_label, fontsize=13)
    if xlim is not None:
        plt.xlim(xlim)

    # Ticks formatting
    plt.xticks(fontsize=13)
    plt.yticks(fontsize=13)

    if legend:
        # Legend formatting
        plt.legend(frameon=False, fontsize=13)

    # Axis border (journal look: all sides visible, no grid)
    for spine in plt.gca().spines.values():
        spine.set_linewidth(1.0)

    plt.text(0.02, 1.05, char, transform=plt.gca().transAxes, fontsize=13, fontweight='bold')

    # Tight layout for publication
    plt.tight_layout()

    # Save high-resolution version (for journal submission)
    output_path = Path("fig") / "correlation" / f"{fig_name}.pdf"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path)

    # Show plot
    # plt.show()

def read_with_error(data_path, error_path):
    data = read_file(data_path)
    error = read_file(error_path)
    return data, error

def plot_from_files(
    x_path, x_error_path, y_path, y_error_path,
    x_label, y_label, char, fig_name, legend,
    absolute_y=False, xlim=None
):
    (datax, datax_summer, datax_winter), x_errors = read_with_error(x_path, x_error_path)
    (datay, datay_summer, datay_winter), y_errors = read_with_error(y_path, y_error_path)
    datax_error, datax_summer_error, datax_winter_error = x_errors
    datay_error, datay_summer_error, datay_winter_error = y_errors

    if absolute_y:
        datay = np.abs(datay)
        datay_summer = np.abs(datay_summer)
        datay_winter = np.abs(datay_winter)

    plotting(
        datax, datax_summer, datax_winter,
        datay, datay_summer, datay_winter,
        x_label, y_label, char, fig_name, legend,
        x_error=datax_error,
        x_summer_error=datax_summer_error,
        x_winter_error=datax_winter_error,
        y_error=datay_error,
        y_summer_error=datay_summer_error,
        y_winter_error=datay_winter_error,
        xlim=xlim
    )

def main():
    plot_from_files(
        'data_file/T.txt', 'data_file/T_error.txt',
        'data_file/BC_burden.txt', 'data_file/BC_burden_error.txt',
        "Stratospheric temperature anomaly (K)", "BC burden (Gg)", 
        panel_label('a'), 'bc_burden_vs_T', True
    )

    # plot_from_files(
    #     'data_file/T.txt', 'data_file/T_error.txt',
    #     'data_file/mean_BC_altitude.txt', 'data_file/mean_BC_altitude_error.txt',
    #     "Stratospheric temperature anomaly (K)", "Weighted mean altitude of BC (km)", 
    #     panel_label('c'), 'bc_altitude_vs_T', False
    # )

    plot_from_files(
        'data_file/T.txt', 'data_file/T_error.txt',
        'data_file/mean_BC_latitude.txt', 'data_file/mean_BC_latitude_error.txt',
        "Stratospheric temperature anomaly (K)", "Absolute weighted mean latitude of BC (°)", 
        panel_label('b'), 'bc_latitude_vs_T', False,
        absolute_y=True#, xlim=[0, 90]
    )
    # plot_from_files(
    #     'data_file/BC_burden.txt', 'data_file/BC_burden_error.txt',
    #     'data_file/T.txt', 'data_file/T_error.txt',
    #     "BC burden (Gg)", "Stratospheric temperature anomaly (K)",
    #     panel_label('a'), 'bc_burden_vs_T', True
    # )

    # plot_from_files(
    #     'data_file/mean_BC_altitude.txt', 'data_file/mean_BC_altitude_error.txt',
    #     'data_file/T.txt', 'data_file/T_error.txt',
    #     "Weighted mean altitude of BC (km)", "Stratospheric temperature anomaly (K)",
    #     panel_label('c'), 'bc_altitude_vs_T', False
    # )

    # plot_from_files(
    #     'data_file/mean_BC_latitude.txt', 'data_file/mean_BC_latitude_error.txt',
    #     'data_file/T.txt', 'data_file/T_error.txt',
    #     "Absolute weighted mean latitude of BC (°)", "Stratospheric temperature anomaly (K)",
    #     panel_label('b'), 'bc_latitude_vs_T', False,
    #     absolute_x=True#, xlim=[0, 90]
    # )

    plot_from_files(
        'data_file/T_NH.txt', 'data_file/T_NH_error.txt',
        'data_file/O3_NH.txt', 'data_file/O3_NH_error.txt',
        "NH stratospheric temperature anomaly (K)", 'NH ozone anomaly (DU)',
        panel_label('c'), 'T_NH_vs_O3_NH', False
    )

    plot_from_files(
        'data_file/T_SH.txt', 'data_file/T_SH_error.txt',
        'data_file/O3_SH.txt', 'data_file/O3_SH_error.txt',
        "0° N to 60° S stratospheric temperature anomaly (K)",
        '0° N to 60° S ozone anomaly (DU)',
        panel_label('d'), 'T_MSH_vs_O3_MSH', False
    )

    plot_from_files(
        'data_file/CLOUD_SP.txt', 'data_file/CLOUD_SP_error.txt',
        'data_file/O3_SP.txt', 'data_file/O3_SP_error.txt',
        "60° S to 90° S stratospheric cloud fraction anomaly",
        '60° S to 90° S ozone anomaly (DU)',
        panel_label('e'), 'CLD_SP_vs_O3_SP', False
    )

if __name__ == "__main__":
    main()
