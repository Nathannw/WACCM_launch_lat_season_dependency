import matplotlib.pyplot as plt
from scipy.stats import pearsonr
import numpy as np
from analysis.func import *

# To plot correlation

def plotting(x, x_summer, x_winter,y, y_summer, y_winter, x_label, y_label, char, fig_name, legend):
    x_all = np.hstack((x, x_summer, x_winter))
    y_all = np.hstack((y, y_summer, y_winter))
    r, p = pearsonr(x_all[~np.isnan(x_all)], y_all[~np.isnan(y_all)])
    # Create figure
    plt.figure(figsize=(6, 4))

    # Scatter plots (journal style: distinct but not flashy)
    plt.scatter(x, y,
                color="forestgreen", marker="o", s=50, label="Year-round", alpha=0.75)
    plt.scatter(x_summer, y_summer,
                color="darkorange", marker="s", s=50, label="Boreal Summer-only", alpha=0.75)
    plt.scatter(x_winter, y_winter,
                color="navy", marker="^", s=50, label="Boreal Winter-only", alpha=0.75)
    
    plt.text(
        0.95, 0.95,
        f"r = {r:.2f}\np = {p:.2e}",
        transform=plt.gca().transAxes,
        ha="right",
        va="top",
        bbox=dict(facecolor="white", alpha=0.8, edgecolor="none")
    )


    # Axis labels and title
    plt.ylabel(y_label, fontsize=12)
    plt.xlabel(x_label, fontsize=12)
    # plt.ylim([60, 100])

    # Ticks formatting
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)

    if legend:
        # Legend formatting
        plt.legend(frameon=True, fontsize=12)

    # Axis border (journal look: all sides visible, no grid)
    for spine in plt.gca().spines.values():
        spine.set_linewidth(1.0)

    plt.text(0.02, 1.05, char, transform=plt.gca().transAxes, fontsize=12, fontweight='bold')

    # Tight layout for publication
    plt.tight_layout()

    # Save high-resolution version (for journal submission)
    plt.savefig(f"fig/correlation/{fig_name}.pdf")

    # Show plot
    # plt.show()

def main():

    datax, datax_summer, datax_winter = read_file('data_file/BC_burden.txt')
    datay, datay_summer, datay_winter = read_file('data_file/T.txt')
    x_label = "BC burden (Gg)"
    y_label = "Stratospheric Temperature Anomaly (K)"
    char = '(a)'
    fig_name = 'bc_burden_vs_T'
    plotting(datax, datax_summer, datax_winter, datay, datay_summer, datay_winter, x_label, y_label, char, fig_name, True)

    datax, datax_summer, datax_winter = read_file('data_file/mean_BC_altitude.txt')
    datay, datay_summer, datay_winter = read_file('data_file/T.txt')
    x_label = "Weighted Mean Altitude of BC (km)"
    y_label = "Stratospheric Temperature Anomaly (K)"
    char = '(c)'
    fig_name = 'bc_altitude_vs_T'
    plotting(datax, datax_summer, datax_winter, datay, datay_summer, datay_winter, x_label, y_label, char, fig_name, False)

    datax, datax_summer, datax_winter = read_file('data_file/mean_BC_latitude.txt')
    datay, datay_summer, datay_winter = read_file('data_file/T.txt')
    x_label = "Weighted Mean Latitude of BC (°)"
    y_label = "Stratospheric Temperature Anomaly (K)"
    char = '(b)'
    fig_name = 'bc_latitude_vs_T'
    plotting(datax, datax_summer, datax_winter, datay, datay_summer, datay_winter, x_label, y_label, char, fig_name, False)

    datax, datax_summer, datax_winter = read_file('data_file/T_NH.txt')
    datay, datay_summer, datay_winter = read_file('data_file/O3_NH.txt')
    x_label = "NH Stratospheric Temperature Anomaly (K)"
    y_label = 'NH Ozone Anomaly (DU)'
    char = '(d)'
    fig_name = 'T_NH_vs_O3_NH'
    plotting(datax, datax_summer, datax_winter, datay, datay_summer, datay_winter, x_label, y_label, char, fig_name, False)

    datax, datax_summer, datax_winter = read_file('data_file/T_SH.txt')
    datay, datay_summer, datay_winter = read_file('data_file/O3_SH.txt')
    x_label = "0°N - 60°S Stratospheric Temperature Anomaly (K)"
    y_label = '0°N - 60°S Ozone Anomaly (DU)'
    char = '(e)'
    fig_name = 'T_MSH_vs_O3_MSH'
    plotting(datax, datax_summer, datax_winter, datay, datay_summer, datay_winter, x_label, y_label, char, fig_name, False)

    datax, datax_summer, datax_winter = read_file('data_file/CLOUD_SP.txt')
    datay, datay_summer, datay_winter = read_file('data_file/O3_SP.txt')
    x_label = "60°S - 90°S Stratospheric Cloud Fraction Anomaly"
    y_label = '60°S - 90°S Ozone Anomaly (DU)'
    char = '(f)'
    fig_name = 'CLD_SP_vs_O3_SP'
    plotting(datax, datax_summer, datax_winter, datay, datay_summer, datay_winter, x_label, y_label, char, fig_name, False)

if __name__ == "__main__":
    main()

