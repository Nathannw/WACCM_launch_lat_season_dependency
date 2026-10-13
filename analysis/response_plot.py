import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import PchipInterpolator
from analysis.func import *
from analysis.figure_style import COLORS, panel_label

# To plot responses vs launch latitudes

def plot(data, data_s, data_w, label, alphabet, response, legend,
         data_error=None, data_s_error=None, data_w_error=None):
    # Create figure
    plt.figure(figsize=(6, 4))
    x_range = np.linspace(latitudes.min(), latitudes.max(), 400)

    f = PchipInterpolator(latitudes, data)
    y_pchip = f(x_range)

    # plt.plot(x_range, y_pchip, linestyle = '--', color = 'black', alpha=0.75)
    # Scatter plots (journal style: distinct but not flashy)
    plot_points(latitudes, data, data_error, COLORS["green"], "o", "Year-round")
    plot_points(latitudes, data_s, data_s_error, COLORS["orange"], "s", "Boreal summer-only")
    plot_points(latitudes, data_w, data_w_error, COLORS["blue"], "^", "Boreal winter-only")

    # Axis labels and title
    plt.ylabel("Launch latitude (°)", fontsize=15)
    plt.xlabel(label, fontsize=15)
    # plt.ylim([60, 100])

    # Ticks formatting
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)

    if legend:
        # Legend formatting
        plt.legend(frameon=False, fontsize=15, loc="best")

    # Axis border (journal look: all sides visible, no grid)
    for spine in plt.gca().spines.values():
        spine.set_linewidth(1.0)

    plt.text(0.02, 1.05, f'{alphabet}', transform=plt.gca().transAxes, fontsize=15, fontweight='bold')
    # plt.legend(frameon=True, fontsize=12)
    # Tight layout for publication
    plt.tight_layout()

    # Save high-resolution version (for journal submission)
    plt.savefig(f"fig/response/{response}_vs_launch_latitude.pdf")

    # Show plot
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

labels = ['Stratospheric temperature anomaly (K)', 'Ozone anomaly (DU)', r'All-sky ERF ($\mathrm{W}\ \mathrm{m}^{-2}$)', r'Clear-sky ERF ($\mathrm{W}\ \mathrm{m}^{-2}$)', r'All-sky SW anomaly ($\mathrm{W}\ \mathrm{m}^{-2}$)', r'All-sky LW anomaly ($\mathrm{W}\ \mathrm{m}^{-2}$)', r'Clear-sky SW anomaly ($\mathrm{W}\ \mathrm{m}^{-2}$)', r'Clear-sky LW anomaly ($\mathrm{W}\ \mathrm{m}^{-2}$)']
tag = [panel_label(letter) for letter in ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h']]
# Data points
latitudes = np.array([-55, -29, 0, 29,  55, 70])

def main ():
    for k, response in enumerate(['T', 'O3', 'ERF', 'ERFC', 'FSNT', 'FLNT', 'FSNTC', 'FLNTC']):
        data, data_summer, data_winter = read_file(f'data_file/{response}.txt')
        data_error, data_summer_error, data_winter_error = read_file(f'data_file/{response}_error.txt')
        if k == 0:
            legend = True
        else:
            legend = False
        plot(
            data, data_summer, data_winter, labels[k], tag[k], response, legend,
            data_error=data_error,
            data_s_error=data_summer_error,
            data_w_error=data_winter_error
        )

if __name__ == "__main__":
    main()
