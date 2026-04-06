import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import PchipInterpolator
from analysis.func import *

# To plot responses vs launch latitudes

def plot(data, data_s, data_w, label, alphabet, response, legend):
    # Create figure
    plt.figure(figsize=(6, 4))
    x_range = np.linspace(latitudes.min(), latitudes.max(), 400)

    f = PchipInterpolator(latitudes, data)
    y_pchip = f(x_range)

    plt.plot(x_range, y_pchip, linestyle = '--', color = 'black', alpha=0.75)
    # Scatter plots (journal style: distinct but not flashy)
    plt.scatter(latitudes, data, 
                color="forestgreen", marker="o", s=50, label="Year-round", alpha=0.75)
    plt.scatter(latitudes, data_s, 
                color="darkorange", marker="s", s=50, label="Boreal Summer-only", alpha=0.75)
    plt.scatter(latitudes, data_w, 
                color="navy", marker="^", s=50, label="Boreal Winter-only", alpha=0.75)

    # Axis labels and title
    plt.xlabel("Launch Latitude (°)", fontsize=12)
    plt.ylabel(label, fontsize=12)
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

    plt.text(0.02, 1.05, f'{alphabet}', transform=plt.gca().transAxes, fontsize=12, fontweight='bold')
    # plt.legend(frameon=True, fontsize=12)
    # Tight layout for publication
    plt.tight_layout()

    # Save high-resolution version (for journal submission)
    plt.savefig(f"fig/response/{response}_vs_launch_latitude.pdf")

    # Show plot
    # plt.show()

labels = ['Stratospheric Temperature Anomaly (K)', 'Ozone Anomaly (DU)', 'All-sky ERF (W/m2)', 'Clear-sky ERF (W/m2)', 'All-sky SW Anomaly (W/m2)', 'All-sky LW Anomaly (W/m2)', 'Clear-sky SW Anomaly (W/m2)', 'Clear-sky LW Anomaly (W/m2)']
tag = ['(a)', '(b)', '(c)', '(d)', '(e)', '(f)', '(g)', '(h)']
# Data points
latitudes = np.array([-55, -29, 0, 29,  55, 70])

def main ():
    for k, response in enumerate(['T', 'O3', 'ERF', 'ERFC', 'FSNT', 'FLNT', 'FSNTC', 'FLNTC']):
        data, data_summer, data_winter = read_file(f'data_file/{response}.txt')
        if k == 0:
            legend = True
        else:
            legend = False
        plot(data, data_summer, data_winter, labels[k], tag[k], response, legend)

if __name__ == "__main__":
    main()