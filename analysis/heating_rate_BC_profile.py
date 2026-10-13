import netCDF4 as nc
from netCDF4 import Dataset as NetCDFFile
import matplotlib.pyplot as plt
import numpy as np
import os
from scipy.stats import ttest_1samp, ttest_ind
import scipy.stats as stats
from analysis.figure_style import panel_label

# To produce Figure 4 d, e, f, g, and h

def vertical_plot_delta_no_sig(nc, data, label, name, 
                        range_min, range_max, vmin, vmax, unit, cmap="cividis"):

    # Extract altitude (km) and latitude
    altitudes = np.mean(nc.variables['Z3'][:, range_min:range_max, :], axis=(0, 2)) / 1000
    lev = nc.variables['lev'][range_min:range_max] 
    lats = nc.variables['lat'][:]

    # Create figure
    fig, ax = plt.subplots(figsize=(6, 4))

    # Filled contour/mesh plot
    pcm = ax.pcolormesh(lats, altitudes, data, cmap=cmap, vmin=vmin, vmax=vmax)

    cbar = plt.colorbar(
        pcm,
        ax=ax,
        pad=0.2,   # increase spacing
        aspect=30
    )
    cbar.set_label(unit, fontsize=15)
    cbar.ax.tick_params(labelsize=15)


    # Axis labels
    ax.set_xlabel("Latitude (°)", fontsize=15)
    ax.set_ylabel("Altitude (km)", fontsize=15)
    ax.set_xlim([-90, 90])

    # Tick formatting
    ax.tick_params(axis="both", which="major", labelsize=15)


    X, Y = np.meshgrid(lats, altitudes)

    # --- Add right y-axis showing lev ---
    idx = np.linspace(0, len(lev) - 1, 7, dtype=int)
    ax2 = ax.twinx()
    # Match limits
    ax2.set_ylim(ax.get_ylim())
    ax2.set_yticks(altitudes[idx])
    ax2.set_yticklabels(np.round(lev[idx], 1))

    ax2.set_ylabel("Pressure (hPa)", fontsize=14)

    # Border thickness
    for spine in ax.spines.values():
        spine.set_linewidth(1)

    # Layout and save (vector format, high res)
    plt.text(0.02, 1.05, label, transform=plt.gca().transAxes, fontsize=15, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f"fig/BC/BC_{name}.pdf", bbox_inches="tight")
    # plt.show()

def vertical_plot_delta_sig(nc, data, label, name, 
                        range_min, range_max, vmin, vmax, unit,p_val, cmap="cividis"):

    # Extract altitude (km) and latitude
    altitudes = np.mean(nc.variables['Z3'][:, range_min:range_max, :], axis=(0, 2)) / 1000
    lev = nc.variables['lev'][range_min:range_max] 
    lats = nc.variables['lat'][:]

    # Create figure
    fig, ax = plt.subplots(figsize=(6, 4))

    # Filled contour/mesh plot
    pcm = ax.pcolormesh(lats, altitudes, data, cmap=cmap, vmin=vmin, vmax=vmax)

    cbar = plt.colorbar(
        pcm,
        ax=ax,
        pad=0.2,   # increase spacing
        aspect=30
    )
    cbar.set_label(unit, fontsize=15)
    cbar.ax.tick_params(labelsize=15)


    # Axis labels
    ax.set_xlabel("Latitude (°)", fontsize=15)
    ax.set_ylabel("Altitude (km)", fontsize=15)
    ax.set_xlim([-90, 90])

    # Tick formatting
    ax.tick_params(axis="both", which="major", labelsize=15)
    # --- Mask non-significant regions ---
    mask = p_val > 0.05  # True where NOT significant

    X, Y = np.meshgrid(lats, altitudes)

    ax.contourf(
        X, Y, mask,
        levels=[0.5, 1],
        colors=['lightgray'],
        alpha=0.2
    )

    # --- Add right y-axis showing lev ---
    idx = np.linspace(0, len(lev) - 1, 7, dtype=int)
    ax2 = ax.twinx()
    # Match limits
    ax2.set_ylim(ax.get_ylim())
    ax2.set_yticks(altitudes[idx])
    ax2.set_yticklabels(np.round(lev[idx], 1))

    ax2.set_ylabel("Pressure (hPa)", fontsize=15)

    # ax.set_title(f"{season}", fontsize=12, pad=10)

    # Border thickness
    for spine in ax.spines.values():
        spine.set_linewidth(1)

    # Layout and save (vector format, high res)
    plt.text(0.02, 1.05, label, transform=plt.gca().transAxes, fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f"fig/QRS/QRS_{name}.pdf", bbox_inches="tight")
    # plt.show()

def avg_case_data(dir_case, response, range_min, range_max):
    control_list = []
    case_list = []

    dir_control = 'results_cloud/lon/JUL_8_control/'

    # For control files
    for root, dirs, files in os.walk(dir_control):
        for file in files:
            if file.endswith('.nc'):
                control_list.append(os.path.join(root, file))
    control_list.sort(key=lambda x: os.path.basename(x).lower())

    # For case files
    for root, dirs, files in os.walk(dir_case):
        for file in files:
            if file.endswith('.nc'):
                case_list.append(os.path.join(root, file))
    case_list.sort(key=lambda x: os.path.basename(x).lower())

    control_list = control_list[120:]
    case_list = case_list[120:]
    data_list_vertical = []
    for control, case in zip(control_list, case_list):
        nc = NetCDFFile(control)
        nc1 = NetCDFFile(case)
        # Vertical
        data1 = np.mean(nc1.variables[response][:, range_min:range_max, :], axis=(0)) 
        data = np.mean(nc.variables[response][:, range_min:range_max, :], axis=(0)) 
        data = data1 - data
        data_list_vertical.append(data)
    averaged_annual = [sum(data_list_vertical[i:i+12]) / 12 for i in range(0, len(data_list_vertical) - len(data_list_vertical) % 12, 12)]
    averaged = np.mean(averaged_annual, axis = (0))
    data_list_vertical = averaged
    return data_list_vertical, averaged_annual

dir_control = 'results_cloud/lon/JUL_8_control/'
dir_case_list_QRS = ['results_cloud/lon/JUL_8_30Gg_summer_RU/', 'results_cloud/lon/JUL_8_30Gg_winter_RU/']
alp_QRS = [panel_label('g'), panel_label('h')]
dir_case_list_BC = ['results_cloud/lon/AUG_13_30Gg_all_year_AU/', 'results_cloud/lon/JUL_8_30Gg_all_year_RU/', 'results_cloud/lon/AUG_13_30Gg_summer_EQ/']
alp_BC = [panel_label('d'), panel_label('e'), panel_label('f')]
nc = NetCDFFile('results_cloud/lon/JUL_8_control/JUL_8_control.cam.h0.1999-12.nc')

def main():
    for i, case in enumerate(dir_case_list_QRS):
        QRS, QRS_annual = avg_case_data(case, 'QRS', 18, None)
        t_stat, _ = ttest_1samp(np.array(QRS_annual), 0)
        QRS_annual= np.array(QRS_annual)
        n = QRS_annual.shape
        nlat = n[2]
        nlev = n[1]
        p_val_correct = np.ones((nlev, nlat))

        for k in range(nlat):
            for l in range(nlev):
                x = np.array(QRS_annual)
                x = x[:,l,k]
                rho1 = np.corrcoef(x[1:], x[:-1])[0, 1]

                # sample sizes
                N = 15
                Neff_case = N * (1 - rho1) / (1 + rho1)
                Neff = np.minimum(Neff_case, N)
                # effective degrees of freedom
                dof_eff = Neff - 1
                p_val = 2 * stats.t.sf(abs(t_stat[l,k]), dof_eff)
                p_val_correct[l,k] = p_val

        vertical_plot_delta_sig(nc, QRS, alp_QRS[i],i, 18, None, -1e-5, 2.5e-5, r'$\mathrm{K}\ \mathrm{s}^{-1}$',p_val_correct, 'viridis')
    for i, case in enumerate(dir_case_list_BC):
        bc_a1, _ = avg_case_data(case, 'bc_a1', 18, None)
        bc_a4, _ = avg_case_data(case, 'bc_a4', 18, None)
        bc = bc_a1 + bc_a4
        vertical_plot_delta_no_sig(nc, bc, alp_BC[i],i, 18, None, None, 1e-9, r'$\mathrm{kg}_{\mathrm{BC}}\ \mathrm{kg}_{\mathrm{air}}^{-1}$', 'viridis')

if __name__ == "__main__":
    main()
