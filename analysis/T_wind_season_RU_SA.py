import netCDF4 as nc
from netCDF4 import Dataset as NetCDFFile
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
from scipy.stats import ttest_1samp, ttest_ind
import scipy.stats as stats

# To plot temperature and wind vector anomalies for 70°N and 55°S seasonal cases

def avg_case_data(dir_case, response, range_min, range_max, i):
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

    control_list = control_list[60:]
    case_list = case_list[60:]
    data_list_vertical = []
    for control, case in zip(control_list, case_list):
        nc = NetCDFFile(control)
        nc1 = NetCDFFile(case)
        # Vertical
        data1 = np.mean(nc1.variables[response][:, range_min:range_max, :], axis=(0)) # -50 + 50 --> 19:-19
        data = np.mean(nc.variables[response][:, range_min:range_max, :], axis=(0)) 
        data = data1 - data
        data_list_vertical.append(data)
    list = []
    for i in range(i, len(data_list_vertical)-2, 12):
        list.append(data_list_vertical[i:i+3]) # Dec to Feb 3
    averaged_3month = np.mean(list, axis = (1))
    t_stat, p_val = ttest_1samp(np.array(averaged_3month), 0)

    averaged = np.mean(averaged_3month, axis = (0))
    data_list_vertical = averaged
    yearly_average = averaged_3month
    return data_list_vertical, t_stat, yearly_average

def vertical_plot_deltnew(ax, nc, data, data_wind, data_V, 
                          range_min, range_max, vmin, vmax, color, i_month):
    altitudes = np.mean(nc.variables['Z3'][:, range_min:range_max, :], axis=(0,2))/1000
    lats = nc.variables['lat'][:]  

    # Wind components
    Wzm = data_wind
    Wzm[Wzm > 1e3] = 0
    V = data_V

    step = 4
    lats_sub = lats[::step]
    altitudes_sub = altitudes  
    Wzm_sub = Wzm[:, ::step] * 50
    V_sub = V[:, ::step]/2

    # Meshgrid for quiver
    lat2d, alt2d = np.meshgrid(lats_sub, altitudes_sub)

    # Plot altitude vs latitude
    c = ax.pcolormesh(lats, altitudes, data, cmap=color, vmin=vmin, vmax=vmax)
    ax.quiver(lat2d, alt2d, V_sub, Wzm_sub, scale=7.5, width=0.004, alpha=1, color='#ff3333')
    ax.set_title(f"{i_month}")
    ax.set_xlim([-90,90])
    
    return c 

alp = '(d)'
dir_control = 'results_cloud/lon/JUL_8_control/'

fig, axes = plt.subplots(1, 4, figsize=(12, 4))

month = ['JJA (70°N Summer-only)', 'JJA (70°N Winter-only)', 'DJF (55°S Summer-only)', 'DJF (55°S Winter-only)']
i_list = [6, 6 ,0, 0]
dir_case_list = ['results_cloud/lon/JUL_8_30Gg_summer_RU/', 'results_cloud/lon/JUL_8_30Gg_winter_RU/', 'results_cloud/lon/JUL_8_30Gg_summer_SA/', 'results_cloud/lon/JUL_8_30Gg_winter_SA/']

def main():
    for j, i in enumerate(i_list):
        dir_case = dir_case_list[j]
        # --- Load data ---
        Wzm, _, _ = avg_case_data(dir_case, 'Wzm', None, None, i)
        Wzm = Wzm[:-1,:,0]

        V, _, _ = avg_case_data(dir_case, 'V', None, None, i)
        T, t_stat_T, T_annual = avg_case_data(dir_case, 'T', None, None, i)


        n = t_stat_T.shape
        nlat = n[1]
        nlev = n[0]
        p_val_T_correct = np.ones((nlev, nlat))

        for k in range(nlat):
            for l in range(nlev):
                x = np.array(T_annual)
                x = x[:,l,k]
                rho1 = np.corrcoef(x[1:], x[:-1])[0, 1]

                # sample sizes
                N = 15

                # rho = 0.5 * (rho_control + rho1_case)
                Neff_case = N * (1 - rho1) / (1 + rho1)
                Neff = np.minimum(Neff_case, N)
                # effective degrees of freedom
                dof_eff = Neff - 1
                p_val_T = 2 * stats.t.sf(abs(t_stat_T[l,k]), dof_eff)
                p_val_T_correct[l,k] = p_val_T

        ax = axes[j]

        Wzm_sig = Wzm
        V_sig = V

        # --- Main plot ---
        nc = NetCDFFile(dir_control+'JUL_8_control.cam.h0.1999-12.nc')
        c = vertical_plot_deltnew(
            ax, nc, T, Wzm_sig, V_sig,
            None, None, -6, 6, 'viridis', month[j]
        )


        # --- Mask non-significant regions ---
        mask = p_val_T_correct > 0.05  # True where NOT significant

        # --- Overlay hatching aligned to coordinates ---
        altitudes = np.mean(nc.variables['Z3'][:, :, :], axis=(0,2))/1000
        lats = nc.variables['lat'][:]  
        X, Y = np.meshgrid(lats, altitudes)
        ax.contourf(
            X, Y, mask,
            levels=[0.5, 1],
            colors=['lightgray'], 
            alpha=0.2   # Adjust opacity for subtle effect
        )


    # --- Shared colorbar ---
    cbar_ax = fig.add_axes([0.9, 0.15, 0.02, 0.7])
    fig.colorbar(c, cax=cbar_ax, label="K")

    # --- Shared labels ---
    fig.text(0.025, 0.95, f"{alp}", fontsize=12, fontweight="bold", va="center")
    fig.text(0.5, 0.025, "Latitude (°)", ha="center", fontsize=14)
    fig.text(0.025, 0.5, "Altitude (km)", va="center", rotation="vertical", fontsize=14)



    plt.tight_layout(rect=[0.05, 0.05, 0.9, 1])  # rect=[left, bottom, right, top]
    plt.savefig(f'fig/T_wind/4panel_T_RU_SA.pdf')
    # plt.show()

if __name__ == "__main__":
    main()





