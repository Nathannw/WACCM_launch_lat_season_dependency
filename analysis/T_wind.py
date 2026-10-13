import netCDF4 as nc
from netCDF4 import Dataset as NetCDFFile
import matplotlib.pyplot as plt
import numpy as np
import os
from scipy.stats import ttest_1samp, ttest_ind
import scipy.stats as stats
from analysis.figure_style import COLORS, panel_label

# To plot temperature and wind vector anomalies

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

    control_list = control_list[120:]
    case_list = case_list[120:]
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
    ax.quiver(lat2d, alt2d, V_sub, Wzm_sub, scale=7.5, width=0.004, alpha=1, color=COLORS["vermillion"])
    ax.set_title(f"{i_month}")
    ax.set_xlim([-90,90])
    
    return c 


case_list_initial = ['all_year', 'all_year', 'summer', 'winter', 'all_year' ,'all_year', 'summer', 'winter', 'all_year', 'summer', 'winter','all_year','all_year', 'summer', 'winter']
location_list = ['CC_BC_only', 'SA', 'SA', 'SA', 'AU', 'EQ', 'EQ', 'EQ', 'CC', 'CC', 'CC', 'UK' , 'RU', 'RU', 'RU']
lat_list = [29,-55,-55,-55,-29,-0,0,0,29,29,29,55,70,70,70]
date_list = ['AUG_13', 'JUL_8','JUL_8','JUL_8','AUG_13','AUG_13','AUG_13','AUG_13','JUL_8','JUL_8','JUL_8','AUG_13','JUL_8','JUL_8','JUL_8']
alp = ['', panel_label('d'), '', '', '', panel_label('c'), '', '', panel_label('b'), '', '', '', panel_label('a'), '', '']

def main():
    for m in range(len(case_list_initial)):
        launch_season = case_list_initial[m]
        location = location_list[m]
        date = date_list[m]
        
        # Create a unique readable identifier for the current case
        case_identifier = f"{date}_30Gg_{launch_season}_{location}"
        dir_case = f'results_cloud/lon/{case_identifier}/'
        dir_control = 'results_cloud/lon/JUL_8_control/'

        fig, axes = plt.subplots(1, 4, figsize=(16, 4))

        month = ['DJF', 'MAM', 'JJA', 'SON']
        i_list = [0, 3, 6, 9]

        for j, i in enumerate(i_list):
        
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
            # vertical_plot_deltnew should plot only the masked values
            # Get reference altitude
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
            X, Y = np.meshgrid(lats, altitudes)  # or how
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
        fig.text(0.025, 0.95, f"{alp[m]}", fontsize=16, fontweight="bold", va="center")
        fig.text(0.5, 0.025, "Latitude (°)", ha="center", fontsize=16)
        fig.text(0.025, 0.5, "Altitude (km)", va="center", rotation="vertical", fontsize=16)


        if launch_season == 'summer':
            launch_season_title = 'Summer-Only'
        elif launch_season == 'winter':
            launch_season_title = 'Winter-Only'
        elif launch_season == 'all_year':
            launch_season_title = 'Year-Round'
            
        plt.tight_layout(rect=[0.05, 0.05, 0.9, 1])  # rect=[left, bottom, right, top]
        plt.savefig(f'fig/T_wind/4panel_{location}_{launch_season_title}.pdf')
        # plt.show()

if __name__ == "__main__":
    main()
