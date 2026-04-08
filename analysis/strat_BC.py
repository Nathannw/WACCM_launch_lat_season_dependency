import netCDF4 as nc
from netCDF4 import Dataset as NetCDFFile
import matplotlib.pyplot as plt
import numpy as np
import os
from scipy.optimize import curve_fit
from analysis.emission_funcs import *
from analysis.func import *

case_list_initial = ['all_year', 'all_year', 'summer', 'winter', 'all_year', 'summer', 'winter', 'all_year', 'summer', 'winter','all_year', 'summer', 'winter','all_year', 'all_year']
location_list = ['CC_BC_only', 'SA', 'SA', 'SA', 'CC', 'CC', 'CC', 'RU', 'RU', 'RU', 'EQ', 'EQ', 'EQ', 'UK', 'AU']
date_list = ['AUG_13','JUL_8','JUL_8','JUL_8','JUL_8','JUL_8','JUL_8','JUL_8','JUL_8','JUL_8','AUG_13','AUG_13','AUG_13','AUG_13','AUG_13']
label = ['', '', '', '', '', '', '', '', '', '', '(a)', '(b)', '(c)', '', '']

def main():
    for i in range(len(case_list_initial)):
        control_list = []
        case_list = []
        launch_season = case_list_initial[i]
        location = location_list[i]
        dir_case = f'results_cloud/lon/{date_list[i]}_30Gg_{launch_season}_{location}/'
        dir_control = 'results_cloud/lon/JUL_8_control/'
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

        data_list = []
        for j, (control, case) in enumerate(zip(control_list, case_list)):
            nc = NetCDFFile(control)
            nc1 = NetCDFFile(case)

            # Assume Tropopause for the first 5 year since the diagnosed tropopause data was not collected.
            if j <= 59:
                data1 = nc.variables['bc_a4'][:,18:53,:] + nc.variables['bc_a1'][:,18:53,:]
                data1 = np.sum(data1*nc.variables['MASS'][:,18:53,:])
                data2 = nc1.variables['bc_a4'][:,18:53,:] + nc1.variables['bc_a1'][:,18:53,:]
                data2 = np.sum(data2*nc1.variables['MASS'][:,18:53,:])
                data = data2-data1
            else:
                aggregated_data_temp = []
                bc_c = nc.variables['bc_a4'][:] + nc.variables['bc_a1'][:]
                mass_c = nc.variables['MASS'][:]
                bc_case = nc1.variables['bc_a4'][:] + nc1.variables['bc_a1'][:]
                mass_case = nc1.variables['MASS'][:]
                
                for a in range(96):
                    idx_lower_c, idx_upper_c, ratio_c = find_trop_func(nc, a)
                    l_c, u_c = idx_lower_c[0], idx_upper_c[0]
                    
                    data1_1 = np.sum(bc_c[:, 18:l_c, a] * mass_c[:, 18:l_c, a])
                    data1_2 = np.sum(bc_c[:, 18:u_c, a] * mass_c[:, 18:u_c, a])
                    agg_1 = data1_1 * (1 - ratio_c[0]) + data1_2 * ratio_c[0]

                    idx_lower_case, idx_upper_case, ratio_case = find_trop_func(nc1, a)
                    l_case, u_case = idx_lower_case[0], idx_upper_case[0]
                    
                    data2_1 = np.sum(bc_case[:, 18:l_case, a] * mass_case[:, 18:l_case, a])
                    data2_2 = np.sum(bc_case[:, 18:u_case, a] * mass_case[:, 18:u_case, a])
                    agg_2 = data2_1 * (1 - ratio_case[0]) + data2_2 * ratio_case[0]
                    
                    aggregated_data_temp.append(agg_2 - agg_1)
                data = np.ma.masked_array(np.sum(aggregated_data_temp))
            data_list.append(data)
        data_con = [sum(data_list[i:i+12])*144 / 12 for i in range(0, len(data_list) - len(data_list) % 12, 12)]
        rho1_case = np.corrcoef(data_con[1:], data_con[:-1])[0, 1]
        Neff, dof_eff = Neff_calculation(rho1_case)
        # Exclude first 5 year to calculate statistical parameters

        if launch_season == 'summer':
            launch_season_title = 'Summer-Only'
        elif launch_season == 'winter':
            launch_season_title = 'Winter-Only'
        elif launch_season == 'all_year':
            launch_season_title = 'Year-Round'

        # Your exponential function
        def exp_func(x, a, b, c):
            return a * np.exp(b * x) + c
        
        def exp_func_year(x, a, b, c):
            return a * np.exp(b * x * 12) + c

        # Prepare data
        x_data = np.arange(1, len(data_list)+1)
        y_data = np.array(data_list) / 1e6 * 144

        # Fit curve
        params, _ = curve_fit(exp_func, x_data, y_data, p0=(1, 0.01, 0))

        # Get fitted values
        fitted_y = exp_func_year(x_data/12, *params)

        x_steady  = np.log(0.1)/params[1]/12

        y_steady = exp_func(x_steady*12, *params)
        x = np.arange(1,len(data_list)+1)/12
        y = np.array(data_list)/1e6*144
        # print(f'BC loading at year 1: {y[11]} Gg')
        fitted_x = x
        if launch_season == 'summer':
            fitted_x = fitted_x
            fitted_y = fitted_y
            x = x
            y = y

        if launch_season == 'summer':
            launch_season_title = 'Summer-Only'
        elif launch_season == 'winter':
            launch_season_title = 'Winter-Only'
        elif launch_season == 'all_year':
            launch_season_title = 'Year-Round'
        plt.figure(figsize=(8, 6))
        steady_state = fitted_y[-1]
        plt.plot(x, y, label='Black Carbon Burden', linewidth=2, linestyle='-', color = 'black')    
        plt.axhline(steady_state,
                    color='red',
                    linestyle='-',
                    linewidth=1.5,
                    label='Quasi-steady state')

        plt.axhline(y_steady,
                    color='blue',
                    linestyle=':',
                    linewidth=1.5,
                    label=f'10% below quasi-steady state (≈{x_steady:.1f} yrs)')

        
        plt.plot(fitted_x, fitted_y,    label=(
            "Exponential fit\n"
            rf"$f(x)={params[0]:.1f}e^{{{params[1]*12:.2f}x}}+{params[2]:.1f}$"
        ), linewidth=2, linestyle='--', color = 'black')
        plt.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.7)
        plt.legend(fontsize=16, loc='best', frameon=False)
        plt.xticks(fontsize=16)
        plt.yticks(fontsize=16)
        plt.xlabel("Simulation Year", fontsize=18)
        plt.ylabel("Black Carbon Burden (Gg)", fontsize=18)
        plt.ylim([0,120])
        plt.xlim([0,20])
        plt.text(0.02, 1.05, label[i], transform=plt.gca().transAxes, fontsize=18, fontweight='bold')
        plt.savefig('fig/BC_burdens/'+f'strat_BC_{launch_season_title}_{location}.pdf')

if __name__ == "__main__":
    main()