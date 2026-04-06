from analysis.emission_funcs import *
from analysis.func import *
import netCDF4 as nc
from netCDF4 import Dataset as NetCDFFile
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.basemap import Basemap
import os
import seaborn as sns
from scipy.stats import ttest_1samp, ttest_ind
from datetime import datetime, timedelta
from scipy.stats import f_oneway
from mpl_toolkits.axes_grid1 import make_axes_locatable
import scipy.stats as stats

case_list_initial = ['all_year', 'all_year', 'summer', 'winter', 'all_year' ,'all_year', 'summer', 'winter', 'all_year', 'summer', 'winter','all_year','all_year', 'summer', 'winter']
location_list = ['CC_BC_only', 'SA', 'SA', 'SA', 'AU', 'EQ', 'EQ', 'EQ', 'CC', 'CC', 'CC', 'UK' , 'RU', 'RU', 'RU']
lat_list = [29,-55,-55,-55,-29,-0,0,0,29,29,29,55,70,70,70]
date_list = ['AUG_13', 'JUL_8','JUL_8','JUL_8','AUG_13','AUG_13','AUG_13','AUG_13','JUL_8','JUL_8','JUL_8','AUG_13','JUL_8','JUL_8','JUL_8']

def avg_case_data(dir, response):
    case_list = []
    for root, dirs, files in os.walk(dir):
        for file in files:
            if file.endswith('.nc'):
                case_list.append(os.path.join(root, file))
    case_list.sort(key=lambda x: os.path.basename(x).lower())
    # Exclude first 5 years
    case_list = case_list[60:]
    data_list = []
    for case in case_list:
        nc = NetCDFFile(case)
        if response in ['FLNT', 'FSNT', 'FLNTC', 'FSNTC', 'FLN200', 'FLN200C', 'FSN200', 'FSN200C']:
            aggregated_data = np.mean(nc.variables[response])
        elif response == 'O3':
            mol_air = nc.variables['MASS'][0, :, :]*1000/28.97  # (lev, lat)
            O3 = nc.variables['O3'][0, :, :]         # (lev, lat)
            mol_O3 = O3*mol_air
            area, x, y = col_area()
            zonal_area = np.mean(area, axis=1) # cm^2
            mol_O3percm2 = np.sum(mol_O3/zonal_area,axis = 0)
            DU = mol_O3percm2/4.4615e-8
            aggregated_data = np.mean(DU)
        elif response in ['CLOUD','CLDICE', 'CLDLIQ', 'T']:
            aggregated_data_temp = []
            for a in range(96):
                idx_lower, idx_upper, ratio = find_trop_func(nc, a)
                X1 = np.mean(nc.variables[response][:, 25:idx_lower[0],a])
                X2 = np.mean(nc.variables[response][:, 25:idx_upper[0],a]) 
                aggregated_data_1 = X1 * (1 - ratio[0]) + X2 * ratio[0]
                aggregated_data_temp.append(aggregated_data_1)
            masked = np.mean(np.ma.masked_array(aggregated_data_temp))
            aggregated_data = masked
        data_list.append(aggregated_data)
    averaged = [sum(data_list[i:i+12]) / 12 for i in range(0, len(data_list) - len(data_list) % 12, 12)]
    return averaged, data_list

def processing(dir_control, dir_case, response):
    data_list_control, _ = avg_case_data(dir_control, response)

    dir = dir_case
    data_list, _  = avg_case_data(dir, response)

    # print(f"Diagnose autocorrelation Lag-1: {response}")
    x = np.array(data_list) - np.array(data_list_control)  # 1D array
    rho1 = np.corrcoef(x[1:], x[:-1])[0, 1]
    Neff, dof_eff = Neff_calculation(rho1)

## -------------- Delta plot -----------------
    # Prepare differences
    diff = np.array(data_list) - np.array(data_list_control)
    # Calculate statistics
    mean = np.mean(diff)
    sem = stats.sem(diff)  # Standard error of the mean
    confidence = 0.95
    # Get t critical value for 95% CI and n-1 degrees of freedom
    t_crit = stats.t.ppf((1 + confidence) / 2, df=dof_eff)
    t_stat = np.mean(diff) / stats.sem(diff)
    p_value = 2 * stats.t.sf(abs(t_stat), dof_eff)
    print(f"{response}: {mean:.2f} ± {t_crit * sem:.2f}\n")
    return mean

def main():
    for k, response in enumerate(['T', 'O3', 'FLNT', 'FSNT', 'FLNTC', 'FSNTC']):
        results = []
        for i in range(len(case_list_initial)):
            control_list = []
            case_list = []
            launch_season = case_list_initial[i]
            location = location_list[i]
            date = date_list[i]
            
            # Create a unique readable identifier for the current case
            case_identifier = f"{date}_30Gg_{launch_season}_{location}"
            dir_case = f'results_cloud/lon/{case_identifier}/'
            if launch_season == 'all_year' and location != 'CC_BC_only':
                dir_control = 'results_cloud/lon/JUL_8_control/'
            elif launch_season != 'all_year':
                dir_control = f'results_cloud/lon/{date}_30Gg_all_year_{location}/'
            elif location == 'CC_BC_only':
                dir_control = f'results_cloud/lon/JUL_8_30Gg_all_year_CC/'
            
            for root, _, files in os.walk(dir_control):
                for file in files:
                    if file.endswith('.nc'):
                        control_list.append(os.path.join(root, file))
            control_list.sort(key=lambda x: os.path.basename(x).lower())

            for root, _, files in os.walk(dir_case):
                for file in files:
                    if file.endswith('.nc'):
                        case_list.append(os.path.join(root, file))
            case_list.sort(key=lambda x: os.path.basename(x).lower())
            # Exclude first 5 year
            control_list = control_list[60:]
            case_list = case_list[60:]
        
            print(f"Processing case: {case_identifier}...")

            if case_list and control_list:
                # Store the returned string output into the dictionaries
                results.append(processing(dir_control, dir_case, response))
            else:
                print(f"Warning: Missing files for {case_identifier}. Skipping...")

        # Helper function to write a dictionary out to a file
        def write_to_file(filename, datas):
            with open(filename, 'w') as f:
                f.write(f"{lat_list}\n")
                f.write(f"{case_list_initial}\n")
                f.write(f"{location_list}\n")
                f.write(f"{datas}")
            print(f"Saved: {filename}")

        # Write all the accumulated data out to respective text files
        print("\nWriting output to .txt files...")
        write_to_file(f'{response}_table.txt', results)
        print("Process Complete!")

if __name__ == "__main__":
    main()