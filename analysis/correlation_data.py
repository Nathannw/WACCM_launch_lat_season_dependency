from analysis.emission_funcs import *
from analysis.func import *
import netCDF4 as nc
from netCDF4 import Dataset as NetCDFFile
import matplotlib.pyplot as plt
import numpy as np
import os
import scipy.stats as stats

# To generate data for correlation analysis

def area_weighted_mean(values, weights):
    values = np.ma.masked_invalid(np.ma.array(values))
    weights = np.ma.array(weights, mask=np.ma.getmaskarray(values))
    return np.ma.sum(values * weights) / np.ma.sum(weights)

def latitude_area_weights():
    area, _, _ = col_area()
    return np.mean(area, axis=1)

def regional_mean(data, response, lat_bound):
    weights = latitude_area_weights()
    return float(area_weighted_mean(data[lat_bound], weights[lat_bound]))

def annual_average(data_list):
    return [
        np.mean(data_list[i:i+12], axis=0)
        for i in range(0, len(data_list) - len(data_list) % 12, 12)
    ]

def effective_dof(data, min_dof=2.0):
    data = np.asarray(data, dtype=float)
    data = data[np.isfinite(data)]
    n = len(data)
    if n <= 1:
        return None
    if n <= min_dof + 1 or np.isclose(np.std(data), 0):
        return n - 1

    rho1_case = np.corrcoef(data[1:], data[:-1])[0, 1]
    if not np.isfinite(rho1_case) or rho1_case <= 0:
        return n - 1

    neff = n * (1 - rho1_case) / (1 + rho1_case)
    dof_eff = neff - 1
    return min(max(dof_eff, min_dof), n - 1)

def confidence_error(data, dof):
    data = np.asarray(data, dtype=float)
    data = data[np.isfinite(data)]
    if dof is None or len(data) <= 1:
        return 0.0

    sem = stats.sem(data)
    if not np.isfinite(sem):
        return 0.0
    t_crit = stats.t.ppf(0.975, df=dof)
    return float(t_crit * sem)

def avg_case_data(dir, response):
    case_list = []
    for root, dirs, files in os.walk(dir):
        for file in files:
            if file.endswith('.nc'):
                case_list.append(os.path.join(root, file))
    case_list.sort(key=lambda x: os.path.basename(x).lower())
    # Exclude first 10 years
    case_list = case_list[120:]
    data_list = []
    for case in case_list:
        nc = NetCDFFile(case)
        if response == 'O3':
            mol_air = nc.variables['MASS'][0, :, :]*1000/28.97  # (lev, lat)
            O3 = nc.variables['O3'][0, :, :]         # (lev, lat)
            mol_O3 = O3*mol_air
            area, x, y = col_area()
            zonal_area = np.mean(area, axis=1) # cm^2
            mol_O3percm2 = np.sum(mol_O3/zonal_area,axis = 0)
            DU = mol_O3percm2/4.4615e-8
            aggregated_data = DU
        elif response in ['CLOUD','CLDICE', 'CLDLIQ', 'T']:
            aggregated_data_temp = []
            response_data = nc.variables[response][:]
            pressure_weights = pressure_level_weights(nc)
            for a in range(96):
                idx_lower, idx_upper, ratio = find_trop_func(nc, a)
                X1 = pressure_weighted_mean(
                    response_data[:, 25:idx_lower[0], a],
                    pressure_weights[25:idx_lower[0]]
                )
                X2 = pressure_weighted_mean(
                    response_data[:, 25:idx_upper[0], a],
                    pressure_weights[25:idx_upper[0]]
                )
                aggregated_data_1 = X1 * (1 - ratio[0]) + X2 * ratio[0]
                aggregated_data_temp.append(aggregated_data_1)
            masked = np.ma.masked_array(aggregated_data_temp)
            aggregated_data = masked
        data_list.append(aggregated_data)
        averaged = np.mean(data_list, axis = 0)
    # averaged = [sum(data_list[i:i+12]) / 12 for i in range(0, len(data_list) - len(data_list) % 12, 12)]
    return averaged, data_list

def processing(dir_control, dir_case, response, lat_bound):
    data_list_control, monthly_control = avg_case_data(dir_control, response)

    dir = dir_case
    data_list, monthly_case  = avg_case_data(dir, response)

## -------------- Delta plot -----------------
    # Prepare differences
    diff = np.array(data_list) - np.array(data_list_control)
    # Calculate statistics
    mean = regional_mean(diff, response, lat_bound)

    monthly_diff = np.array(monthly_case) - np.array(monthly_control)
    annual_diff = annual_average(monthly_diff)
    annual_regional_diff = [
        regional_mean(data, response, lat_bound)
        for data in annual_diff
    ]
    dof_eff = effective_dof(annual_regional_diff)
    if dof_eff is not None:
        error = confidence_error(annual_regional_diff, dof_eff)
    else:
        error = 0.0

    print(f"{response}: {mean:.3f} ± {error:.2f}")
    return mean, error

case_list_initial = ['all_year', 'all_year', 'summer', 'winter', 'all_year' ,'all_year', 'summer', 'winter', 'all_year', 'summer', 'winter','all_year','all_year', 'summer', 'winter']
location_list = ['CC_BC_only', 'SA', 'SA', 'SA', 'AU', 'EQ', 'EQ', 'EQ', 'CC', 'CC', 'CC', 'UK' , 'RU', 'RU', 'RU']
lat_list = [29,-55,-55,-55,-29,0,0,0,29,29,29,55,70,70,70]
date_list = ['AUG_13', 'JUL_8','JUL_8','JUL_8','AUG_13','AUG_13','AUG_13','AUG_13','JUL_8','JUL_8','JUL_8','AUG_13','JUL_8','JUL_8','JUL_8']


def main(region, lat_bound):
    for k, response in enumerate(['T', 'O3', 'CLOUD']):
        results = []
        errors = []
        for i in range(len(case_list_initial)):
            control_list = []
            case_list = []
            launch_season = case_list_initial[i]
            location = location_list[i]
            date = date_list[i]
            
            # Create a unique readable identifier for the current case
            case_identifier = f"{date}_30Gg_{launch_season}_{location}"
            dir_case = f'results_cloud/lon/{case_identifier}/'
            dir_control = 'results_cloud/lon/JUL_8_control/'
            
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
            control_list = control_list[120:]
            case_list = case_list[120:]
        
            
            print(f"Processing case: {case_identifier}...")

            if case_list and control_list:
                # Store the returned string output into the dictionaries
                mean, error = processing(dir_control, dir_case, response, lat_bound)
                results.append(mean)
                errors.append(error)
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
        write_to_file(f'data_file/{response}_{region}.txt', results)
        write_to_file(f'data_file/{response}_{region}_error.txt', errors)
        print("Process Complete!")

if __name__ == "__main__":
    main('NH', slice(49, None))
    main('SH', slice(16, 49))
    main('SP', slice(0, 16))
