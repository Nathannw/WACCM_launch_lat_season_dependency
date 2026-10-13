from analysis.emission_funcs import *
from analysis.func import *
import netCDF4 as nc
from netCDF4 import Dataset as NetCDFFile
import numpy as np
import os
import scipy.stats as stats

# To generate climate responses data

def area_weighted_mean(values, weights):
    values = np.ma.masked_invalid(np.ma.array(values))
    weights = np.ma.array(weights, mask=np.ma.getmaskarray(values))
    return np.ma.sum(values * weights) / np.ma.sum(weights)

def global_area_weighted_mean(values):
    values = np.ma.masked_invalid(np.ma.array(values))
    area, _, _ = col_area()
    zonal_area = np.mean(area, axis=1)

    if values.shape[-1] == len(zonal_area):
        weights = zonal_area
        # print("Using zonal area weights for global area weighted mean.")
    elif values.ndim >= 2 and values.shape[-2] == len(zonal_area):
        weights = zonal_area[:, np.newaxis]
        # print("Using zonal area weights for global area weighted mean.")
    else:
        # print("Warning: Unable to determine appropriate area weights for global area weighted mean. Using unweighted mean instead.")
        return np.ma.mean(values)
        

    weights = np.broadcast_to(weights, values.shape)
    weights = np.ma.array(weights, mask=np.ma.getmaskarray(values))
    return np.ma.sum(values * weights) / np.ma.sum(weights)

def effective_dof(data, min_dof=2.0):
    data = np.asarray(data, dtype=float)
    data = data[np.isfinite(data)]
    n = len(data)
    if n <= 1:
        return None
    if n <= min_dof + 1 or np.isclose(np.std(data), 0):
        return n - 1

    rho1 = np.corrcoef(data[1:], data[:-1])[0, 1]
    if not np.isfinite(rho1) or rho1 <= 0:
        return n - 1

    neff = n * (1 - rho1) / (1 + rho1)
    dof = neff - 1
    return min(max(dof, min_dof), n - 1)

def confidence_error(data, dof):
    data = np.asarray(data, dtype=float)
    data = data[np.isfinite(data)]
    if dof is None or len(data) <= 1:
        return 0.0

    sem = stats.sem(data)
    if not np.isfinite(sem):
        return 0.0

    return float(stats.t.ppf(0.975, df=dof) * sem)

def avg_case_data(dir, response):
    case_list = []
    for root, dirs, files in os.walk(dir):
        for file in files:
            if file.endswith('.nc'):
                case_list.append(os.path.join(root, file))
    case_list.sort(key=lambda x: os.path.basename(x).lower())
    # Exclude first 5 year
    case_list = case_list[120:]
    data_list = []
    for case in case_list:
        nc = NetCDFFile(case)
        if response in ['FLNT', 'FSNT', 'FLNTC', 'FSNTC', 'FLN200', 'FLN200C', 'FSN200', 'FSN200C']:
            aggregated_data = global_area_weighted_mean(nc.variables[response][:])
        elif response == 'O3':
            mol_air = nc.variables['MASS'][0, :, :]*1000/28.97  # (lev, lat)
            O3 = nc.variables['O3'][0, :, :]         # (lev, lat)
            mol_O3 = O3*mol_air
            area, x, y = col_area()
            zonal_area = np.mean(area, axis=1) # cm^2 per-grid-cell area by latitude
            mol_O3percm2 = np.sum(mol_O3/zonal_area,axis = 0)
            DU = mol_O3percm2/4.4615e-8
            aggregated_data = area_weighted_mean(DU, zonal_area)
        elif response in ['CLOUD','CLDICE', 'CLDLIQ', 'T']:
            aggregated_data_temp = []
            response_data = nc.variables[response][:]
            pressure_weights = pressure_level_weights(nc)
            nlat = response_data.shape[-1]
            for a in range(nlat):
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
            area, _, _ = col_area()
            zonal_area = np.mean(area, axis=1)
            aggregated_data = area_weighted_mean(aggregated_data_temp, zonal_area[:nlat])
        data_list.append(aggregated_data)
    averaged = [sum(data_list[i:i+12]) / 12 for i in range(0, len(data_list) - len(data_list) % 12, 12)]
    return averaged, data_list

def processing(dir_control, dir_case, response):
    data_list_control, _ = avg_case_data(dir_control, response)

    dir = dir_case
    data_list, _  = avg_case_data(dir, response)

## -------------- Delta plot -----------------
    # Prepare differences
    diff = np.array(data_list) - np.array(data_list_control)
    # Calculate statistics
    mean = np.mean(diff)
    dof_eff = effective_dof(diff)
    error = confidence_error(diff, dof_eff)
    print(f"{response}: {mean:.3f} ± {error:.2f}\n")
    return float(mean), float(error)

case_list_initial = ['all_year', 'all_year', 'summer', 'winter', 'all_year' ,'all_year', 'summer', 'winter', 'all_year', 'summer', 'winter','all_year','all_year', 'summer', 'winter']
location_list = ['CC_BC_only', 'SA', 'SA', 'SA', 'AU', 'EQ', 'EQ', 'EQ', 'CC', 'CC', 'CC', 'UK' , 'RU', 'RU', 'RU']
lat_list = [29,-55,-55,-55,-29,0,0,0,29,29,29,55,70,70,70]
date_list = ['AUG_13', 'JUL_8','JUL_8','JUL_8','AUG_13','AUG_13','AUG_13','AUG_13','JUL_8','JUL_8','JUL_8','AUG_13','JUL_8','JUL_8','JUL_8']


def main():
    for k, response in enumerate(['T', 'O3', 'FLNT', 'FSNT', 'FLNTC', 'FSNTC']):
    # for k, response in enumerate(['FLNT', 'FSNT', 'FLNTC', 'FSNTC']):
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
            
            print(f"Processing case: {case_identifier}...")

            if case_list and control_list:
                # Store the returned string output into the dictionaries
                mean, error = processing(dir_control, dir_case, response)
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
        write_to_file(f'data_file/{response}.txt', results)
        write_to_file(f'data_file/{response}_error.txt', errors)
        if response == 'FLNT':
            FLNT = results
            FLNT_error = errors
        elif response == 'FSNT':
            FSNT = results
            FSNT_error = errors
        elif response == 'FLNTC':
            FLNTC = results
            FLNTC_error = errors
        elif response == 'FSNTC':
            FSNTC = results
            FSNTC_error = errors
    print("Evaluation ERF and ERFC...")
    write_to_file(f'data_file/ERF.txt', (np.array(FSNT) - np.array(FLNT)).tolist())
    write_to_file(
        f'data_file/ERF_error.txt',
        np.sqrt(np.array(FSNT_error)**2 + np.array(FLNT_error)**2).tolist()
    )
    write_to_file(f'data_file/ERFC.txt', (np.array(FSNTC) - np.array(FLNTC)).tolist())
    write_to_file(
        f'data_file/ERFC_error.txt',
        np.sqrt(np.array(FSNTC_error)**2 + np.array(FLNTC_error)**2).tolist()
    )
    print("Process Complete!")

if __name__ == "__main__":
    main()
