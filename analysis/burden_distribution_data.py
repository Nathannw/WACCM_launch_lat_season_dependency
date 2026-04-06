import os
import numpy as np
import scipy.stats as stats
from netCDF4 import Dataset as NetCDFFile
from analysis.emission_funcs import *
from analysis.func import *

# To generate emission burden data and the distribution of BC (vertical and latitudinal)

lats = np.loadtxt('geo_file/lat.txt')

case_list_initial = ['all_year', 'all_year', 'summer', 'winter', 'all_year' ,'all_year', 'summer', 'winter', 'all_year', 'summer', 'winter','all_year','all_year', 'summer', 'winter']
location_list = ['CC_BC_only', 'SA', 'SA', 'SA', 'AU', 'EQ', 'EQ', 'EQ', 'CC', 'CC', 'CC', 'UK' , 'RU', 'RU', 'RU']
lat_list = [29,-55,-55,-55,-29,-0,0,0,29,29,29,55,70,70,70]
date_list = ['AUG_13', 'JUL_8','JUL_8','JUL_8','AUG_13','AUG_13','AUG_13','AUG_13','JUL_8','JUL_8','JUL_8','AUG_13','JUL_8','JUL_8','JUL_8']


def confidence(data, dof, unit_conversion, unit):
    mean = np.mean(data)
    sem = stats.sem(data)  
    confidence_level = 0.95

    # Calculate actual margin of error using t_crit
    t_crit = stats.t.ppf((1 + confidence_level) / 2, df=dof)
    margin_of_error = t_crit * sem
    
    mean_conv = mean / unit_conversion
    moe_conv = margin_of_error / unit_conversion
    
    print(f"{mean_conv:.2f} ± {moe_conv:.2f} {unit} (95% CI)")
    return mean_conv

def BC_burden(control_list, case_list):
    data_list = []
    for control, case in zip(control_list, case_list):
        aggregated_data_temp = []
        
        with NetCDFFile(control) as nc, NetCDFFile(case) as nc1:
            # Read into memory ONCE per file
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
                
        data_list.append(np.ma.masked_array(np.sum(aggregated_data_temp)))
        
    data_con = [sum(data_list[i:i+12])*144 / 12 for i in range(0, len(data_list) - len(data_list) % 12, 12)]
    
    if len(data_con) > 1:
        rho1_case = np.corrcoef(data_con[1:], data_con[:-1])[0, 1]
        Neff, dof_eff = Neff_calculation(rho1_case)
        print('BC Burden...')
        mean_conv = confidence(data_con[:], dof_eff, 1e6, 'Gg')
    return mean_conv

def H2O_burden(control_list, case_list):
    data_list = []
    conversion_factor = 0.018016 / 0.02897
    
    for control, case in zip(control_list, case_list):
        aggregated_data_temp = []
        
        with NetCDFFile(control) as nc, NetCDFFile(case) as nc1:
            h2o_c = nc.variables['H2O'][:]
            mass_c = nc.variables['MASS'][:]
            h2o_case = nc1.variables['H2O'][:]
            mass_case = nc1.variables['MASS'][:]
            
            for a in range(96):
                idx_lower_c, idx_upper_c, ratio_c = find_trop_func(nc, a)
                l_c, u_c = idx_lower_c[0], idx_upper_c[0]
                
                data1_1 = np.sum(h2o_c[:, 18:l_c, a] * mass_c[:, 18:l_c, a] * conversion_factor)
                data1_2 = np.sum(h2o_c[:, 18:u_c, a] * mass_c[:, 18:u_c, a] * conversion_factor)
                agg_1 = data1_1 * (1 - ratio_c[0]) + data1_2 * ratio_c[0]

                idx_lower_case, idx_upper_case, ratio_case = find_trop_func(nc1, a)
                l_case, u_case = idx_lower_case[0], idx_upper_case[0]
                
                data2_1 = np.sum(h2o_case[:, 18:l_case, a] * mass_case[:, 18:l_case, a] * conversion_factor)
                data2_2 = np.sum(h2o_case[:, 18:u_case, a] * mass_case[:, 18:u_case, a] * conversion_factor)
                agg_2 = data2_1 * (1 - ratio_case[0]) + data2_2 * ratio_case[0]
                
                aggregated_data_temp.append(agg_2 - agg_1)
                
        data_list.append(np.ma.masked_array(np.sum(aggregated_data_temp)))
        
    data_con = [sum(data_list[i:i+12])*144 / 12 for i in range(0, len(data_list) - len(data_list) % 12, 12)]
    
    if len(data_con) > 1:
        rho1_case = np.corrcoef(data_con[1:], data_con[:-1])[0, 1]
        Neff, dof_eff = Neff_calculation(rho1_case)
        print('H2O Burden...')
        mean_conv = confidence(data_con[:], dof_eff, 1e9, 'Tg')
    return mean_conv

def NOX_burden(control_list, case_list):
    data_list = []
    conversion_factor = 0.046 / 0.02897
    for control, case in zip(control_list, case_list):
        nc = NetCDFFile(control)
        nc1 = NetCDFFile(case)
        data1 = nc.variables['NOX'][:,18:,:]
        data1 = np.sum(data1*(nc.variables['MASS'][:,18:,:]*conversion_factor))
        data2 = nc1.variables['NOX'][:,18:,:]
        data2 = np.sum(data2*(nc1.variables['MASS'][:,18:,:]*conversion_factor))
        data = data2 - data1
        data_list.append(data)
    data_con = [sum(data_list[i:i+12]*144) / 12 for i in range(0, len(data_list) - len(data_list) % 12, 12)]
    if len(data_con) > 1:
        rho1_case = np.corrcoef(data_con[1:], data_con[:-1])[0, 1]
        Neff, dof_eff = Neff_calculation(rho1_case)
        print('NOx Burden...')
        mean_conv = confidence(data_con[:], dof_eff, 1e6, 'Gg')
    return mean_conv

def mean_BC_latitude(control_list, case_list):
    data_list = []
    for control, case in zip(control_list, case_list):
        aggregated_data_temp = []
        
        with NetCDFFile(control) as nc, NetCDFFile(case) as nc1:
            bc_c = nc.variables['bc_a4'][:] + nc.variables['bc_a1'][:]
            mass_c = nc.variables['MASS'][:]
            bc_case = nc1.variables['bc_a4'][:] + nc1.variables['bc_a1'][:]
            mass_case = nc1.variables['MASS'][:]
            
            data1_1 = np.sum(bc_c[:, 18:, :] * mass_c[:, 18:, :], axis=0)
            data2_1 = np.sum(bc_case[:, 18:, :] * mass_case[:, 18:, :], axis=0)
            
        data_list.append(data2_1 - data1_1)        
        # data_list.append(aggregated_data_temp)
    data_list_mean = np.mean(data_list, axis=0)
    print("Mean BC Latitude...")
    mean_lat  = np.sum(data_list_mean * lats) / np.sum(data_list_mean)
    print(f"{mean_lat:.1f} degrees.")
    return mean_lat

def mean_BC_altitude(control_list, case_list):
    data_list = []
    data_list_alt = []
    
    for control, case in zip(control_list, case_list):
        with NetCDFFile(control) as nc, NetCDFFile(case) as nc1:
            bc_c = nc.variables['bc_a4'][:, 18:, :] + nc.variables['bc_a1'][:, 18:, :]
            mass_c = nc.variables['MASS'][:, 18:, :]
            agg_1 = np.sum(bc_c * mass_c, axis=0)

            bc_case = nc1.variables['bc_a4'][:, 18:, :] + nc1.variables['bc_a1'][:, 18:, :]
            mass_case = nc1.variables['MASS'][:, 18:, :]
            agg_2 = np.sum(bc_case * mass_case, axis=0)

            data = agg_2 - agg_1
            altitudes = np.mean(nc1.variables['Z3'][:, 18:, :], axis=0) / 1000

        data_list.append(data)
        data_list_alt.append(altitudes)
        
    data_list_mean = np.mean(data_list, axis=0)
    alt_mean = np.mean(data_list_alt, axis=0)
    print("Mean BC Altitude...")
    mean_alt = np.sum(data_list_mean * alt_mean) / np.sum(data_list_mean)
    print(f"{mean_alt:.1f} km.")
    return mean_alt

def main():
    results_BC_burden = []
    results_H2O_burden = []
    results_NOX_burden = []
    results_mean_lat = []
    results_mean_alt = []

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
        control_list = control_list[60:]
        case_list = case_list[60:]
        
        print(f"Processing case: {case_identifier}...")

        if case_list and control_list:
            # Store the returned string output into the dictionaries
            results_BC_burden.append(BC_burden(control_list, case_list))
            results_H2O_burden.append(H2O_burden(control_list, case_list))
            results_NOX_burden.append(NOX_burden(control_list, case_list))
            results_mean_lat.append(mean_BC_latitude(control_list, case_list))
            results_mean_alt.append(mean_BC_altitude(control_list, case_list))
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
    write_to_file('data_file/BC_burden.txt', results_BC_burden)
    write_to_file('data_file/H2O_burden.txt', results_H2O_burden)
    write_to_file('data_file/NOX_burden.txt', results_NOX_burden)
    write_to_file('data_file/mean_BC_latitude.txt', results_mean_lat)
    write_to_file('data_file/mean_BC_altitude.txt', results_mean_alt)
    print("Process Complete!")

if __name__ == "__main__":
    main()