import os
import numpy as np
import scipy.stats as stats
from netCDF4 import Dataset as NetCDFFile
from analysis.emission_funcs import *
from analysis.func import *

# To generate emission burden data and the distribution of BC (vertical and latitudinal)

lats = np.loadtxt('geo_file/lat.txt')
START_LEVEL = 18
N_LONGITUDES = 144

case_list_initial = ['all_year', 'all_year', 'summer', 'winter', 'all_year' ,'all_year', 'summer', 'winter', 'all_year', 'summer', 'winter','all_year','all_year', 'summer', 'winter']
location_list = ['CC_BC_only', 'SA', 'SA', 'SA', 'AU', 'EQ', 'EQ', 'EQ', 'CC', 'CC', 'CC', 'UK' , 'RU', 'RU', 'RU']
lat_list = [29,-55,-55,-55,-29,-0,0,0,29,29,29,55,70,70,70]
date_list = ['AUG_13', 'JUL_8','JUL_8','JUL_8','AUG_13','AUG_13','AUG_13','AUG_13','JUL_8','JUL_8','JUL_8','AUG_13','JUL_8','JUL_8','JUL_8']


def confidence(data, dof, unit_conversion, unit):
    data = np.asarray(data, dtype=float)
    data = data[np.isfinite(data)]
    mean = np.mean(data)
    sem = stats.sem(data)
    confidence_level = 0.95

    # Calculate actual margin of error using t_crit
    t_crit = stats.t.ppf((1 + confidence_level) / 2, df=dof)
    margin_of_error = t_crit * sem
    
    mean_conv = mean / unit_conversion
    moe_conv = margin_of_error / unit_conversion
    
    print(f"{mean_conv:.2f} ± {moe_conv:.2f} {unit} (95% CI)")
    return float(mean_conv), float(moe_conv)

def annual_average(data_list):
    return [
        sum(data_list[i:i+12]) / 12
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

def weighted_mean(values, weights):
    return float(np.sum(values * weights) / np.sum(weights))

def cell_edges_from_centers(centers, lower_bound=None, upper_bound=None):
    centers = np.asarray(centers, dtype=float)
    edges = np.empty(len(centers) + 1, dtype=float)
    edges[1:-1] = (centers[:-1] + centers[1:]) / 2
    edges[0] = centers[0] - (edges[1] - centers[0])
    edges[-1] = centers[-1] + (centers[-1] - edges[-2])
    if lower_bound is not None or upper_bound is not None:
        edges = np.clip(edges, lower_bound, upper_bound)
    return edges

def stratospheric_level_weights(nc, lat_idx):
    nlev = len(nc.dimensions['lev'])
    weights = np.zeros(nlev - START_LEVEL)
    idx_lower, _, ratio = find_trop_func(nc, lat_idx)
    lower = int(idx_lower[0])
    boundary_fraction = float(ratio[0])

    full_stop = min(max(lower, START_LEVEL), nlev)
    weights[:full_stop - START_LEVEL] = 1.0

    boundary_idx = lower - START_LEVEL
    if 0 <= boundary_idx < len(weights):
        weights[boundary_idx] = boundary_fraction

    return weights

def monthly_stratospheric_BC_delta(control_nc, case_nc):
    bc_control = control_nc.variables['bc_a4'][:, START_LEVEL:, :] + control_nc.variables['bc_a1'][:, START_LEVEL:, :]
    mass_control = control_nc.variables['MASS'][:, START_LEVEL:, :]
    bc_case = case_nc.variables['bc_a4'][:, START_LEVEL:, :] + case_nc.variables['bc_a1'][:, START_LEVEL:, :]
    mass_case = case_nc.variables['MASS'][:, START_LEVEL:, :]

    nlev = bc_case.shape[1]
    nlat = bc_case.shape[2]
    delta = np.zeros((nlev, nlat))

    for lat_idx in range(nlat):
        control_weights = stratospheric_level_weights(control_nc, lat_idx)
        case_weights = stratospheric_level_weights(case_nc, lat_idx)
        control_mass = np.sum(bc_control[:, :, lat_idx] * mass_control[:, :, lat_idx], axis=0)
        case_mass = np.sum(bc_case[:, :, lat_idx] * mass_case[:, :, lat_idx], axis=0)
        delta[:, lat_idx] = case_mass * case_weights - control_mass * control_weights

    return delta

def to_builtin(data):
    if isinstance(data, np.ma.MaskedArray):
        data = data.filled(np.nan)
    if isinstance(data, np.ndarray):
        return to_builtin(data.tolist())
    if isinstance(data, np.generic):
        return to_builtin(data.item())
    if isinstance(data, dict):
        return {key: to_builtin(value) for key, value in data.items()}
    if isinstance(data, (list, tuple)):
        return [to_builtin(value) for value in data]
    if isinstance(data, float) and not np.isfinite(data):
        return None
    return data

def BC_burden(control_list, case_list):
    data_list = []
    for control, case in zip(control_list, case_list):
        with NetCDFFile(control) as nc, NetCDFFile(case) as nc1:
            data_list.append(np.sum(monthly_stratospheric_BC_delta(nc, nc1)))
        
    data_con = [sum(data_list[i:i+12]) * N_LONGITUDES / 12 for i in range(0, len(data_list) - len(data_list) % 12, 12)]
    
    dof_eff = effective_dof(data_con)
    if dof_eff is not None:
        print('BC Burden...')
        mean_conv, error_conv = confidence(data_con[:], dof_eff, 1e6, 'Gg')
    else:
        mean_conv = float(np.mean(data_con) / 1e6) if data_con else 0.0
        error_conv = 0.0
    return mean_conv, error_conv

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
    
    dof_eff = effective_dof(data_con)
    if dof_eff is not None:
        print('H2O Burden...')
        mean_conv, error_conv = confidence(data_con[:], dof_eff, 1e9, 'Tg')
    else:
        mean_conv = float(np.mean(data_con) / 1e9) if data_con else 0.0
        error_conv = 0.0
    return mean_conv, error_conv

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
    dof_eff = effective_dof(data_con)
    if dof_eff is not None:
        print('NOx Burden...')
        mean_conv, error_conv = confidence(data_con[:], dof_eff, 1e6, 'Gg')
    else:
        mean_conv = float(np.mean(data_con) / 1e6) if data_con else 0.0
        error_conv = 0.0
    return mean_conv, error_conv

def mean_BC_latitude(control_list, case_list):
    data_list = []
    for control, case in zip(control_list, case_list):
        with NetCDFFile(control) as nc, NetCDFFile(case) as nc1:
            data_list.append(monthly_stratospheric_BC_delta(nc, nc1))

    data_list_mean = np.mean(data_list, axis=0) * N_LONGITUDES
    latitude_edges = cell_edges_from_centers(lats, -90, 90)
    latitude_widths = np.diff(latitude_edges)
    bc_mass_by_latitude = np.sum(data_list_mean, axis=0)
    lat_distribution = {
        'latitude_deg': lats,
        'latitude_edges_deg': latitude_edges,
        'bc_mass_kg_by_latitude': bc_mass_by_latitude,
        'bc_mass_density_kg_per_degree': bc_mass_by_latitude / latitude_widths,
        'bc_mass_kg_by_level_lat': data_list_mean,
        'bc_burden_gg': np.sum(data_list_mean) / 1e6,
    }
    print("Mean BC Latitude...")
    mean_lat = weighted_mean(lats, data_list_mean)

    annual_data = annual_average(data_list)
    annual_mean_lat = [weighted_mean(lats, data) for data in annual_data]
    dof_eff = effective_dof(annual_mean_lat)
    if dof_eff is not None:
        mean_lat, error_lat = confidence(annual_mean_lat, dof_eff, 1, 'degrees')
    else:
        error_lat = 0.0
        print(f"{mean_lat:.1f} degrees.")
    return mean_lat, error_lat, lat_distribution

def mean_BC_altitude(control_list, case_list):
    data_list = []
    data_list_alt = []
    
    for control, case in zip(control_list, case_list):
        with NetCDFFile(control) as nc, NetCDFFile(case) as nc1:
            data = monthly_stratospheric_BC_delta(nc, nc1)
            altitudes = np.mean(nc1.variables['Z3'][:, START_LEVEL:, :], axis=0) / 1000

        data_list.append(data)
        data_list_alt.append(altitudes)
        
    data_list_mean = np.mean(data_list, axis=0) * N_LONGITUDES
    alt_mean = np.mean(data_list_alt, axis=0)
    bc_mass_by_level = np.sum(data_list_mean, axis=1)
    altitude_by_level = np.mean(alt_mean, axis=1)
    altitude_order = np.argsort(altitude_by_level)
    altitude_sorted = altitude_by_level[altitude_order]
    altitude_edges = cell_edges_from_centers(altitude_sorted)
    altitude_widths = np.diff(altitude_edges)
    bc_mass_by_level_sorted = bc_mass_by_level[altitude_order]
    alt_distribution = {
        'altitude_km_by_level': altitude_by_level,
        'altitude_edges_km': altitude_edges,
        'bc_mass_kg_by_level': bc_mass_by_level,
        'bc_mass_density_kg_per_km': bc_mass_by_level_sorted / altitude_widths,
        'altitude_km_by_level_lat': alt_mean,
        'bc_mass_kg_by_level_lat': data_list_mean,
        'bc_burden_gg': np.sum(data_list_mean) / 1e6,
    }
    print("Mean BC Altitude...")
    mean_alt = weighted_mean(alt_mean, data_list_mean)

    annual_data = annual_average(data_list)
    annual_alt = annual_average(data_list_alt)
    annual_mean_alt = [
        weighted_mean(altitudes, data)
        for data, altitudes in zip(annual_data, annual_alt)
    ]
    dof_eff = effective_dof(annual_mean_alt)
    if dof_eff is not None:
        mean_alt, error_alt = confidence(annual_mean_alt, dof_eff, 1, 'km')
    else:
        error_alt = 0.0
        print(f"{mean_alt:.1f} km.")
    return mean_alt, error_alt, alt_distribution

def main():
    results_BC_burden = []
    results_BC_burden_error = []
    results_H2O_burden = []
    results_H2O_burden_error = []
    results_NOX_burden = []
    results_NOX_burden_error = []
    results_mean_lat = []
    results_mean_lat_error = []
    results_BC_lat_distribution = []
    results_mean_alt = []
    results_mean_alt_error = []
    results_BC_alt_distribution = []

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

        # Exclude first 10 year
        control_list = control_list[120:]
        case_list = case_list[120:]
        
        print(f"Processing case: {case_identifier}...")

        if case_list and control_list:
            # Store the returned string output into the dictionaries
            mean, error = BC_burden(control_list, case_list)
            results_BC_burden.append(mean)
            results_BC_burden_error.append(error)

            mean, error = H2O_burden(control_list, case_list)
            results_H2O_burden.append(mean)
            results_H2O_burden_error.append(error)

            mean, error = NOX_burden(control_list, case_list)
            results_NOX_burden.append(mean)
            results_NOX_burden_error.append(error)

            mean, error, lat_distribution = mean_BC_latitude(control_list, case_list)
            results_mean_lat.append(mean)
            results_mean_lat_error.append(error)
            results_BC_lat_distribution.append(lat_distribution)

            mean, error, alt_distribution = mean_BC_altitude(control_list, case_list)
            results_mean_alt.append(mean)
            results_mean_alt_error.append(error)
            results_BC_alt_distribution.append(alt_distribution)
        else:
            print(f"Warning: Missing files for {case_identifier}. Skipping...")

    # Helper function to write a dictionary out to a file
    def write_to_file(filename, datas):
        with open(filename, 'w') as f:
            f.write(f"{lat_list}\n")
            f.write(f"{case_list_initial}\n")
            f.write(f"{location_list}\n")
            f.write(repr(to_builtin(datas)))
        print(f"Saved: {filename}")

    # Write all the accumulated data out to respective text files
    print("\nWriting output to .txt files...")
    write_to_file('data_file/BC_burden.txt', results_BC_burden)
    write_to_file('data_file/BC_burden_error.txt', results_BC_burden_error)
    write_to_file('data_file/H2O_burden.txt', results_H2O_burden)
    write_to_file('data_file/H2O_burden_error.txt', results_H2O_burden_error)
    write_to_file('data_file/NOX_burden.txt', results_NOX_burden)
    write_to_file('data_file/NOX_burden_error.txt', results_NOX_burden_error)
    write_to_file('data_file/mean_BC_latitude.txt', results_mean_lat)
    write_to_file('data_file/mean_BC_latitude_error.txt', results_mean_lat_error)
    write_to_file('data_file/BC_latitude_distribution.txt', results_BC_lat_distribution)
    write_to_file('data_file/mean_BC_altitude.txt', results_mean_alt)
    write_to_file('data_file/mean_BC_altitude_error.txt', results_mean_alt_error)
    write_to_file('data_file/BC_altitude_distribution.txt', results_BC_alt_distribution)
    print("Process Complete!")

if __name__ == "__main__":
    main()
