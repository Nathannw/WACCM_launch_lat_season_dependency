from analysis.emission_funcs import *
from analysis.func import *
import csv
import netCDF4 as nc
from netCDF4 import Dataset as NetCDFFile
import matplotlib.pyplot as plt
import numpy as np
import os
from pathlib import Path
from scipy.stats import ttest_1samp, ttest_ind
from analysis.figure_style import COLORS, panel_label

cases = {
    'RU':    {'dir': 'JUL_8_30Gg_all_year_RU',          'color': 'black',       'label': '70° N',         'ls': '-'},
    'UK':    {'dir': 'AUG_13_30Gg_all_year_UK',         'color': COLORS["orange"],     'label': '55° N',         'ls': '-'},
    'CC':    {'dir': 'JUL_8_30Gg_all_year_CC',          'color': COLORS["green"],      'label': '29° N',         'ls': '-'},
    'CC_bc': {'dir': 'AUG_13_30Gg_all_year_CC_BC_only', 'color': COLORS["green"],      'label': '29° N BC-only', 'ls': ':'},
    'EQ':    {'dir': 'AUG_13_30Gg_all_year_EQ',         'color': COLORS["purple"],     'label': '0° N',          'ls': '-'},
    'AU':    {'dir': 'AUG_13_30Gg_all_year_AU',         'color': COLORS["vermillion"], 'label': '29° S',         'ls': '-'},
    'SA':    {'dir': 'JUL_8_30Gg_all_year_SA',          'color': COLORS["sky_blue"],   'label': '55° S',         'ls': '-'}
}

titles_delta = [
    'Stratospheric temperature anomaly (case minus control) (K)',
    'Ozone column anomaly (case minus control) (DU)',
    'Stratospheric cloud fraction anomaly (case minus control)',
    r'Stratospheric cloud ice content anomaly (case minus control) ($\mathrm{kg}\ \mathrm{kg}_{\mathrm{air}}^{-1}$)',
    r'Stratospheric cloud liquid content anomaly (case minus control) ($\mathrm{kg}\ \mathrm{kg}_{\mathrm{air}}^{-1}$)'
]

responses = ['T', 'O3', 'CLOUD']

def avg_case_data(dir, response):
    case_list = []
    for root, dirs, files in os.walk(dir):
        for file in files:
            if file.endswith('.nc'):
                case_list.append(os.path.join(root, file))
    case_list.sort(key=lambda x: os.path.basename(x).lower())
    data_list = []
    # Exclude first 10 year
    case_list = case_list[120:]
    for case in case_list:
        nc = NetCDFFile(case)
        # print(nc)
        if response in ['FLNT', 'FSNT', 'FLNTC', 'FSNTC', 'FLN200', 'FLN200C', 'FSN200', 'FSN200C']:
            aggregated_data = np.mean(nc.variables[response], axis = (0))
        elif response == 'O3':
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
    averaged_annual = [sum(data_list[i:i+12]) / 12 for i in range(0, len(data_list) - len(data_list) % 12, 12)]
    return averaged, averaged_annual

def save_latitude_diffs(response, lats, case_results):
    output_path = f'data_file/{response}_lat.csv'
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with open(output_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['launch_latitude', 'latitude', 'diff'])
        for case, results in case_results.items():
            diff = np.ma.asarray(results['diff'], dtype=float).filled(np.nan)
            launch_latitude = cases[case]['label']
            for lat, diff_value in zip(lats, diff):
                writer.writerow([launch_latitude, f"{lat:.10e}", f"{diff_value:.10e}"])

    print(f"Saved: {output_path}")

def main():
    lats = np.loadtxt('geo_file/lat.txt')

    for k, response in enumerate(responses):
        print(f"Diagnose autocorrelation Lag-1: {response}")
        
        # 1. Load Control Data once
        dir_control = 'results_cloud/lon/JUL_8_control/'
        ctrl_data, ctrl_err = [np.array(x) for x in avg_case_data(dir_control, response)]
        
        case_results = {}

        # 2. Loop through the dictionary to load data, get diffs, and calc stats
        for case, cfg in cases.items():
            dir_path = f'results_cloud/lon/{cfg["dir"]}/'
            data, err = [np.array(x) for x in avg_case_data(dir_path, response)]
            
            diff = data - ctrl_data
            err_diff = err - ctrl_err
            t_stat, _ = ttest_1samp(err_diff, 0)
            
            p_vals = []
            for j in range(len(t_stat)):
                x = err_diff[:, j]
                rho1 = np.corrcoef(x[1:], x[:-1])[0, 1]
                _, _, p_val = Neff_calculation_p_val(rho1, t_stat[j])
                p_vals.append(p_val)
                
            case_results[case] = {'diff': diff, 'p_vals': p_vals}

        save_latitude_diffs(response, lats, case_results)

        # 3. Plotting
        plt.figure(figsize=(8, 8))
        
        for case, cfg in cases.items():
            diff = case_results[case]['diff']
            p_vals = case_results[case]['p_vals']
            
            # Plot segment by segment
            for i in range(len(lats) - 1):
                # Forces lw=1 for CC_bc. 
                if case == 'CC_bc':
                    lw = 1 
                else:
                    lw = 4 if p_vals[i] < 0.05 else 1
                    
                plt.plot(diff[i:i+2], lats[i:i+2], color=cfg['color'], linewidth=lw, linestyle=cfg['ls'])
            
            plt.plot([], [], color=cfg['color'], linewidth=2, linestyle=cfg['ls'], label=cfg['label'])

        # 4. Annotations and Formatting
        if response in ['O3', 'T']:
            plt.text(0.02, 1.05, panel_label('a'), transform=plt.gca().transAxes, fontsize=17, fontweight='bold')
        elif response == 'CLOUD':
            plt.text(0.02, 1.05, panel_label('b'), transform=plt.gca().transAxes, fontsize=17, fontweight='bold')
            # plt.ylim([-90, -50])

        plt.ylabel("Latitude (°)", fontsize=17)
        
        xlabel_title = titles_delta[k] if k < len(titles_delta) else f'{response} Anomaly'
        plt.xlabel(xlabel_title, fontsize=17)
        
        plt.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.7)
        plt.legend(fontsize=17, loc='best', frameon=False)
        plt.xticks(fontsize=17)
        plt.yticks(fontsize=17)
        plt.tight_layout()
        output_path = Path("fig") / "lat" / f"lat_{response}.pdf"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path)
        # plt.show()

if __name__ == "__main__":
    main()
