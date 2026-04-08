from analysis.emission_funcs import *
from analysis.func import *
import netCDF4 as nc
from netCDF4 import Dataset as NetCDFFile
import matplotlib.pyplot as plt
import numpy as np
import os
from scipy.stats import ttest_1samp, ttest_ind

# To plot South Pole cloud liquid / ice content

def avg_case_data(dir, response):
    case_list = []
    for root, dirs, files in os.walk(dir):
        for file in files:
            if file.endswith('.nc'):
                case_list.append(os.path.join(root, file))
    case_list.sort(key=lambda x: os.path.basename(x).lower())
    data_list = []
    # Exclude first 5 years
    case_list = case_list[60:]
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
            for a in range(96):
                idx_lower, idx_upper, ratio = find_trop_func(nc, a)
                X1 = np.mean(nc.variables[response][:, 25:idx_lower[0],a], axis = (0,1))
                X2 = np.mean(nc.variables[response][:, 25:idx_upper[0],a], axis = (0,1)) 
                aggregated_data_1 = X1 * (1 - ratio[0]) + X2 * ratio[0]
                aggregated_data_temp.append(aggregated_data_1)
            masked = np.ma.masked_array(aggregated_data_temp)
            aggregated_data = masked
        data_list.append(aggregated_data)
    averaged = np.mean(data_list, axis = 0)
    averaged_annual = [sum(data_list[i:i+12]) / 12 for i in range(0, len(data_list) - len(data_list) % 12, 12)]
    return averaged, averaged_annual

cases = {
    'RU':    {'dir': 'JUL_8_30Gg_all_year_RU',          'color': 'black'},
    'UK':    {'dir': 'AUG_13_30Gg_all_year_UK',         'color': 'darkorange'},
    'CC':    {'dir': 'JUL_8_30Gg_all_year_CC',          'color': 'forestgreen'},
    'CC_bc': {'dir': 'AUG_13_30Gg_all_year_CC_BC_only', 'color': 'forestgreen'},
    'EQ':    {'dir': 'AUG_13_30Gg_all_year_EQ',         'color': 'purple'},
    'SA':    {'dir': 'JUL_8_30Gg_all_year_SA',          'color': 'teal'},
    'AU':    {'dir': 'AUG_13_30Gg_all_year_AU',         'color': 'crimson'}
}

labels = ["Ice amount", "Liquid amount"]
line_styles = ['-', '-.']
titles_delta = [
    'Stratospheric Cloud Ice and Liquid Content Anomaly (Case – Control) (kg/kg)', 
    'Stratospheric Cloud Ice and Liquid Content Anomaly \n(Case – Control) (kg/kg)'
]

def main():
    lats = np.loadtxt('geo_file/lat.txt')

    # Initialize figure once for both iterations
    plt.figure(figsize=(8, 6))

    # 2. Iterate through responses
    for k, response in enumerate(['CLDICE', 'CLDLIQ']):
        print(f"Diagnose autocorrelation Lag-1: {response}")
        
        # Load Control Data
        dir_control = 'results_cloud/lon/JUL_8_control/'
        ctrl_data, ctrl_err = [np.array(x) for x in avg_case_data(dir_control, response)]
        
        # Process and Plot each case
        for case, cfg in cases.items():
            dir_path = f'results_cloud/lon/{cfg["dir"]}/'
            data, err = [np.array(x) for x in avg_case_data(dir_path, response)]
            
            diff = data - ctrl_data
            err_diff = err - ctrl_err
            t_stat, _ = ttest_1samp(err_diff, 0)
            
            # Calculate p-values
            p_vals = []
            for j in range(len(t_stat)):
                x = err_diff[:, j]
                rho1 = np.corrcoef(x[1:], x[:-1])[0, 1]
                _, _, p_val = Neff_calculation_p_val(rho1, t_stat[j])
                p_vals.append(p_val)
                
            # Determine base linestyle (CC_bc is always dotted)
            ls = ':' if case == 'CC_bc' else line_styles[k]
            
            # Plot segments
            for i in range(len(lats) - 1):
                # CLDLIQ or CC_bc always get linewidth 1. Otherwise, evaluate significance.
                if response == 'CLDLIQ' or case == 'CC_bc':
                    lw = 1
                else:
                    lw = 4 if p_vals[i] < 0.05 else 1
                    
                plt.plot(diff[i:i+2], lats[i:i+2], color=cfg['color'], linewidth=lw, linestyle=ls)
                
        # Add a grey legend line to distinguish Ice vs Liquid
        plt.plot([], [], color='gray', linewidth=2, linestyle=line_styles[k], label=labels[k])

    # 3. Final Annotations and Formatting
    # Add the specific label for CC_bc
    plt.plot([], [], color='forestgreen', linewidth=2, linestyle=':', label='29°N BC-only Ice amount')

    plt.text(0.02, 1.05, '(c)', transform=plt.gca().transAxes, fontsize=12, fontweight='bold')

    plt.ylim([-90, -50])
    plt.xlim([-0.6e-8, 0.6e-8])
    plt.ylabel("Latitude (°)", fontsize=14)

    # Use the final title from your array (since k=1 when the loop finishes)
    plt.xlabel(titles_delta[-1], fontsize=14)

    plt.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.7)
    plt.legend(fontsize=11, loc='best', frameon=False) # Only need to call this once
    plt.tight_layout()
    plt.savefig('fig/lat/lat_cloud_ice_liquid.pdf')
    # plt.show()

if __name__ == "__main__":
    main()