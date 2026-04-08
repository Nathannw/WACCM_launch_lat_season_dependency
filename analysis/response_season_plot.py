from analysis.emission_funcs import *
from analysis.func import *
import netCDF4 as nc
from netCDF4 import Dataset as NetCDFFile
import matplotlib.pyplot as plt
import numpy as np
import os
from scipy.stats import ttest_1samp, ttest_ind

# To obtain and plot latitudinal response anomalies of seasonal case compared to year-round case

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
            # aggregated_data = np.array(aggregated_data_temp)
            masked = np.ma.masked_array(aggregated_data_temp)
            aggregated_data = masked
        elif response in ['V','Wzm']:
            aggregated_data = np.mean(nc.variables[response][:, 25:53,:], axis = (0,1)) # 25 is 50km 53 is 12km
        elif response in ['QRSC','FDSC'] :
            aggregated_data = np.mean(nc.variables[response][:,33:53,:], axis = (0,1)) # 25 is 50km 53 is 12km
        data_list.append(aggregated_data)
    averaged = np.mean(data_list, axis = 0)
    averaged_annual = [sum(data_list[i:i+12]) / 12 for i in range(0, len(data_list) - len(data_list) % 12, 12)]
    return averaged, averaged_annual

loc_configs = {
    "55°S": {"suffix": "SA", "folder": "JUL_8_30Gg"},
    "0°N":  {"suffix": "EQ", "folder": "AUG_13_30Gg"},
    "29°N": {"suffix": "CC", "folder": "JUL_8_30Gg"},
    "70°N": {"suffix": "RU", "folder": "JUL_8_30Gg"}
}

responses = ['T', 'O3', 'CLOUD']
titles_delta = [
    'Stratospheric Temperature Anomaly (Seasonal – Year-Round) (K)',
    'Ozone Column Anomaly (Seasonal – Year-Round) (DU)',
    'Stratospheric Cloud Fraction Anomaly (Seasonal – Year-Round)'
]

def main():
    lats = np.loadtxt('geo_file/lat.txt')

    for k, response in enumerate(responses):
        print(f"Processing: {response}")
        
        # Storage for subplot data
        subplot_data = []

        # 2. Data Loading & Processing Loop
        for loc_name, cfg in loc_configs.items():
            sfx = cfg['suffix']
            fld = cfg['folder']
            
            # Load the three versions for this location
            ctrl_path = 'results_cloud/lon/JUL_8_control/'
            ay_path   = f'results_cloud/lon/{fld}_all_year_{sfx}/'
            sm_path   = f'results_cloud/lon/{fld}_summer_{sfx}/'
            wn_path   = f'results_cloud/lon/{fld}_winter_{sfx}/'
            
            _, ay_err = [np.array(x) for x in avg_case_data(ay_path, response)]
            sm_data, sm_err = [np.array(x) for x in avg_case_data(sm_path, response)]
            wn_data, wn_err = [np.array(x) for x in avg_case_data(wn_path, response)]
            ay_data, _ = [np.array(x) for x in avg_case_data(ay_path, response)]

            # Calculate Diffs and Stats
            results = []
            for d_case, e_case in [(sm_data, sm_err), (wn_data, wn_err)]:
                diff = d_case - ay_data
                err_diff = e_case - ay_err
                t_stat, _ = ttest_1samp(err_diff, 0)
                
                p_vals = []
                for j in range(len(t_stat)):
                    x = err_diff[:, j]
                    rho1 = np.corrcoef(x[1:], x[:-1])[0, 1]
                    _, _, p = Neff_calculation_p_val(rho1, t_stat[j])
                    p_vals.append(p)
                
                results.append({'diff': diff, 'p_vals': p_vals})
            
            subplot_data.append({'title': loc_name, 'seasons': results})

        # 3. Plotting Logic
        fig, axes = plt.subplots(2, 2, figsize=(8, 8), sharey=True)
        axes = axes.flatten()
        colors = ["darkorange", "navy"]
        labels = ["Boreal Summer-Only", "Boreal Winter-Only"]

        for i, ax in enumerate(axes):
            loc_res = subplot_data[i]
            
            for s_idx, season in enumerate(loc_res['seasons']):
                diff = season['diff']
                p_vals = season['p_vals']
                
                for j in range(len(lats) - 1):
                    lw = 4 if p_vals[j] < 0.05 else 1
                    ax.plot(diff[j:j+2], lats[j:j+2], color=colors[s_idx], linewidth=lw)

            ax.set_title(loc_res['title'], fontsize=12)
            ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.6)
            
            # Subplot labels
            letter = '(a)' if response == 'O3' else '(b)'
            if i == 0: ax.text(0.02, 1.02, letter, transform=ax.transAxes, fontweight="bold")
            
            if response == 'CLOUD':
                plt.setp(ax.get_xticklabels(), rotation=-45)

        # 4. Final Figure Formatting
        fig.text(0.0, 0.5, "Latitude (°)", va="center", rotation="vertical", fontsize=14)
        fig.text(0.5, 0.04, titles_delta[k], ha="center", fontsize=14)
        
        # Create a single legend
        legend_elements = [plt.Line2D([0], [0], color=c, lw=2, label=l) for c, l in zip(colors, labels)]
        fig.legend(handles=legend_elements, loc="lower center", ncol=2, frameon=False)

        plt.tight_layout(rect=[0.05, 0.07, 1, 0.98])
        plt.savefig(f"fig/season/season_{response}.pdf")
        # plt.show()

if __name__ == "__main__":
    main()