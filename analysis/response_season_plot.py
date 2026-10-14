from analysis.emission_funcs import *
from analysis.func import *
import netCDF4 as nc
from netCDF4 import Dataset as NetCDFFile
import matplotlib.pyplot as plt
import numpy as np
import os
from pathlib import Path
import re
from scipy.stats import ttest_1samp, ttest_ind
from analysis.figure_style import COLORS, panel_label

# To obtain and plot latitudinal response anomalies of seasonal case compared to year-round case

SEASONS = {
    "DJF": (12, 1, 2),
    # "MAM": (3, 4, 5),
    "JJA": (6, 7, 8),
    # "SON": (9, 10, 11),
    "Annual": tuple(range(1, 13)),
}


def periods_for_response(response):
    # if response == "CLOUD":
    #     return {"Annual": SEASONS["Annual"]}
    return SEASONS


def case_year_for_season(year, month, season):
    if season == "DJF" and month == 12:
        return year + 1
    return year


def case_month(path):
    match = re.search(r"\.(\d{4})-(\d{2})\.nc$", os.path.basename(path))
    if not match:
        raise ValueError(f"Could not parse year-month from {path}")
    return int(match.group(1)), int(match.group(2))


def load_case_data(dir, response):
    case_list = []
    for root, dirs, files in os.walk(dir):
        for file in files:
            if file.endswith('.nc'):
                case_list.append(os.path.join(root, file))
    case_list.sort(key=lambda x: os.path.basename(x).lower())
    records = []
    # Exclude first 10 years
    case_list = case_list[120:]
    for case in case_list:
        year, month = case_month(case)
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
            # aggregated_data = np.array(aggregated_data_temp)
            masked = np.ma.masked_array(aggregated_data_temp)
            aggregated_data = masked
        records.append((year, month, aggregated_data))
    return records


def avg_case_data_by_season(dir, response, periods):
    records = load_case_data(dir, response)
    seasonal_data = {}

    for season, months in periods.items():
        grouped = {}
        for year, month, data in records:
            if month not in months:
                continue

            season_year = case_year_for_season(year, month, season)
            grouped.setdefault(season_year, {})[month] = data

        samples_by_year = {}
        for year in sorted(grouped):
            month_data = grouped[year]
            if all(month in month_data for month in months):
                samples_by_year[year] = np.mean([month_data[month] for month in months], axis=0)

        samples = np.asarray([samples_by_year[year] for year in sorted(samples_by_year)])
        seasonal_data[season] = {
            "mean": np.mean(samples, axis=0),
            "samples": samples,
            "samples_by_year": samples_by_year,
        }

    return seasonal_data


def seasonal_case_difference(case_data, all_year_data, periods):
    results = {}

    for season in periods:
        case_season = case_data[season]
        all_year_season = all_year_data[season]
        common_years = sorted(
            set(case_season["samples_by_year"]) & set(all_year_season["samples_by_year"])
        )

        case_samples = np.asarray([case_season["samples_by_year"][year] for year in common_years])
        all_year_samples = np.asarray([all_year_season["samples_by_year"][year] for year in common_years])

        if len(common_years) == 0:
            raise ValueError(f"No overlapping complete {season} samples found")

        diff = np.mean(case_samples, axis=0) - np.mean(all_year_samples, axis=0)
        err_diff = case_samples - all_year_samples
        t_stat, _ = ttest_1samp(err_diff, 0)

        p_vals = []
        for j in range(len(t_stat)):
            x = err_diff[:, j]
            rho1 = np.corrcoef(x[1:], x[:-1])[0, 1]
            _, _, p = Neff_calculation_p_val(rho1, t_stat[j])
            p_vals.append(p)

        results[season] = {'diff': diff, 'p_vals': p_vals}

    return results

loc_configs = {
    "70° N": {"suffix": "RU", "folder": "JUL_8_30Gg"},
    "29° N": {"suffix": "CC", "folder": "JUL_8_30Gg"},
    "0° N":  {"suffix": "EQ", "folder": "AUG_13_30Gg"},
    "55° S": {"suffix": "SA", "folder": "JUL_8_30Gg"},
}

responses = ['T', 'O3', 'CLOUD']
titles_delta = [
    'Stratospheric temperature anomaly (seasonal minus year-round) (K)',
    'Ozone column anomaly (seasonal minus year-round) (DU)',
    'Stratospheric cloud fraction anomaly \n (seasonal minus year-round)'
]

def main():
    lats = np.loadtxt('geo_file/lat.txt')

    for k, response in enumerate(responses):
        print(f"Processing: {response}")
        plot_periods = periods_for_response(response)
        
        # Storage for subplot data
        subplot_data = []

        # 2. Data Loading & Processing Loop
        for loc_name, cfg in loc_configs.items():
            sfx = cfg['suffix']
            fld = cfg['folder']
            
            # Load the three versions for this location
            ay_path   = f'results_cloud/lon/{fld}_all_year_{sfx}/'
            sm_path   = f'results_cloud/lon/{fld}_summer_{sfx}/'
            wn_path   = f'results_cloud/lon/{fld}_winter_{sfx}/'
            
            ay_data = avg_case_data_by_season(ay_path, response, plot_periods)
            sm_data = avg_case_data_by_season(sm_path, response, plot_periods)
            wn_data = avg_case_data_by_season(wn_path, response, plot_periods)

            results = {
                "summer": seasonal_case_difference(sm_data, ay_data, plot_periods),
                "winter": seasonal_case_difference(wn_data, ay_data, plot_periods),
            }
            
            subplot_data.append({'title': loc_name, 'seasons': results})

        # 3. Plotting Logic
        fig, axes = plt.subplots(
            len(loc_configs),
            len(plot_periods),
            figsize=(4, 12) if len(plot_periods) == 1 else (8, 12),
            sharey=True,
            squeeze=False,
        )
        colors = {"summer": COLORS["orange"], "winter": COLORS["blue"]}
        labels = {"summer": "Boreal summer-only", "winter": "Boreal winter-only"}

        for row_idx, loc_res in enumerate(subplot_data):
            for col_idx, season_name in enumerate(plot_periods):
                ax = axes[row_idx, col_idx]
                ax.tick_params(axis="both", labelsize=13)

                for case_name in ("summer", "winter"):
                    season = loc_res["seasons"][case_name][season_name]
                    diff = season["diff"]
                    p_vals = season["p_vals"]

                    for j in range(len(lats) - 1):
                        lw = 4 if p_vals[j] < 0.05 else 1
                        ax.plot(
                            diff[j:j+2],
                            lats[j:j+2],
                            color=colors[case_name],
                            linewidth=lw,
                        )

                if row_idx == 0:
                    ax.set_title(season_name, fontsize=17)
                if col_idx == 0:
                    ax.set_ylabel(loc_res["title"], fontsize=17)

                ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.6)

                if row_idx == 0 and col_idx == 0:
                    if response == 'O3':
                        letter = panel_label('a') 
                    elif response == 'T':
                        letter = ''
                    else:
                        letter = panel_label('b')
                    ax.text(0.02, 1.04, letter, transform=ax.transAxes, fontweight="bold", fontsize=17)

                if response == 'CLOUD':
                    plt.setp(ax.get_xticklabels(), rotation=-45)

        # 4. Final Figure Formatting
        fig.text(0.0, 0.5, "Latitude (°)", va="center", rotation="vertical", fontsize=17)
        fig.text(0.5, 0.07, titles_delta[k], ha="center", fontsize=17)
        
        # if response != "CLOUD":
        # Create a single legend
        legend_elements = [
            plt.Line2D([0], [0], color=colors[name], lw=2, label=labels[name])
            for name in ("summer", "winter")
        ]
        fig.legend(
            handles=legend_elements,
            loc="lower center",
            bbox_to_anchor=(0.5, 0.015),
            ncol=2,
            frameon=False,
            fontsize=17,
        )

        plt.tight_layout(rect=[0.02, 0.09, 1, 0.98])
        # plt.tight_layout()
        output_path = Path("fig") / "season" / f"season_{response}.pdf"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path)
        plt.close(fig)
        # plt.show()

if __name__ == "__main__":
    main()
