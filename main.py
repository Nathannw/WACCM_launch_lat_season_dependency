from analysis import (
    response_data,
    correlation_data,
    burden_distribution_data,
    burden_distribution_plot,
    correlation_plot,
    response_plot,
    response_latitude_plot,
    response_lat_ice_cloud,
    response_season_plot,
    heating_rate_BC_profile,
    T_wind,
    T_wind_season_RU_SA,
    strat_BC,
    response_data_table,
    response_ERF_table,
)

import warnings
warnings.filterwarnings("ignore")

tasks = [
    ("Preparing Data...", response_data.main),
    ("Preparing Correlation Data for Northern Hemisphere...", lambda: correlation_data.main('NH', slice(49, None))),
    ("Preparing Correlation Data for Southern Hemisphere...", lambda: correlation_data.main('SH', slice(16, 49))),
    ("Preparing Correlation Data for South Pole...", lambda: correlation_data.main('SP', slice(0, 16))),
    ("Preparing Emission Burdens...", burden_distribution_data.main),
    ("Plotting Response...", response_plot.main),
    ("Plotting Correlation...", correlation_plot.main),
    ("Plotting Burden Distribution...", burden_distribution_plot.main),
    ("Plotting Latitudinal Data...", response_latitude_plot.main),
    ("Plotting Ice Cloud Figures...", response_lat_ice_cloud.main),
    ("Plotting Seasonal Figures...", response_season_plot.main),
    ("Plotting Heating Rate + BC Profile...", heating_rate_BC_profile.main),
    ("Plotting Temperature & Wind Figures...", T_wind.main),
    ("Plotting Seasonal Wind Figures...", T_wind_season_RU_SA.main),
    ("Figures Stratospheric BC Temporal Profile...", strat_BC.main),
    ("Table Data...", response_data_table.main),
    ("ERF Table...", lambda: response_ERF_table.main(['FSNT', 'FLNT'])),
    ("ERFC Table...", lambda: response_ERF_table.main(['FSNTC', 'FLNTC'])),
]

for message, func in tasks:
    print(message)
    func()