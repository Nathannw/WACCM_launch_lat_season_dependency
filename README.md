# WACCM_launch_lat_season_dependency
This repository contains emission file generation scripts, WACCM6/CESM2 model outputs, and analysis code used to produce figures and datasets for the manuscript “Dependence of Atmospheric and Climate Impacts on Launch Latitude and Seasonal Variation in Rocket Emissions.”
# Data discription
Output from WACCM6/CESM2 stored inside results_cloud.zip 
all_year refers to Year-Round configuration
summer refers to Summer-Only configuration
winter refers to Winter-Only configuration

"RU" refers to {"lat": 70, "lon": 120} configuration
"UK" refers to {"lat": 55, "lon": -1.6} configuration
"CC" refers to {"lat": 28.47, "lon": -80.57} configuration
"EQ" refers to {"lat": 0, "lon": -50} configuration
"AU" refers to {"lat": -29, "lon": 153.5} configuration
"SA" refers to {"lat": -55, "lon": -68} configuration


# To generate emission files
Run create_emission.py
This create_emission.py takes the launch location (latitude, longitude) and launch season. This will generate 11-70km BC emission of 30Gg per year.
def emission_profile function inside analysis/emission_funcs.py will read Falcon9_trajectory_THAICOM_8.csv, where it calculates altitude profile of Falcon9 emission before applying emission indices

# To run data analysis
In order to run the analysis, results_cloud.zip has to be decompressed.
Run main.py
This main.py calls the fuctions under analysis folder. It begins with reading the results data stored under results_cloud directory. Then stores the required data for analysis in .txt format under data_file directory.
The plots will then be generated and stored under fig directory.
Finally it will run additional data used to construct Tables in manuscript.


