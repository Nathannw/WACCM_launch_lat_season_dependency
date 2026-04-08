Date: 6 April, 2026

Dataset Title: Dependence of Atmospheric and Climate Impacts on Launch Latitude and Seasonal Variation in Rocket Emissions

Dataset Contact: Nattanan Wongprapinkul nattanan@umich.edu

Dataset Creators: 
Name: Nattanan Wongprapinkul
Email: nattanan@umich.edu
Institution: University of Michigan Department of Aerospace Engineering
ORCID: 

Name: Gokcin Cinar
Email: cinar@umich.edu
Institution: University of Michigan Department of Aerospace Engineering
ORCID: 

Name: Oliver Jia-Richards
Email: oliverjr@umich.edu
Institution: University of Michigan Department of Aerospace Engineering
ORCID: 

Funding: Royal Thai Government Scholarship, funded by the Geo-Informatics and Space Technology Development Agency, Thailand.

Methodology:
The data are model output form climate simulations conducted with the Whole Atmosphere Community Climate Model (WACCM) version 6/Community Earth System Model (CESM) version 2.2.2.
Date Coverage: 1999-2020

Data Description:
Model outputs from WACCM6/CESM2 are stored within `results_cloud.zip`. The configurations are categorized by season and location:

Seasonal Configurations
all_year: Year-Round configuration
summer: Summer-Only configuration
winter: Winter-Only configuration

Launch Location Configurations
| RU | Latitude: 70 | Longitude: 120 | Description: Russia |
| UK | Latitude: 55 | Longitude: -1.6 | Description: The United Kingdom |
| CC | Latitude: 28.47 | Longitude: -80.57 | Description: Cape Canaveral |
| EQ | Latitude: 0 | Longitude: -50 | Description: French Guiana |
| AU | Latitude: -29 | Longitude : 153.5 | Description: Australia |
| SA | Latitude: -55 | Longitude: -68 | Description: South America |

Generating Emission Files
To generate emission files, run:
'create_emission.py'
The 'create_emission.py' script requires the launch location (latitude and longitude) and the launch season as inputs. 
This generates Black Carbon emissions between 11–70 km at a total rate of 30 Gg per year and output as .nc files store under 'emission_file/*case*/'.
The `emission_profile` function inside `analysis/emission_funcs.py` reads `Falcon9_trajectory_THAICOM_8.csv`. 
It calculates the altitude profile of Falcon 9 emissions before applying the relevant emission indices.

Running the Analysis
1.  Decompress Data: `results_cloud.zip` must be decompressed before starting the analysis.
2.  Execute Analysis:
    The `main.py` script invokes the functions located in the `analysis/` folder. It performs the following steps:
    * Reads the results data stored in the `results_cloud/` directory.
    * Processes and stores the required analysis data in `.txt` format within the `data_file/` directory.
    * Generates and saves plots in the `fig/` directory.
    * Computes additional data used to construct the tables presented in the manuscript.

Use and Access:
This data set is made available under a Creative Commons Public Domain license (CC0 1.0).
