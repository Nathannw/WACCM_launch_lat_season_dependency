# WACCM_launch_lat_season_dependency

This repository contains emission file generation scripts, WACCM6/CESM2 model outputs, and analysis code used to produce figures and datasets for the manuscript: **“Dependence of Atmospheric and Climate Impacts on Launch Latitude and Seasonal Variation in Rocket Emissions.”**

## Requirements
* **Python:** ≥ 3.8
* **Key Libraries:**
    * `numpy`
    * `xarray`
    * `matplotlib`
    * `netcdf4`

## Data Description
Model outputs from **WACCM6/CESM2** are stored within `results_cloud.zip`. The configurations are categorized by season and location:

### Seasonal Configurations
* `all_year`: Year-Round configuration
* `summer`: Summer-Only configuration
* `winter`: Winter-Only configuration

### Launch Location Configurations
| ID | Latitude | Longitude | Location Description |
| :--- | :--- | :--- | :--- |
| **RU** | 70 | 120 | Russia |
| **UK** | 55 | -1.6 | The United Kingdom |
| **CC** | 28.47 | -80.57 | Cape Canaveral |
| **EQ** | 0 | -50 | French Guiana |
| **AU** | -29 | 153.5 | Australia |
| **SA** | -55 | -68 | South America |

---

## Generating Emission Files
To generate emission files, run:
'create_emission.py'
The 'create_emission.py' script requires the launch location (latitude and longitude) and the launch season as inputs. This generates Black Carbon emissions between 11–70 km at a total rate of 30 Gg per year.

The `emission_profile` function inside `analysis/emission_funcs.py` reads `Falcon9_trajectory_THAICOM_8.csv`. It calculates the altitude profile of Falcon 9 emissions before applying the relevant emission indices.

---

## Running the Analysis
1.  **Decompress Data:** `results_cloud.zip` must be decompressed before starting the analysis.
2.  **Execute Analysis:**
    ```bash
    python main.py
    ```
    The `main.py` script invokes the functions located in the `analysis/` folder. It performs the following steps:
    * Reads the results data stored in the `results_cloud/` directory.
    * Processes and stores the required analysis data in `.txt` format within the `data_file/` directory.
    * Generates and saves plots in the `fig/` directory.
    * Computes additional data used to construct the tables presented in the manuscript.


