from analysis.emission_funcs import *
import netCDF4 as nc
from netCDF4 import Dataset as NetCDFFile
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.basemap import Basemap

# To create the emission files of Black Carbon, NOx and H2O in .nc to be ready for WACCM

def create_emission(launch_lat, launch_lon, launch_season, dir, max_alt = 70):
    nc_num_bc = NetCDFFile(dir+'emissions-cmip6_num_bc_a4_rocket_emission_30Gg_2000climo_1.9x2.5_c20200422.nc', "r+")
    nc_bc = NetCDFFile(dir+'emissions-cmip6_bc_a4_rocket_emission_30Gg_2000climo_1.9x2.5_c20200422.nc', "r+")
    nc_NOx = NetCDFFile(dir+'emissions-cmip6_NO2_rocket_emission_30Gg_2000climo_1.9x2.5_c20200422.nc', "r+")
    nc_H2O = NetCDFFile(dir+'emissions-cmip6_H2O_rocket_emission_30Gg_2000climo_1.9x2.5_c20200422.nc', "r+")

    M_BC = 12.01 # g/mol
    M_NOx = 46 # g/mol
    M_H2O = 18 # g/mol
    AVOG = 6.0221408e23 # 1/mol

    altitude_int = np.array(np.linspace(0, max_alt, 26))
    altitude = (altitude_int[:-1] + altitude_int[1:]) / 2
    nc_num_bc.variables['altitude'][:] = altitude
    nc_num_bc.variables['altitude_int'][:] = altitude_int
    nc_bc.variables['altitude'][:] = altitude
    nc_bc.variables['altitude_int'][:] = altitude_int
    nc_NOx.variables['altitude'][:] = altitude
    nc_NOx.variables['altitude_int'][:] = altitude_int
    nc_H2O.variables['altitude'][:] = altitude
    nc_H2O.variables['altitude_int'][:] = altitude_int

    # Generate time and date arrays
    idx = 0
    yr = 2000
    for mon in range(1, 13):
        nc_num_bc.variables['date'][idx] = yr * 10000 + mon * 100 + 1
        nc_num_bc.variables['time'][idx] = julday(mon, 1, yr)
        nc_bc.variables['date'][idx] = yr * 10000 + mon * 100 + 1
        nc_bc.variables['time'][idx] = julday(mon, 1, yr)
        nc_NOx.variables['date'][idx] = yr * 10000 + mon * 100 + 1
        nc_NOx.variables['time'][idx] = julday(mon, 1, yr)
        nc_H2O.variables['date'][idx] = yr * 10000 + mon * 100 + 1
        nc_H2O.variables['time'][idx] = julday(mon, 1, yr)
        idx += 1

    alt_bin = max_alt*1000/25
    emission_rate = 30000000 # kg
    print(f"Achieving BC from 11 to 70 km of {emission_rate/1e6} Gg per year")
    BC_kg_yearly, NOx_kg_yearly, H2O_kg_yearly = emission_profile(alt_bin, emission_rate)

    # print(f"BC Mass Emitted: {int(np.sum(BC_kg_monthly[0:25]))} kg per month")
    print(f"BC Mass Emitted: {int(np.sum(BC_kg_yearly[0:25]))} kg per year")
    print(f"Strat BC Mass Emitted: {int(np.sum(BC_kg_yearly[4:25]))} kg per year")
    print(f"H2O Mass Emitted: {int(np.sum(H2O_kg_yearly[0:25]))} kg per year")
    print(f"Strat H2O Mass Emitted: {int(np.sum(H2O_kg_yearly[4:25]))} kg per year")
    print(f"NOx Mass Emitted: {int(np.sum(NOx_kg_yearly[0:25]))} kg per year")
    print(f"Strat NOx Mass Emitted: {int(np.sum(NOx_kg_yearly[4:25]))} kg per year")

    # Calculate number of molecules
    BC_molecules_yearly = 1000/M_BC * AVOG * BC_kg_yearly
    NOx_molecules_yearly = 1000/M_NOx * AVOG * NOx_kg_yearly
    H2O_molecules_yearly = 1000/M_H2O * AVOG * H2O_kg_yearly

    # Calculate total area
    columnArea, lat, lon = col_area()

    lat_values = lat
    lon_values = lon
    nlat = lat.shape[0]
    nlon = lon.shape[0]
    nlev = 25
    ntim = 12

    # Create 2D latitude and longitude arrays
    lat_grid, lon_grid = np.meshgrid(lat_values, lon_values, indexing="ij")

    maskArr, emis_index = create_min_distance_mask(lat_grid, lon_grid, launch_lat, launch_lon)
    totalArea=np.sum(columnArea*maskArr)

    # Emission duration (seconds)
    if launch_season == 'winter':
        index = [-1,0,1]
        emission_duration = 60*60*24*90 # s
    elif launch_season == 'summer':
        index = [5,6,7]
        emission_duration = 60*60*24*92 # s
    elif launch_season == 'all_year':
        index = [0,1,2,3,4,5,6,7,8,9,10,11]
        emission_duration = 60*60*24*365 # s



    # Column Emission
    BC_column_molecules = BC_molecules_yearly/totalArea/emission_duration # molecules/cm^2/sec
    NOx_column_molecules = NOx_molecules_yearly/totalArea/emission_duration # molecules/cm^2/sec
    H2O_column_molecules = H2O_molecules_yearly/totalArea/emission_duration # molecules/cm^2/sec

    # Emission rate
    depth = alt_bin*100
    BC_emis_rate = BC_column_molecules/depth # molecules/cm^3/s
    NOx_emis_rate = NOx_column_molecules/depth # molecules/cm^3/s
    H2O_emis_rate = H2O_column_molecules/depth # molecules/cm^3/s

    # Calculate num_bc_a4
    diam_BC = 0.134e-6 #m
    rho_BC = 1700 #kg/m^3
    mass_particle = rho_BC*(np.pi/6)*(diam_BC)**3  # mass per particle (kg/particle)
    emis_num_BC = BC_emis_rate * M_BC/mass_particle  #(particles/cm3/s)(molecules/mole)(g/kg)

    nc_bc.variables['emiss_rocket'][:] = 0
    nc_NOx.variables['emiss_rocket'][:] = 0
    nc_H2O.variables['emiss_rocket'][:] = 0
    nc_num_bc.variables['num_bc_a4_rocket'][:] = 0

    # Transfer data
    for ilev in range(nlev):
        # print(index, ilev)
        nc_bc.variables['emiss_rocket'][index,ilev,emis_index[0],emis_index[1]] = BC_emis_rate[ilev]
        nc_NOx.variables['emiss_rocket'][index,ilev,emis_index[0],emis_index[1]] = NOx_emis_rate[ilev]
        nc_H2O.variables['emiss_rocket'][index,ilev,emis_index[0],emis_index[1]] = H2O_emis_rate[ilev]
        # num_bc_a4
        nc_num_bc.variables['num_bc_a4_rocket'][index,ilev,emis_index[0],emis_index[1]] = emis_num_BC[ilev]
        # print(emis_num_BC[ilev]/BC_emis_rate[ilev])

    # Check emission file
    dir = f'{dir}/'
    nc = NetCDFFile(dir+'emissions-cmip6_bc_a4_rocket_emission_30Gg_2000climo_1.9x2.5_c20200422.nc')
    horizontal_plot(nc)
    vertical_plot(nc)

def format_lat(lat):
    return f"{abs(lat)}{'N' if lat >= 0 else 'S'}"

def format_lon(lon):
    return f"{abs(lon)}{'E' if lon >= 0 else 'W'}"

# ----- Parameters -----
cases = {
    "CC": {"lat": 28.47, "lon": -80.57},
    "RU": {"lat": 70, "lon": 120},
    "SA": {"lat": -55, "lon": -68},
    "EQ": {"lat": 0, "lon": -50},
    "UK": {"lat": 55, "lon": -1.6},
    "AU": {"lat": -29, "lon": 153.5},
}

# Select emission case
selected_case = "CC"
launch_season = "all_year"
max_alt = 70  # km

launch_lat = cases[selected_case]["lat"]
launch_lon = cases[selected_case]["lon"]

dir = f"emission_file/lat_{format_lat(launch_lat)}lon_{format_lon(launch_lon)}_{launch_season}_30Gg/"
# ----------------------

create_emission(launch_lat, launch_lon, launch_season, dir, max_alt=max_alt)