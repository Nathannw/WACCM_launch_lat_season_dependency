from netCDF4 import Dataset
import numpy as np
from scipy.spatial.distance import cdist
import pandas as pd
from datetime import datetime, timedelta
from mpl_toolkits.basemap import Basemap
import matplotlib.pyplot as plt
# Note: Only use for 1.9x2.5 grid
# Falcon 9

def col_area():
    r_earth = 637.8*1e6 # Earth's radius (cm)

    SAearth = 4.0*np.pi*r_earth*r_earth

    # Open the template file in read mode
    templateFile = Dataset('geo_file/coords_1.9x2.5_L70.nc', "r")
    gw_file = Dataset('geo_file/coords_1.9x2.5_L88_c150828.nc', "r")

    # Extract variables
    gw = gw_file.variables['gw'][:]  # [lat | 96]
    lat = templateFile.variables['lat'][:]
    lon = templateFile.variables['lon'][:]

    # Get dimensions
    nLat = len(lat)
    nLon = len(lon)

    # Compute column area
    columnArea = np.tile(gw[:, np.newaxis], (1, nLon))  # Conform to (nLat, nLon)
    columnArea = columnArea * SAearth / nLon / np.sum(gw)  # Area of each grid column in cm²

    # Print results
    # print(f"SAearth={SAearth}; sum(columnArea)={np.sum(columnArea)}") # Should be the similar

    return columnArea, lat, lon

def haversine(lat1, lon1, lat2, lon2):
    """Compute great-circle distance using Haversine formula."""
    R = 6371  # Earth radius in km
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    
    a = np.sin(dlat / 2.0)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0)**2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    
    return R * c  # Distance in km

def create_min_distance_mask(lat_grid, lon_grid, launch_lat, launch_lon):
    """
    Create a mask where only the grid cell closest to the launch site is marked.
    """
    nLat, nLon = lat_grid.shape
    mask = np.zeros((nLat, nLon), dtype=np.float64)  # Initialize mask with zeros

    # Compute distances for all grid cells
    distances = np.array([[haversine(launch_lat, launch_lon, lat_grid[i, j], lon_grid[i, j]) 
                           for j in range(nLon)] for i in range(nLat)])

    # Find the indices of the minimum distance
    min_index = np.unravel_index(np.argmin(distances), distances.shape)

    # Mask only the closest grid cell
    mask[min_index] = 1.0  

    return mask, min_index

# Obtaining altitude emission profile
def emission_profile(alt_bin, emission_rate):
    # Load CSV file
    df = pd.read_csv("Falcon9_trajectory_THAICOM_8.csv")

    burntime_1st_stage = 162 # s
    burntime_2nd_stage = 397 # s
    # Separate into two datasets based on time
    df_1st_stage = df[df["Time, in s"] < burntime_1st_stage].copy()
    df_2nd_stage = df[df["Time, in s"] >= burntime_1st_stage].copy()

    # Define altitude bins (every 5000 meters)
    bin_edges = np.arange(0, df_2nd_stage["Altitude in m"].max() + alt_bin, alt_bin)
    bin_labels = [f"({int(b)/1000}-{int(b+alt_bin)/1000}) km" for b in bin_edges[:-1]]
    # Assign altitude values to bins
    df_1st_stage["Altitude Bin"] = pd.cut(df_1st_stage["Altitude in m"], bins=bin_edges, labels=bin_labels, right=False)

    # Define altitude bins (every 5000 meters)
    bin_labels = [f"({int(b)/1000}-{int(b+alt_bin)/1000}) km" for b in bin_edges[:-1]]
    # Assign altitude values to bins
    df_2nd_stage["Altitude Bin"] = pd.cut(df_2nd_stage["Altitude in m"], bins=bin_edges, labels=bin_labels, right=False)

    # Construct altitude bin
    altitude_bin = np.array([(int(b+alt_bin)/1000 + int(b)/1000)/2 for b in bin_edges[:-1]])

    # Calculate time spent in each altitude bin
    time_spent_1st_stage = df_1st_stage.groupby("Altitude Bin")["Time, in s"].diff().fillna(0).groupby(df_1st_stage["Altitude Bin"]).sum()
    # Calculate time spent in each altitude bin
    time_spent_2nd_stage = df_2nd_stage.groupby("Altitude Bin")["Time, in s"].diff().fillna(0).groupby(df_2nd_stage["Altitude Bin"]).sum()

    # 1st stage propellant mass from Garner 2022
    prop_mass_1st_stage = 412854 # kg
    # 2nd stage propellant mass from Garner 2022
    prop_mass_2nd_stage = 107500 # kg

    # Estimate propellant flow rate
    prop_flow_rate_1st_stage = prop_mass_1st_stage/burntime_1st_stage
    prop_flow_rate_2nd_stage = prop_mass_2nd_stage/burntime_2nd_stage

    # Mass deposition
    mass_deposition_1st_stage = time_spent_1st_stage*prop_flow_rate_1st_stage
    mass_deposition_2nd_stage = time_spent_2nd_stage*prop_flow_rate_2nd_stage

    total_mass_deposited = mass_deposition_1st_stage+mass_deposition_2nd_stage

    # H2O EI
    EI_H2O = 0.374 # kg/kg

    # NOx EI
    EI_NOx = 33/1000*np.exp(-0.26*altitude_bin)

    # BC EI
    EI_BC = 25/1000 * np.maximum(0.04, np.minimum(1, 0.04 * np.exp(0.12 * (altitude_bin - 15))))

    # NOx deposited
    NOx_kg = total_mass_deposited*EI_NOx

    # BC deposited
    BC_kg = total_mass_deposited*EI_BC

    # H2O deposited
    H2O_kg = total_mass_deposited*EI_H2O

    # Stratospheric focus
    BC_kg_strat = BC_kg[4:25] # 12.8 to 70 km
    BC_kg_strat_sum = np.sum(BC_kg_strat)
    # print(BC_kg_strat)
    num_launch = emission_rate/BC_kg_strat_sum
    # num_launch = 17261
    print(f"Number of Launch: {int(num_launch)} per year")

    # Total mass emission monthly
    BC_kg_yearly = BC_kg*num_launch
    NOx_kg_yearly = NOx_kg*num_launch
    H2O_kg_yearly = H2O_kg*num_launch

    return BC_kg_yearly, NOx_kg_yearly, H2O_kg_yearly

# Function to calculate Julian day since 1750-01-01
def julday(month, day, year):
    ref_date = datetime(1750, 1, 1)
    curr_date = datetime(year, month, day)
    return (curr_date - ref_date).days

def horizontal_plot(nc):
    # Extract data
    aggregated_data = np.sum(nc.variables['emiss_rocket'][:, :, :, :], axis=(0,1))  # Sum over altitude
    # aggregated_data = np.transpose(aggregated_data)
    # aggregated_data = np.sum(aggregated_data, axis=2)  # Sum over time
    lats = nc.variables['lat'][:]
    lons = nc.variables['lon'][:]

    lon, lat = np.meshgrid(lons, lats)
    # Get some parameters for the Stereographic Projection
    lon_0 = lons.mean()
    lat_0 = lats.mean()

    m = Basemap(width=5000000,height=3500000,
                resolution='l',projection='stere',\
                lat_ts=40,lat_0=lat_0,lon_0=lon_0)
    xi, yi = m(lon, lat)

    # Create figure
    fig, ax = plt.subplots(figsize=(12, 6))

    # Initialize Basemap
    m = Basemap(projection='cyl', llcrnrlat=lat.min(), urcrnrlat=lat.max(),
                llcrnrlon=lon.min(), urcrnrlon=lon.max(), resolution='l', ax=ax)

    # Draw map features
    m.drawcoastlines()
    m.drawcountries()
    m.drawparallels(np.arange(-90, 91, 20), labels=[1, 0, 0, 0], linewidth=0.2)
    m.drawmeridians(np.arange(-180, 181, 30), labels=[0, 0, 0, 1], linewidth=0.2)

    cs = m.pcolormesh(lon, lat, aggregated_data, latlon=True, shading='auto',
                    vmin=1, vmax=100)


    cbar = m.colorbar(cs, location='bottom', pad="10%")
    cbar.set_label("molecules/cm3/s")

    # Show plot
    plt.show()

def vertical_plot(nc):
    # Load dataset (assuming nc is already loaded)
    data = np.sum(nc.variables['emiss_rocket'][:, :, :, :], axis=(0, 3))  # Sum over time and longitude (time, altitude, lat, lon)
    # data = np.transpose(data)     
    altitudes = nc.variables['altitude'][:]  # Extract altitude levels
    lats = nc.variables['lat'][:]  # Extract latitudes

    # Create figure
    fig, ax = plt.subplots(figsize=(8, 6))

    # Plot altitude vs latitude
    c = ax.pcolormesh(lats, altitudes, data, shading='auto')#, vmin=1, vmax=1000000)
    cbar = plt.colorbar(c, ax=ax)
    cbar.set_label("molecules/cm3/s")

    # Labels and title
    ax.set_xlabel("Latitude")
    ax.set_ylabel("Altitude (m)")

    # Show plot
    plt.show()