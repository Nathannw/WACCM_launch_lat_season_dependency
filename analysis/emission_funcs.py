try:
    from netCDF4 import Dataset
except ImportError:
    Dataset = None
    from scipy.io import netcdf_file
else:
    netcdf_file = None
import numpy as np
try:
    import pandas as pd
except ImportError:
    pd = None
from datetime import datetime, timedelta
try:
    from mpl_toolkits.basemap import Basemap
except ImportError:
    Basemap = None
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
# import seaborn as sns
import pickle
from pathlib import Path
from analysis.figure_style import apply_journal_style

apply_journal_style()
# Note: Only use for 1.9x2.5 grid
# Falcon 9

def col_area(coords_dir="geo_file"):
    """Return grid-column areas, latitudes, and longitudes.

    ``coords_dir`` defaults to the historical project-relative location, while
    callers in reusable analysis modules can pass an absolute directory.
    """
    r_earth = 637.8*1e6 # Earth's radius (cm)

    SAearth = 4.0*np.pi*r_earth*r_earth

    coords_dir = Path(coords_dir)
    template_path = coords_dir / "coords_1.9x2.5_L70.nc"
    weights_path = coords_dir / "coords_1.9x2.5_L88_c150828.nc"
    if Dataset is not None:
        template_context = Dataset(template_path, "r")
        weights_context = Dataset(weights_path, "r")
    else:
        template_context = netcdf_file(template_path, mode="r", mmap=False)
        weights_context = netcdf_file(weights_path, mode="r", mmap=False)

    with template_context as template_file, weights_context as weights_file:
        # Copy values so they remain valid after scipy.io files are closed.
        gw = np.array(weights_file.variables['gw'][:], dtype=float, copy=True)
        lat = np.array(template_file.variables['lat'][:], dtype=float, copy=True)
        lon = np.array(template_file.variables['lon'][:], dtype=float, copy=True)

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
def emission_profile(alt_bin, emission_rate, profile_xlims=None):
    """Calculate emissions and plot profiles with optional per-species x-limits.

    Example: profile_xlims={"BC": (0, 250), "H2O": (0, 40000), "NOx": (0, 2500)}
    """
    if pd is None:
        raise ImportError("emission_profile requires pandas")

    # if profile_xlims is None:
    #     profile_xlims = {}
    profile_xlims = {"BC": (0, 250), "H2O": (0, 40000), "NOx": (0, 2500)}
    valid_species = {"BC", "H2O", "NOx"}
    unknown_species = set(profile_xlims) - valid_species
    if unknown_species:
        raise ValueError(f"Unknown profile_xlims keys: {sorted(unknown_species)}")
    for species, limits in profile_xlims.items():
        try:
            xmin, xmax = limits
        except (TypeError, ValueError) as error:
            raise ValueError(
                f"profile_xlims['{species}'] must be (xmin, xmax)"
            ) from error
        if xmin >= xmax:
            raise ValueError(f"profile_xlims['{species}'] must be (xmin, xmax)")

    # Load CSV file
    trajectory_path = (
        Path(__file__).resolve().parent.parent
        / "launch_trajectory"
        / "Falcon9_trajectory_THAICOM_8.csv"
    )
    df = pd.read_csv(trajectory_path)

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
    
    # --- 1. Time vs altitude ---
    with open('max_payload_convex_trajectory.pkl', "rb") as f:
        times, X_ref_final, U_ref_final, m_BC, mask, EI_BC = pickle.load(f)
    R_E = 6375253   
    df[["Time, in s"]].assign(Altitude_km=df["Altitude in m"])
    df["Time_s"] = pd.to_numeric(df["Time, in s"], errors="coerce")
    df["Altitude in km"] = pd.to_numeric(df["Altitude in m"]/1000, errors="coerce")

    df = df.dropna(subset=["Altitude in km"])

    plt.figure()
    Altitude = np.linalg.norm(X_ref_final[:, 0:3], axis=1) - R_E
    plt.plot(df["Time_s"], df["Altitude in km"], label = "SpaceX Webcast Data", linestyle = ':')
    plt.plot(times, Altitude/1e3, label = "Max PL Simulated Traj")
    plt.xlabel('Time (s)')
    plt.ylabel('Altitude (km)')
    plt.ylim([0,100])
    plt.xlim([0,200])
    plt.legend()
    plt.savefig("fig/emission_profile/time_vs_altitude_webcast.pdf")
    plt.show()

    # --- 2. Altitude emission profiles ---
    fig, axes = plt.subplots(1, 3, figsize=(10, 3), sharey=True)
    panel_specs = (
        ("BC", "(a)", BC_kg, "BC mass deposited\nper launch (kg)"),
        ("H2O", "(b)", H2O_kg, "H$_2$O mass deposited\nper launch (kg)"),
        ("NOx", "(c)", NOx_kg, "NO$_x$ mass deposited\nper launch (kg)"),
    )
    visible_altitudes = (altitude_bin >= 0) & (altitude_bin < 70)

    for ax, (species, panel_label, profile, xlabel) in zip(axes, panel_specs):
        ax.barh(
            altitude_bin[visible_altitudes],
            np.asarray(profile)[visible_altitudes],
            height=alt_bin / 1000 * 0.9,
            color="black",
            edgecolor="black",
        )
        ax.set_title(panel_label, loc="left", fontsize=12, fontweight="bold", pad=5)
        ax.set_xlabel(xlabel, fontsize=14)
        ax.set_ylim(0, 70)
        if species in profile_xlims:
            ax.set_xlim(profile_xlims[species])
        ax.tick_params(labelsize=12)
        ax.xaxis.set_major_locator(MaxNLocator(nbins=5))
        ax.grid(True, which="both", linestyle="--", linewidth=0.5, alpha=0.7)
        ax.set_axisbelow(True)

    axes[0].set_ylabel("Altitude (km)", fontsize=14)
    fig.tight_layout(w_pad=1.5)
    fig.savefig("fig/emission_profile/altitude_emission_profiles.pdf", bbox_inches="tight")
    # plt.show()
    plt.close(fig)

    return BC_kg_yearly, NOx_kg_yearly, H2O_kg_yearly

# Function to calculate Julian day since 1750-01-01
def julday(month, day, year):
    ref_date = datetime(1750, 1, 1)
    curr_date = datetime(year, month, day)
    return (curr_date - ref_date).days

def horizontal_plot(
    nc,
    variable_name="emiss_rocket",
    title=None,
    save_path=None,
    vmin=None,
    vmax=None,
    cbar_label=None,
    show=False,
    cmap="viridis",
):
    # Extract data
    variable = nc.variables[variable_name]
    data = variable[:]
    if data.ndim == 4:
        aggregated_data = np.sum(data, axis=(0, 1))  # Sum over time and altitude
    elif data.ndim == 3:
        aggregated_data = np.sum(data, axis=0)  # Sum over time
    elif data.ndim == 2:
        aggregated_data = data
    else:
        raise ValueError(f"{variable_name} must be 2D, 3D, or 4D; got shape {data.shape}")
    aggregated_data = np.ma.filled(aggregated_data, np.nan)
    # aggregated_data = np.transpose(aggregated_data)
    # aggregated_data = np.sum(aggregated_data, axis=2)  # Sum over time
    lats = nc.variables['lat'][:]
    lons = nc.variables['lon'][:]

    lon, lat = np.meshgrid(lons, lats)

    # Create figure
    fig, ax = plt.subplots(figsize=(12, 6))

    if vmin is None and variable_name == "emiss_rocket":
        vmin = 1
    if vmax is None and variable_name == "emiss_rocket":
        vmax = 100
    if cbar_label is None:
        cbar_label = getattr(variable, "units", "molecules/cm3/s")

    if Basemap is not None:
        # Initialize Basemap
        m = Basemap(projection='cyl', llcrnrlat=lat.min(), urcrnrlat=lat.max(),
                    llcrnrlon=lon.min(), urcrnrlon=lon.max(), resolution='l', ax=ax)

        # Draw map features
        m.drawcoastlines()
        m.drawcountries()
        m.drawparallels(np.arange(-90, 91, 20), labels=[1, 0, 0, 0], linewidth=0.2)
        m.drawmeridians(np.arange(-180, 181, 30), labels=[0, 0, 0, 1], linewidth=0.2)

        cs = m.pcolormesh(lon, lat, aggregated_data, latlon=True, shading='auto',
                          vmin=vmin, vmax=vmax, cmap=cmap)
        cbar = m.colorbar(cs, location='bottom', pad="10%")
    else:
        cs = ax.pcolormesh(lons, lats, aggregated_data, shading='auto',
                           vmin=vmin, vmax=vmax, cmap=cmap)
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")
        cbar = fig.colorbar(cs, ax=ax, location="bottom", pad=0.15)

    cbar.set_label(cbar_label)
    if title:
        ax.set_title(title)
    if save_path:
        plt.savefig(save_path, bbox_inches="tight")

    # Show plot
    if show:
        plt.show()
    return fig, ax

def vertical_plot(
    nc,
    variable_name="emiss_rocket",
    title=None,
    save_path=None,
    vmin=None,
    vmax=None,
    cbar_label=None,
    show=False,
    cmap="viridis",
):
    # Load dataset (assuming nc is already loaded)
    variable = nc.variables[variable_name]
    raw_data = variable[:]
    if raw_data.ndim == 4:
        data = np.sum(raw_data, axis=(0, 3))  # Sum over time and longitude (time, altitude, lat, lon)
    elif raw_data.ndim == 3:
        data = np.sum(raw_data, axis=0)
    elif raw_data.ndim == 2:
        data = raw_data
    else:
        raise ValueError(f"{variable_name} must be 2D, 3D, or 4D; got shape {raw_data.shape}")
    data = np.ma.filled(data, np.nan)
    # data = np.transpose(data)     
    if 'altitude' in nc.variables:
        altitudes = nc.variables['altitude'][:]  # Extract altitude levels
        altitude_label = f"Altitude ({getattr(nc.variables['altitude'], 'units', 'km')})"
    elif 'Z3' in nc.variables:
        altitudes = np.mean(nc.variables['Z3'][:], axis=(0, 3)) / 1000
        altitude_label = "Altitude (km)"
    elif 'lev' in nc.variables:
        altitudes = nc.variables['lev'][:]
        altitude_label = f"Pressure ({getattr(nc.variables['lev'], 'units', 'hPa')})"
    else:
        raise ValueError("Expected an 'altitude', 'Z3', or 'lev' variable for vertical plotting")
    lats = nc.variables['lat'][:]  # Extract latitudes
    if cbar_label is None:
        cbar_label = getattr(variable, "units", "molecules/cm3/s")

    # Create figure
    fig, ax = plt.subplots(figsize=(8, 6))

    # Plot altitude vs latitude
    if np.ndim(altitudes) == 2:
        lat_grid = np.broadcast_to(lats[np.newaxis, :], data.shape)
        c = ax.pcolormesh(lat_grid, altitudes, data, shading='auto',
                          vmin=vmin, vmax=vmax, cmap=cmap)
    else:
        c = ax.pcolormesh(lats, altitudes, data, shading='auto',
                          vmin=vmin, vmax=vmax, cmap=cmap)
    cbar = plt.colorbar(c, ax=ax)
    cbar.set_label(cbar_label)

    # Labels and title
    ax.set_xlabel("Latitude")
    ax.set_ylabel(altitude_label)
    if title:
        ax.set_title(title)
    if 'lev' in nc.variables and 'altitude' not in nc.variables and 'Z3' not in nc.variables:
        ax.invert_yaxis()
    if save_path:
        plt.savefig(save_path, bbox_inches="tight")

    # Show plot
    if show:
        plt.show()
    return fig, ax
