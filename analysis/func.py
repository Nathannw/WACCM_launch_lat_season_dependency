import numpy as np
import ast
import scipy.stats as stats

def read_file(filename):
    read_data = []

    with open(filename, "r") as file:
        for line in file:
            read_data.append(ast.literal_eval(line.strip()))

    lats = read_data[0]; seasons = read_data[1]; location_list = read_data[2]; values = read_data[3]
    target_lats = [-55, -29, 0, 29, 55, 70]
    # Initialize arrays with np.nan (keeps data as floats)
    year_round = np.full(len(target_lats), np.nan)
    summer = np.full(len(target_lats), np.nan)
    winter = np.full(len(target_lats), np.nan)

    # Zip pairs the items from the 3 lists together line-by-line
    for lat, season, loc_list, val in zip(lats, seasons, location_list, values):
        if lat in target_lats:
            
            # Find which index this latitude belongs to
            idx = target_lats.index(lat)
            
            # Route to the correct array
            if season == 'all_year' and loc_list != 'CC_BC_only':
                year_round[idx] = val
            elif season == 'summer':
                summer[idx] = val
            elif season == 'winter':
                winter[idx] = val
    return year_round, summer, winter

def Neff_calculation(rho1_case):
    # sample sizes
    N = 15

    Neff_case = N * (1 - rho1_case) / (1 + rho1_case)
    Neff = np.minimum(Neff_case, N)
    # effective degrees of freedom
    dof_eff = Neff - 1
    return Neff, dof_eff

def Neff_calculation_p_val(rho1_case, t_stat):
    # sample sizes
    N = 15

    # rho = 0.5 * (rho_control + rho1_case)
    Neff_case = N * (1 - rho1_case) / (1 + rho1_case)
    Neff = np.minimum(Neff_case, N)
    # effective degrees of freedom
    dof_eff = Neff - 1
    p_val = 2 * stats.t.sf(abs(t_stat), dof_eff)
    return Neff, dof_eff, p_val


def find_trop_func(nc, i):
    TROP_P = nc["TROP_P"][:,i] / 100
    LEV = nc["lev"]

    LEV = np.array(LEV)

    # ensure monotonic increasing
    if LEV[0] > LEV[-1]:
        LEV = LEV[::-1]

    idx_upper = np.searchsorted(LEV, TROP_P)
    idx_lower = idx_upper - 1

    idx_lower = np.clip(idx_lower, 0, len(LEV)-2)
    idx_upper = np.clip(idx_upper, 1, len(LEV)-1)

    p1 = LEV[idx_lower]
    p2 = LEV[idx_upper]

    ratio = (TROP_P - p1) / (p2 - p1)
    # Validate
    # [266.1497802734375] [53] [54] [0.8110851682482056] [232.82861896] [273.91081676] [266.1497802734375]
    # print(TROP_P, idx_lower, idx_upper, ratio, LEV[idx_lower], LEV[idx_upper], LEV[idx_lower] * (1 - ratio) + LEV[idx_upper] * ratio)

    return idx_lower, idx_upper, ratio


def pressure_level_weights(nc):
    lev = np.asarray(nc["lev"][:], dtype=float)
    if "ilev" in nc.variables and len(nc["ilev"][:]) == len(lev) + 1:
        edges = np.asarray(nc["ilev"][:], dtype=float)
    else:
        mid_edges = 0.5 * (lev[:-1] + lev[1:])
        first_edge = lev[0] - (mid_edges[0] - lev[0])
        last_edge = lev[-1] + (lev[-1] - mid_edges[-1])
        edges = np.concatenate(([first_edge], mid_edges, [last_edge]))

    return np.abs(np.diff(edges))


def pressure_weighted_mean(values, weights):
    values = np.ma.masked_invalid(np.ma.array(values))
    weights = np.ma.masked_invalid(np.ma.array(weights))
    if values.ndim > weights.ndim:
        weights = np.reshape(weights, (1,) * (values.ndim - weights.ndim) + weights.shape)
    weights = np.broadcast_to(weights, values.shape)
    weights = np.ma.array(
        weights,
        mask=np.ma.getmaskarray(values) | np.ma.getmaskarray(weights)
    )
    return np.ma.sum(values * weights) / np.ma.sum(weights)
