"""Compare webcast and simulated Falcon 9 emission profiles."""

from argparse import ArgumentParser
from pathlib import Path
import pickle

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from analysis.figure_style import apply_journal_style


apply_journal_style()


EARTH_RADIUS_M = 6_375_253.0
FIRST_STAGE_BURN_S = 162.0
SECOND_STAGE_BURN_S = 397.0
FIRST_STAGE_PROPELLANT_KG = 412_854.0
SECOND_STAGE_PROPELLANT_KG = 107_500.0
PROFILE_XLIMS = {"BC": (0, 350), "H2O": (0, 40000), "NOx": (0, 2500)}
PROFILE_XTICKS = {
    "BC": np.arange(0, 351, 50),
    "H2O": np.arange(0, 40_001, 10_000),
    "NOx": np.arange(0, 2_501, 500),
}

def distribute_interval_mass(altitude_pairs_m, interval_mass_kg, bin_edges_m):
    """Distribute each interval's mass across every altitude bin it crosses."""
    binned_mass = np.zeros(len(bin_edges_m) - 1, dtype=float)

    for (z0, z1), mass in zip(altitude_pairs_m, interval_mass_kg):
        if not np.isfinite(z0 + z1 + mass) or mass <= 0:
            continue

        if np.isclose(z0, z1):
            bin_index = np.searchsorted(bin_edges_m, z0, side="right") - 1
            if 0 <= bin_index < len(binned_mass):
                binned_mass[bin_index] += mass
            continue

        low, high = sorted((z0, z1))
        crossed_edges = bin_edges_m[(bin_edges_m > low) & (bin_edges_m < high)]
        fractions = [0.0, 1.0]
        fractions.extend((crossed_edges - z0) / (z1 - z0))
        fractions = np.sort(np.asarray(fractions))

        for fraction0, fraction1 in zip(fractions[:-1], fractions[1:]):
            fraction = fraction1 - fraction0
            midpoint = z0 + 0.5 * (fraction0 + fraction1) * (z1 - z0)
            bin_index = np.searchsorted(bin_edges_m, midpoint, side="right") - 1
            if 0 <= bin_index < len(binned_mass):
                binned_mass[bin_index] += mass * fraction

    return binned_mass


def webcast_propellant_profile(csv_path, bin_edges_m):
    data = pd.read_csv(csv_path)
    first_stage = data[data["Time, in s"] < FIRST_STAGE_BURN_S].copy()
    second_stage = data[data["Time, in s"] >= FIRST_STAGE_BURN_S].copy()

    first_stage["Altitude Bin"] = pd.cut(
        first_stage["Altitude in m"], bins=bin_edges_m, right=False
    )
    second_stage["Altitude Bin"] = pd.cut(
        second_stage["Altitude in m"], bins=bin_edges_m, right=False
    )

    first_stage_time = (
        first_stage.groupby("Altitude Bin", observed=False)["Time, in s"]
        .diff()
        .fillna(0)
        .groupby(first_stage["Altitude Bin"], observed=False)
        .sum()
    )
    second_stage_time = (
        second_stage.groupby("Altitude Bin", observed=False)["Time, in s"]
        .diff()
        .fillna(0)
        .groupby(second_stage["Altitude Bin"], observed=False)
        .sum()
    )

    first_stage_mass = first_stage_time * FIRST_STAGE_PROPELLANT_KG / FIRST_STAGE_BURN_S
    second_stage_mass = second_stage_time * SECOND_STAGE_PROPELLANT_KG / SECOND_STAGE_BURN_S
    return (first_stage_mass + second_stage_mass).to_numpy()


def simulated_propellant_profile(pickle_path, bin_edges_m):
    with open(pickle_path, "rb") as stream:
        saved_data = pickle.load(stream)

    if not isinstance(saved_data, (tuple, list)) or len(saved_data) < 2:
        raise ValueError("Simulation pickle must contain at least times and states")
    times, states = saved_data[:2]

    altitude_m = np.linalg.norm(states[:, :3], axis=1) - EARTH_RADIUS_M
    mass_kg = states[:, -1]
    if np.nanmax(mass_kg) < 1_000:
        mass_kg = np.exp(mass_kg)

    interval_mass_kg = np.maximum(-np.diff(mass_kg), 0.0)
    time_step_s = np.diff(times)
    mass_loss_rate = np.divide(
        interval_mass_kg,
        time_step_s,
        out=np.zeros_like(interval_mass_kg),
        where=time_step_s > 0,
    )

    positive_rates = mass_loss_rate[mass_loss_rate > 0]
    if positive_rates.size:
        staging_threshold = 3.0 * np.percentile(positive_rates, 95)
        interval_mass_kg = np.where(
            mass_loss_rate <= staging_threshold, interval_mass_kg, 0.0
        )

    altitude_pairs = np.column_stack((altitude_m[:-1], altitude_m[1:]))
    return distribute_interval_mass(altitude_pairs, interval_mass_kg, bin_edges_m)


def emission_profiles(propellant_kg, altitude_centers_km):
    ei_h2o = np.full_like(altitude_centers_km, 0.374, dtype=float)
    ei_nox = 33.0 / 1000.0 * np.exp(-0.26 * altitude_centers_km)
    ei_bc = 25.0 / 1000.0 * np.maximum(
        0.04,
        np.minimum(1.0, 0.04 * np.exp(0.12 * (altitude_centers_km - 15.0))),
    )
    return {
        "NOx": propellant_kg * ei_nox,
        "BC": propellant_kg * ei_bc,
        "H2O": propellant_kg * ei_h2o,
    }


def plot_comparison_panels(
    altitude_centers_km,
    webcast_profiles,
    retrograde_profiles,
    iss_profiles,
    bin_width_km,
    output_path,
):
    fig, axes = plt.subplots(1, 3, figsize=(10, 5), sharey=True)
    profile_series = [
        (
            "Publicly Available Launch Telemetry Records",
            webcast_profiles,
            {"color": "black", "edgecolor": "black"},
        ),
        (
            "Traj. Optimization to Retrograde Orbit (700 km, 97.6$^\circ$)",
            retrograde_profiles,
            {"color": "gray", "edgecolor": "black"},
        ),
        (
            "Traj. Optimization to ISS (400 km, 51.6$^\circ$)",
            iss_profiles,
            {"color": "white", "edgecolor": "black", "hatch": "xxxx"},
        ),
    ]
    bar_height = 0.8 * bin_width_km / len(profile_series)
    offsets = (
        np.arange(len(profile_series)) - (len(profile_series) - 1) / 2
    ) * bar_height
    panel_specs = (
        ("BC", "(a)", "BC mass deposited\nper launch (kg)"),
        ("H2O", "(b)", "H$_2$O mass deposited\nper launch (kg)"),
        ("NOx", "(c)", "NO$_x$ mass deposited\nper launch (kg)"),
    )

    for ax, (species, panel_label, xlabel) in zip(axes, panel_specs):
        for offset, (label, profiles, style) in zip(offsets, profile_series):
            ax.barh(
                altitude_centers_km + offset,
                profiles[species],
                height=bar_height,
                linewidth=0.6,
                label=label,
                **style,
            )
        ax.set_title(panel_label, loc="left", fontsize=12, fontweight="bold", pad=5)
        ax.set_xlabel(xlabel, fontsize=14)
        ax.set_ylim(0, altitude_centers_km[-1] + bin_width_km / 2)
        ax.set_xlim(PROFILE_XLIMS[species])
        ax.set_xticks(PROFILE_XTICKS[species])
        ax.set_yticks(np.arange(0, 71, 10))
        ax.tick_params(labelsize=11, direction="out", length=3.5, width=0.8, pad=2)
        ax.grid(True, linestyle="--", linewidth=0.45, color="0.82")
        ax.set_axisbelow(True)
        for spine in ax.spines.values():
            spine.set_linewidth(0.8)

    axes[0].set_ylabel("Altitude (km)", fontsize=14)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        fontsize=12,
        loc="upper center",
        ncol=3,
        frameon=False,
        handlelength=1.6,
        columnspacing=1.5,
        bbox_to_anchor=(0.5, 0.995),
    )
    fig.tight_layout(rect=(0, 0, 1, 0.95), w_pad=1.2)
    fig.savefig(output_path, bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)


def parse_args():
    project_dir = Path(__file__).resolve().parent
    parser = ArgumentParser(description=__doc__)
    parser.add_argument(
        "--webcast",
        type=Path,
        default=project_dir / "Falcon9_trajectory_THAICOM_8.csv",
    )
    parser.add_argument(
        "--retrograde-simulation",
        type=Path,
        default=project_dir
        / "data_fixedGT1_BTL0_PO0_POadjusted1_throttle0_28.47_97.6.pkl",
    )
    parser.add_argument(
        "--iss-simulation",
        type=Path,
        default=project_dir
        / "data_fixedGT1_BTL0_PO0_POadjusted1_throttle0_28.47_51.6.pkl",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=project_dir / "fig" / "emission_profile",
    )
    parser.add_argument("--max-altitude-km", type=float, default=70.0)
    parser.add_argument("--bins", type=int, default=25)
    parser.add_argument(
        "--output-suffix",
        default=None,
        help="Suffix added to each output PDF filename.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    bin_edges_m = np.linspace(0.0, args.max_altitude_km * 1000.0, args.bins + 1)
    altitude_centers_km = 0.5 * (bin_edges_m[:-1] + bin_edges_m[1:]) / 1000.0
    bin_width_km = args.max_altitude_km / args.bins

    webcast_propellant_kg = webcast_propellant_profile(args.webcast, bin_edges_m)
    webcast = emission_profiles(webcast_propellant_kg, altitude_centers_km)
    retrograde = emission_profiles(
        simulated_propellant_profile(args.retrograde_simulation, bin_edges_m),
        altitude_centers_km,
    )
    iss = emission_profiles(
        simulated_propellant_profile(args.iss_simulation, bin_edges_m),
        altitude_centers_km,
    )
    requested_suffix = args.output_suffix or "webcast_vs_retrograde_and_iss"
    output_suffix = "".join(
        character if character.isalnum() or character in "_-" else "_"
        for character in requested_suffix
    )
    output_path = args.output_dir / f"altitude_emission_profiles_{output_suffix}.pdf"
    plot_comparison_panels(
        altitude_centers_km,
        webcast,
        retrograde,
        iss,
        bin_width_km,
        output_path,
    )
    print(f"Created {output_path}")


if __name__ == "__main__":
    main()
