"""Compare retrograde-orbit and ISS altitude versus downrange distance."""

from argparse import ArgumentParser
from pathlib import Path
import pickle
import sys

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import numpy as np

from analysis.figure_style import apply_journal_style


EARTH_RADIUS_M = 6_375_253.0
EARTH_ROTATION_RATE_RAD_S = 7.292115e-5


apply_journal_style()


def load_trajectory(path):
    with open(path, "rb") as stream:
        saved_data = pickle.load(stream)

    if not isinstance(saved_data, (tuple, list)) or len(saved_data) < 2:
        raise ValueError("Trajectory pickle must contain at least times and states")

    times = np.asarray(saved_data[0], dtype=float)
    states = np.asarray(saved_data[1], dtype=float)
    if states.ndim != 2 or states.shape[1] < 3 or len(times) != len(states):
        raise ValueError("Expected times and an N x 3-or-larger state array")
    return times, states[:, :3]


def truncate_at_altitude(times, positions_eci_m, max_altitude_km):
    altitude_km = (
        np.linalg.norm(positions_eci_m, axis=1) - EARTH_RADIUS_M
    ) / 1000.0
    crossing_indices = np.flatnonzero(altitude_km > max_altitude_km)

    if crossing_indices.size == 0:
        keep = altitude_km <= max_altitude_km
        return times[keep], positions_eci_m[keep], altitude_km[keep]

    crossing = crossing_indices[0]
    if crossing == 0:
        raise ValueError("Trajectory starts above the requested maximum altitude")

    lower = crossing - 1
    altitude_span = altitude_km[crossing] - altitude_km[lower]
    fraction = (max_altitude_km - altitude_km[lower]) / altitude_span
    crossing_time = times[lower] + fraction * (times[crossing] - times[lower])
    crossing_position = positions_eci_m[lower] + fraction * (
        positions_eci_m[crossing] - positions_eci_m[lower]
    )

    truncated_times = np.concatenate((times[:crossing], [crossing_time]))
    truncated_positions = np.vstack((positions_eci_m[:crossing], crossing_position))
    truncated_altitudes = np.concatenate((altitude_km[:crossing], [max_altitude_km]))
    return truncated_times, truncated_positions, truncated_altitudes


def eci_to_ecef(positions_eci_m, times):
    angle = EARTH_ROTATION_RATE_RAD_S * times
    cosine = np.cos(angle)
    sine = np.sin(angle)
    x_eci, y_eci, z_eci = positions_eci_m.T
    return np.column_stack(
        (
            cosine * x_eci + sine * y_eci,
            -sine * x_eci + cosine * y_eci,
            z_eci,
        )
    )


def downrange_distance_km(positions_ecef_m):
    unit_positions = positions_ecef_m / np.linalg.norm(
        positions_ecef_m, axis=1, keepdims=True
    )
    launch_direction = unit_positions[0]
    central_angle = np.arctan2(
        np.linalg.norm(np.cross(launch_direction, unit_positions), axis=1),
        np.clip(unit_positions @ launch_direction, -1.0, 1.0),
    )
    return EARTH_RADIUS_M * central_angle / 1000.0


def plot_altitude_vs_downrange(trajectories, output_path):
    fig, ax = plt.subplots(figsize=(6, 4.5))
    line_styles = ("-", "--")
    for (label, downrange_km, altitude_km), linestyle in zip(
        trajectories, line_styles
    ):
        ax.plot(
            downrange_km,
            altitude_km,
            color="black",
            linestyle=linestyle,
            linewidth=1.6,
            label=label,
        )
    ax.set_xlabel("Downrange distance (km)", fontsize=14)
    ax.set_ylabel("Altitude (km)", fontsize=14)
    max_downrange_km = max(values[1].max() for values in trajectories)
    ax.set_xlim(0, np.ceil(max_downrange_km / 10.0) * 10.0)
    ax.set_ylim(0, 70)
    ax.xaxis.set_major_locator(MaxNLocator(nbins=6))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=7, integer=True))
    ax.tick_params(labelsize=12, direction="out", length=3.5, width=0.8)
    ax.grid(True, linestyle="--", linewidth=0.45, color="0.82")
    ax.set_axisbelow(True)
    ax.legend(frameon=False, fontsize=10, loc="upper left")
    for spine in ax.spines.values():
        spine.set_linewidth(0.8)
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)


def parse_args():
    project_dir = Path(__file__).resolve().parent.parent
    parser = ArgumentParser(description=__doc__)
    parser.add_argument(
        "--retrograde-trajectory",
        type=Path,
        default=project_dir
        / "launch_trajectory"
        / "data_28.47_97.6.pkl",
    )
    parser.add_argument(
        "--iss-trajectory",
        type=Path,
        default=project_dir
        / "launch_trajectory"
        / "data_28.47_51.6.pkl",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=project_dir
        / "fig"
        / "emission_profile"
        / "altitude_vs_downrange_retrograde_and_iss.pdf",
    )
    parser.add_argument("--max-altitude-km", type=float, default=70.0)
    return parser.parse_args()


def main():
    args = parse_args()
    trajectories = []
    summaries = []
    for label, path in (
        ("To Retrograde Orbit (700 km, 97.6$^\circ$)", args.retrograde_trajectory),
        ("To ISS (400 km, 51.6$^\circ$)", args.iss_trajectory),
    ):
        times, positions_eci_m = load_trajectory(path)
        times, positions_eci_m, altitude_km = truncate_at_altitude(
            times, positions_eci_m, args.max_altitude_km
        )
        positions_ecef_m = eci_to_ecef(positions_eci_m, times)
        downrange_km = downrange_distance_km(positions_ecef_m)
        trajectories.append((label, downrange_km, altitude_km))
        summaries.append((label, altitude_km[-1], downrange_km[-1], times[-1]))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    plot_altitude_vs_downrange(trajectories, args.output)
    print(f"Created {args.output}")
    for label, altitude_km, downrange_km, time_s in summaries:
        print(
            f"{label} at {altitude_km:.1f} km altitude: "
            f"downrange = {downrange_km:.2f} km, time = {time_s:.2f} s"
        )


if __name__ == "__main__":
    main()
