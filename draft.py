from pathlib import Path
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    from netCDF4 import Dataset as NetCDFFile
except ImportError as exc:
    raise SystemExit(
        "This script requires netCDF4. Install it in this Python environment with "
        "`pip install netCDF4` or run the script from the project's NetCDF-enabled environment."
    ) from exc

from analysis.emission_funcs import horizontal_plot, vertical_plot


BASE_DIR = Path(__file__).resolve().parent
NC_PATH = BASE_DIR / "AUG_13_ID_all_year.cam.h0.1999-12.nc"
OUTPUT_DIR = BASE_DIR / "fig" / "draft_bc"
BC_VARIABLES = ("bc_a1", "bc_a4")


def main():
    os.chdir(BASE_DIR)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    with NetCDFFile(NC_PATH) as nc:
        for variable_name in BC_VARIABLES:
            if variable_name not in nc.variables:
                raise KeyError(f"{variable_name} was not found in {NC_PATH}")

            units = getattr(nc.variables[variable_name], "units", "kg/kg")

            horizontal_path = OUTPUT_DIR / f"{variable_name}_horizontal.png"
            fig, _ = horizontal_plot(
                nc,
                variable_name=variable_name,
                title=f"{variable_name} horizontal distribution",
                save_path=horizontal_path,
                cbar_label=units,
                show=False,
            )
            plt.close(fig)

            vertical_path = OUTPUT_DIR / f"{variable_name}_vertical.png"
            fig, _ = vertical_plot(
                nc,
                variable_name=variable_name,
                title=f"{variable_name} vertical distribution",
                save_path=vertical_path,
                cbar_label=units,
                show=False,
            )
            plt.close(fig)

            print(f"Saved {horizontal_path}")
            print(f"Saved {vertical_path}")


if __name__ == "__main__":
    main()
