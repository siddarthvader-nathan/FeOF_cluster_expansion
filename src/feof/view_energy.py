from pathlib import Path
import argparse
from mchammer import DataContainer
import matplotlib.pyplot as plt


def view_energy(dc):
    """Plot MC energy vs trial step from a DataContainer.
    
    Parameters
    ----------
    dc : mchammer.DataContainer
        DataContainer with MC simulation data.
        
    Returns
    -------
    fig, ax : matplotlib figure and axes objects
    """
    fig, ax = plt.subplots()
    step, energy = dc.get('mctrial', 'potential')
    n_atoms = dc.ensemble_parameters['n_atoms']
    energy_per_site = energy / n_atoms

    ax.set_xlabel("MC trial step", fontsize=14)
    ax.set_ylabel("Energy per site (eV)", fontsize=14)
    ax.set_title("Energy per Site vs. MC Trial Step", fontsize=16)

    ax.plot(step, energy_per_site, linewidth=2)

    ax.tick_params(axis='both', which='major', labelsize=12)
    ax.tick_params(axis='both', which='minor', labelsize=10)
    ax.minorticks_on()

    fig.tight_layout()

    return fig, ax


def main():
    """CLI entry point for viewing MC energy trajectories."""
    parser = argparse.ArgumentParser(
        description="Plot energy vs MC trial step from a DataContainer (.dc) file."
    )
    parser.add_argument(
        "dc_file",
        help="Path to the .dc file (absolute or relative to current directory)."
    )
    args = parser.parse_args()
    
    dc_path = Path(args.dc_file)
    if not dc_path.exists():
        raise FileNotFoundError(f"DataContainer file not found: {dc_path}")
    
    dc = DataContainer.read(str(dc_path))
    fig, ax = view_energy(dc)
    plt.show()


if __name__ == "__main__":
    main()

    
