from multiprocessing import Pool
import ase.io
from ase.build import make_supercell
from icet import ClusterSpace, ClusterExpansion
from icet.tools.structure_generation import occupy_structure_randomly
import numpy as np
from mchammer.calculators import ClusterExpansionCalculator
from mchammer.ensembles import CanonicalAnnealing

# ============================================================================
# PARALLEL MC SIMULATIONS FOR FeOF CLUSTER EXPANSION
# ============================================================================
# This script runs canonical annealing simulations across multiple supercell
# sizes in parallel using multiprocessing. Each process performs independent
# MC simulations with temperature annealing from 8000K to 0K.
#
# TO RUN IN BACKGROUND WITH LOGGING:
#   nohup python mccsa_parallel.py > mccsa_parallel.log 2>&1 &
#
# This command:
#   - nohup: Allows process to continue after terminal closes
#   - > mccsa_parallel.log: Redirects stdout to log file
#   - 2>&1: Redirects stderr to same log file
#   - &: Runs in background
#
# Monitor progress with: tail -f mccsa_parallel.log
# ============================================================================

# Read the initial structure
primitive_cell = ase.io.read('../../FeF2.vasp')

# Decide on supercell sizes for the simulations
size = [2, 3, 4, 5, 6]  # Example sizes

# Define the ClusterExpansion object (assuming it's similar to the provided code). Change the pretrained cluster expansion binary file as needed.
ce_site_energies = ClusterExpansion.read('../../results/trained_ce_models/ce_nonmag_size_3_lasso_cutoffs_7_4')

# Helper function to run parallel canonical annealing simulations
def run_canonical_annealing(size):
    # Build the supercell based on the size
    structure = primitive_cell.repeat(size)
    
    # Make anion ordering random to start with
    cs = ce_site_energies.get_cluster_space_copy()
    occupy_structure_randomly(structure, cs, target_concentrations={'A': {'O': 0.5, 'F': 0.5}})
    
    # Set up the calculator
    calculator = ClusterExpansionCalculator(structure, ce_site_energies)
    
    # Set up and run the Canonical Ensemble. Adjust simulation temperatures, cooling function and number of steps
    camc = CanonicalAnnealing(structure, calculator, T_start=8000.0, T_stop=0.0, n_steps=20000000,
                              dc_filename='size_{}_8000_to_0_random_canonical_anneal_20_mil_steps_size_3_cem.dc'.format(size))
    camc.run()
    ase.io.write('canonical_anneal_size_{}_8000_to_0_20_mil_steps_size_3_cem.vasp'.format(size), camc.estimated_ground_state, direct=True)


if __name__ == "__main__":
    pool = Pool(processes=5)
    results = pool.map_async(run_canonical_annealing, size)
    results.get()
    pool.close()
