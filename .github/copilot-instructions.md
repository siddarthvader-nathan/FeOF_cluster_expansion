# FeOF Computational Materials Science Project

## Project Overview

Research codebase for studying rutile Fe(F/O) phase space using cluster expansion models (CEM), Monte Carlo simulations, using DFT calculations. The primary goal is to predict ground state structures and analyze O/F ordering in FeOF.

## Core Dependencies

- **icet**: Cluster expansion framework (ClusterSpace, ClusterExpansion, StructureContainer)
- **mchammer**: Monte Carlo simulations (CanonicalAnnealing, DataContainer)
- **trainstation**: Machine learning for cluster expansion fitting (CrossValidationEstimator)
- **ASE**: Atomic structure manipulation and I/O
- **Standard stack**: numpy, pandas, matplotlib, seaborn

## Architecture & Data Flow

### 1. Structure Generation & Training Data

- **Input**: `FeF2.vasp` primitive cell serves as structural template
- **Organization**: `size_N_unrelaxed/` contains POSCAR files enumerated as `POSCAR_0`, `POSCAR_1`, etc. for different supercell sizes (N = 1-4)
- **Training energies**: `cem_training_energies_nm.txt` contains DFT energies per structure. Energies are in eV, care must be taken to normalize appropriately before model training.
- **Ground states**: `FeOF_gs_relaxed_structs/` and `candidate_structures/` contain relaxed reference structures

### 2. Cluster Expansion Model (CEM) Training

- **Function**: `cem_generator()` in `cem_functions_new.py` is the main CEM training pipeline
- **Key parameters**:
  - `cutoffs`: Pair and triplet interaction cutoff radii (e.g., `[5, 3]`)
  - `fit_method`: ML algorithm (lasso, ardr, bayesian-ridge, elasticnet, etc.)
  - `is_50_50`: Filter for including only stoichiometric FeOF structures for model training (equal O/F concentration)
- **Output**: Trained ClusterExpansion object with effective cluster interactions (ECIs)

### 3. Monte Carlo Simulations

- **Script**: `mccsa_parallel.py` runs canonical ensemble annealing with multiprocessing support
- **Workflow**:
  1. Read primitive cell (`FeF2.vasp`)
  2. Create supercell (e.g., 20x repeat → 240 atoms)
  3. Randomly occupy with target O/F concentration
  4. Anneal from T_start to T_stop (e.g., 8000K → 1500K)
  5. Save ground state structure and trajectory (`.dc` file)
- **Results**: `mc_simulations_size_N_cem/` directories contain output structures and DataContainer files

### 4. Analysis & Visualization

- **Convex hull**: `get_convex_hull()` compares DFT vs CEM mixing energies
- **ECI plots**: `plot_eci()` and `plot_eci_hist()` visualize cluster interactions vs radius/index
- **Energy trajectories**: `view_energy.py` plots MC energy evolution from DataContainer files

## Key Conventions

### File Naming

- Structures: `POSCAR_N` (unrelaxed), `CONTCAR_N` (relaxed), or descriptive names like `canonical_anneal_size_12_8000_to_1500_2_mil_steps_lower_cutoffs.vasp`
- CEM models: `ce_nonmag_size_2_lasso_cutoffs_5_3` (includes max size of supercells, method, cluster radius cutoffs)
- DataContainers: `size_N_TSTART_to_TSTOP_random_canonical_anneal_NSTEPS_steps.dc`

### Structure Organization

- Directories named by supercell size: `size_1_unrelaxed/`, `mc_simulations_size_2_cem/`, etc.
- "size N" refers to N formula units (6 atoms each: 2 Fe + 4 O/F)
- "_correct" suffix indicates symmetrized/verified structures

### Parametric choices and units

- DFT energies normalized per anion site: `energy / (n_atoms - n_Fe)`
- Mixing energies: deviation from O-F endmembers (FeO2, FeF2)
- Effective Cluster Interactions(ECIs), typically in eV, plots may show meV/atom

### Workflow Patterns

1. **CEM training**: If needed, filter out  bad structures with `get_bad_structures()` (checks strain, displacement). In an ionic system like this, lots of structures will be 'bad' because of displacements caused by differences in Fe-O and Fe-F bonding. Hence, we skip this part here.
2. **50/50 structures**: Use `is_50_50=True` for training with stoichiometric FeOF structures exclusively, otherwise train on full concentration range.
3. **Cross-validation**: Compare fit methods using `get_ecis_with_all_fitting_methods()` before selecting.
4. **MC canonical simulated annealing**: Higher initial temps (8000K) for global search, lower (3000K) for refinement.

## Common Commands and outputs

### Train CEM (in notebook)

```python
from src.feof.cem_functions_new import cem_generator
import ase.io
"""Template starting structure for enumeration and CEM"""
struct = ase.io.read('./FeF2.vasp')
cluster_list = [[['Fe']]*2,[['O','F']]*4]
"""Cutoffs: List of cutoffs for 2,3,4....n body clusters in $\AA$"""
cutoffs = [5,3] #Here, this is a cluster list with cutoffs of 5Å and 3Å for pair and triplet clusters
opt, ce = cem_generator(struct, cluster_list, cutoffs, 'lasso', max_size=2, is_50_50=False)
ce.write('./results/trained_cem_models/ce_model_name')
```

============== CrossValidationEstimator ==============
seed                           : 42
fit_method                     : lasso
standardize                    : True
n_target_values                : 238
n_parameters                   : 11
n_nonzero_parameters           : 10
parameters_norm                : 10.35892
target_values_std              : 0.4460453
rmse_train                     : 0.01210798
R2_train                       : 0.9992595
AIC                            : -2079.832
BIC                            : -2045.109
alpha_optimal                  : 0.001625965
validation_method              : k-fold
n_splits                       : 10
rmse_train_final               : 0.01213803
rmse_validation                : 0.01307205
R2_validation                  : 0.998818
shuffle                        : True

![1770408945120](image/copilot-instructions/1770408945120.png)

### When Making Changes to CEM training

- **Adding structures**: Place in appropriate `size_N_unrelaxed/` dir, update `cem_training_energies_nm.txt` with index-matched energies
- **Modifying cutoffs**: Shorter = fewer clusters (faster, less accurate); adjust both pair and triplet cutoffs
- **MC convergence**: Check energy trajectory flatness; increase `n_steps` if not equilibrating at T_stop
- **Fit method selection**: Lasso for sparsity, ARDR for automatic relevance determination; compare via `rmse_validation`

### Plot Convex Hull

```python
from src.feof.cem_functions_new import get_convex_hull

fig, ax, hull = get_convex_hull('./cem_training_energies_nm.txt', './results/trained_cem_models/ce_model_name') 
#Replace ce_model_name with the appropriate trained cem binary
plt.show()
```

I personally did not use this function here as I studied only FeOF. If looking for ternary ordered phases in the entire phase space, this function is extremely useful.

### Run MC Simulation (equipped to parallelize using multiprocessing)

```bash
# Edit mccsa_parallel.py parameters, then:
nohup python src/feof/mccsa_parallel.py &
```

### Visualize energy as a function of timestep to make sure MC simulations are equilibrated

```bash
feof-view-energy ./results/mc_simulations_size_3_cem/size_20_8000_to_1500_random_canonical_anneal_20_mil_steps.dc
```

### Clean up POSCAR files to view them on VESTA

The .vasp files of the ground state low energy structure generated from the MC simulated anneal are usually unformatted with improper spacing. This makes the VESTA visualization impossible as some atoms may not show up while VESTA reads the .vasp file. To aid this issue, use the `clean_poscar` command:

```bash
feof-clean-poscar ./results/mc_simulations_size_3_cem/<unformatted .vasp file> ./results/mc_simulations_size_3_cem/<formatted .vasp file>
```

## Helper files and python scripts

- **mccsa_parallel.py**: Python script to run MC simulations in parallel
- **clean_poscar.py**: Python script to rearrange atoms of ground state POSCAR files generated from canonical simulated annealing. Helps with visualizing on VESTA
- **DataContainer files**: Contain full MC trajectory; read with `dc = DataContainer.read(filename)`, extract data via `dc.get('mctrial', 'potential')`

## Jupyter Notebooks

- `cem_nm.ipynb`: Main CEM training notebook with cross-validation, ECI analysis, convex hull plots
- `grid.ipynb`: EMD comparison of GRID representations of candidate FeOF DFT-relaxed structures (refer to the documentation in the notebook as well as Zhang et.al, *Digital Discovery* **2**, 81 (2023))

## Citation

All work in this repository was used for our working paper: [https://arxiv.org/abs/2512.12179]()
