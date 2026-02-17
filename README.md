# FeOF Cluster Expansion Model

Research codebase for studying Fe(O/F) phase space using cluster expansion models and Monte Carlo simulations.

## Quick Setup

1. **Install the package in development mode** (from repo root):

   ```bash
   pip install -e .
   ```
2. **Install dependencies** (if not done via the above):

   ```bash
   pip install -r requirements.txt
   ```
3. **You're done!** Now you can:

   - Import from anywhere: `from feof.cem_functions_new import cem_generator`
   - Run scripts from anywhere:
     ```bash
     python -m feof.view_energy ./results/mc_simulations_size_3_cem/size_20_8000_to_1500_random_canonical_anneal_20_mil_steps.dc
     ```
   - Or use the CLI shortcut (if installed):
     ```bash
     feof-view-energy ./results/mc_simulations_size_3_cem/size_20_8000_to_1500_random_canonical_anneal_20_mil_steps.dc
     ```

## Directory Structure

```
FeOF/
├── src/feof/                      # Main package code
│   ├── cem_functions_new.py       # CEM training pipeline
│   ├── mccsa_parallel.py          # MC simulations with multiprocessing
│   ├── view_energy.py             # Visualization tool
│   └── clean_poscar.py            # POSCAR formatting utility
├── cem_nm.ipynb                   # CEM training notebook
├── grid.ipynb                     # GRID analysis notebook
├── size_*_unrelaxed/              # Enumerated structures by size, unrelaxed
├── results/                       # MC outputs and trained models
└── FeOF_gs_relaxed_structs/       # FeOF ground state relaxed candidate structures
└── FeOF_gs_unrelaxed_structs/     # FeOF ground state unrelaxed candidate structures
				     These are generated from structure enumeration
└── INCAR_DFT_relaxation	   # INCAR file used for DFT relaxations for CEM targets
└── cem_training_energies_nm.txt   # DFT calculated energies (in eV) of all structures
				     These are fed into the CEM training as targets
```

## Quick Start

### Train a CEM model (in Jupyter):

```python
from feof.cem_functions_new import cem_generator
import ase.io

struct = ase.io.read('./FeF2.vasp')
cluster_list = [[['Fe']]*2, [['O','F']]*4]
cutoffs = [5, 3]  # pair and triplet cutoffs in Å

opt, ce = cem_generator(struct, cluster_list, cutoffs, 'lasso', max_size=2, is_50_50=False)
ce.write('./results/trained_cem_models/my_model')
```

### Run MC simulations:

```bash
# Edit parameters in src/feof/mccsa_parallel.py, then:
 nohup python -m feof.mccsa_parallel &
```

### View MC energy trajectory:

```bash
feof-view-energy ./results/mc_simulations_size_3_cem/my_simulation.dc
```

### Clean up structures for VESTA:

```bash
python -m feof.clean_poscar ./results/unformatted.vasp ./results/formatted.vasp
```

## Dependencies

- **icet**: Cluster expansion framework
- **mchammer**: Monte Carlo simulations
- **trainstation**: ML fitting for CEMs
- **ase**: Atomic structure tools
- Standard: numpy, pandas, matplotlib, seaborn

See `requirements.txt` for versions.

## Project Details

See [.github/copilot-instructions.md](./.github/copilot-instructions.md) for detailed documentation of the workflow, conventions, and parameters.

**Paper**: [https://arxiv.org/abs/2512.12179](https://arxiv.org/abs/2512.12179)
