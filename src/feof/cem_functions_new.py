"""FUNCTIONS FOR TASKS

Module with helper functions for building and analyzing cluster expansions.
"""
from icet import ClusterExpansion,ClusterSpace,StructureContainer
from trainstation import CrossValidationEstimator
import matplotlib.pyplot as plt 
from icet.tools import ConvexHull
import ase.io
import pandas as pd
import numpy as np
import itertools
import seaborn as sns
import os
import re
import matplotlib.patches as mpatches


def cem_generator(struct,cluster_list,cutoffs,fit_method,max_size,is_50_50=False):
    # This function generates a Cluster Expansion Model (CEM) based on the provided structures and parameters.
    # Parameters:
    # - struct: The initial structure for the CEM.
    # - cluster_list: List of clusters to consider in the model.
    # - cutoffs: Cutoff distances for cluster interactions.
    # - fit_method: The method used for fitting the model (e.g., 'lasso', 'ardr', 'bayesian-ridge').  
    # - max_size: Maximum size of structures to consider.
    # - is_50_50: Boolean indicating if only 50/50 O/F structures should be included.
    # Returns:
    # - opt: Optimized parameters for the CEM.
    # - ce_site_energies: The resulting Cluster Expansion object with site energies.
   
    energy_cols = ['STRUCTURE', 'ENERGY']
    df_energy = pd.read_csv('./cem_training_energies_nm.txt',sep ='\s+',usecols=energy_cols)
    cs = ClusterSpace(structure=struct, cutoffs=cutoffs, chemical_symbols=list(itertools.chain(*cluster_list)))
    sc_site_energies = StructureContainer(cluster_space=cs)
    
    """Getting bad structures: change indices as per number of structures in training set. 
       Disabled for this system, may be more applicable to metal alloy like stuff."""
    #excluded_list = get_bad_structures(0,248)
    excluded_list = []

    """Training CEM"""
   
    struct_list =[] 
    size_dict  ={}
    number_of_structures = 0

    if is_50_50:
        struct_idx = 0
        for i in range(1, max_size + 1):
            size_dir = f'./size_{i}_unrelaxed'
            for filename in os.listdir(size_dir):
                if 'POSCAR' in filename:
                    structure = ase.io.read(f"{size_dir}/{filename}")
                    if structure.get_chemical_symbols().count('F') == structure.get_chemical_symbols().count('O'):
                        n_sites = len(structure) - structure.get_chemical_symbols().count('Fe')
                        sc_site_energies.add_structure(structure, properties = {'dft_energies_per_fu': df_energy['ENERGY'][struct_idx]/n_sites})
                        #Remember to normalize energies by formula unit. Here, formula unit is for the active cluster sites, i.e anionic 6f Wyckoff sites
                    struct_idx += 1 
    else:
        for i in range(1,max_size+1):
            key = f"size_{i}"
            size_dict[key] = len(os.listdir('./size_{}_unrelaxed'.format(i)))
            number_of_structures += size_dict[key]
        
        for j in range(0,number_of_structures):
            struct_list.append(ase.io.read('./unrelaxed_structures_all/POSCAR_{}'.format(j)))
        
        for index,structure in enumerate(struct_list):    
            n_sites = len(structure) - structure.get_chemical_symbols().count('Fe')
            if index not in excluded_list:
                sc_site_energies.add_structure(structure, properties = {'dft_energies_per_fu': df_energy['ENERGY'][index]/n_sites}) 
                #Remember to normalize energies by formula unit. Here, formula unit is for the active cluster sites, i.e anionic 6f Wyckoff sites

    opt = get_cve(fit_method, sc_site_energies)
    ce_site_energies = get_cem_new(opt,cs)
    #ce_site_energies.write('ce_model_anion_site_energies_gs_cutoffs_{}_{}'.format(cutoffs[0],cutoffs[1]))
    print(sc_site_energies)
    return opt,ce_site_energies


def get_cve(fit_method, sc):
    """Cross-validation estimator for cluster expansion fitting.
    
    Trains and validates a cluster expansion model using the specified fitting method.
    
    Parameters:
    - fit_method: The fitting algorithm (e.g., 'lasso', 'ardr', 'bayesian-ridge'). 
     Refer to https://trainstation.materialsmodeling.org for more details on fitting
    - sc: StructureContainer with DFT energies per formula unit
    
    Returns:
    - cve: Trained CrossValidationEstimator object with validation metrics
    """
    cve = CrossValidationEstimator(fit_data=sc.get_fit_data(key='dft_energies_per_fu'), fit_method=fit_method)
    cve.validate()
    cve.train()

    return cve

def get_row(cve, fit_method):
    """Extract fitting metrics from CrossValidationEstimator for model comparison.
    
    Collects validation and training statistics to compare different fitting methods
    in cluster expansion model selection.
    
    Parameters:
    - cve: CrossValidationEstimator object with trained model and metrics
    - fit_method: String name of the fitting algorithm used
    
    Returns:
    - row: Dictionary containing RMSE, BIC, parameter counts, and fit method name
    """
    row = dict()
    row['fit_method'] = fit_method
    row['rmse_validation'] = cve.rmse_validation
    row['rmse_train'] = cve.rmse_train
    row['BIC'] = cve.model.BIC
    row['n_parameters'] = cve.n_parameters
    row['n_nonzero_parameters'] = cve.n_nonzero_parameters
    
    return row

def get_cem_new(cve,cs):
    def get_cem_new(cve, cs):
        """
        Create a new ClusterExpansion instance from an existing cluster expansion and cluster space.
        Args:
            cve: A cluster expansion object containing parameters and metadata/summary information.
            cs: A ClusterSpace object defining the cluster space for the new expansion.
        Returns:
            ClusterExpansion: A new ClusterExpansion instance initialized with the provided cluster space,
                             parameters from cve, and metadata from cve's summary.
        """
    
    ce = ClusterExpansion(cluster_space=cs, parameters=cve.parameters, metadata=cve.summary)
    
    return ce
    
def plot_eci(ce):
    """Plot effective cluster interactions (ECIs) as a function of cluster radius.
    
    Visualizes pair and triplet cluster interactions from a trained cluster expansion model,
    with each cluster type distinguished by color.
    
    Parameters:
    - ce: ClusterExpansion object containing fitted ECIs and cluster information
    
    Returns:
    - fig: matplotlib figure object
    - ax: matplotlib axes object with scatter plot of ECIs vs radius
    """
    df_ce = ce.to_dataframe()
    radius_pair = np.array(df_ce['radius'][df_ce['order']==2])
    radius_trip = np.array(df_ce['radius'][df_ce['order']==3])
    eci_pair = np.array(df_ce['eci'][df_ce['order']==2])
    eci_trip = np.array(df_ce['eci'][df_ce['order']==3])
    
    fig, ax = plt.subplots()
    ax.set_xlabel("Radius($\AA$)")
    ax.set_ylabel("ECI(eV)")
    ax.scatter(radius_pair, eci_pair, c='darkorchid', label='Pairs')
    ax.scatter(radius_trip, eci_trip, c='forestgreen', label='Triplets')
    ax.legend(loc='best')
    
    return fig, ax


def plot_eci_hist(ce):
    """Plot effective cluster interactions (ECIs) as a bar chart.
    
    Visualizes pair and triplet cluster interactions from a trained cluster expansion model
    as a histogram, with each cluster type distinguished by color.
    
    Parameters:
    - ce: ClusterExpansion object containing fitted ECIs and cluster information
    
    Returns:
    - None (displays plot)
    """
    df_ce = ce.to_dataframe()
    eci_pair = np.array(df_ce['eci'][df_ce['order'] == 2])
    eci_trip = np.array(df_ce['eci'][df_ce['order'] == 3])
    
    # Combine pairs and triplets for plotting
    eci_values = np.concatenate((eci_pair, eci_trip))
    colors = ['darkorchid'] * len(eci_pair) + ['forestgreen'] * len(eci_trip)

    # Create bar plot for all ECIs
    plt.figure(figsize=(10, 6))
    plt.bar(range(len(eci_values)), eci_values, color=colors, edgecolor='black')
    plt.xlabel('Index of clusters', fontsize=16)
    plt.ylabel('ECI (meV/atom)', fontsize=16)
    plt.xticks(fontsize=13)
    plt.yticks(fontsize=13)
    plt.grid(True, linestyle='--', linewidth=0.5)
    
    # Create custom legend entries
    pair_patch = mpatches.Patch(color='darkorchid', label='Pairs')
    triplet_patch = mpatches.Patch(color='forestgreen', label='Triplets')
    
    # Add the legend to the plot
    plt.legend(handles=[pair_patch, triplet_patch], loc='best', fontsize=15)
    
    plt.tight_layout()
    plt.show()

def get_ecis_with_all_fitting_methods(sc_refined):
    """Compare cluster expansion fits across multiple fitting methods.
    
    Trains cluster expansion models using different fitting algorithms and generates
    ECI plots for each method to support model selection.
    
    Parameters:
    - sc_refined: StructureContainer with DFT energies per formula unit
    
    Returns:
    - df_fits: DataFrame containing RMSE, BIC, and parameter counts for each fit method
    """
    fit_results = []
    fit_methods = ['ardr', 'bayesian-ridge', 'elasticnet', 'lasso', 'least-squares', 'omp', 'rfe', 'ridge', 'split-bregman']

    for fit_method in fit_methods:
        # Train cross-validation estimator for current fit method
        cve = get_cve(fit_method, sc_refined)
        # Extract fitting metrics (RMSE, BIC, parameter counts)
        fit_results.append(get_row(cve))
        # Generate cluster expansion from trained parameters
        ce = get_cem_new(cve)
        # Plot and save ECI scatter plot for visual comparison
        fig, ax = plot_eci(ce)
        fig.savefig(f'{fit_method}_ECI.pdf', dpi=500)

    # Compile results into comparison dataframe
    df_fits = pd.DataFrame(fit_results)
    return df_fits


def get_convex_hull(energies, cem):
    """Plot convex hull comparing DFT vs cluster expansion energies.
    
    Constructs and visualizes the convex hull of mixing energies for structures
    across the FeO2-FeF2 composition range, comparing DFT reference energies with
    cluster expansion predictions.
    
    Parameters:
    - energies: Path to CSV file containing DFT energies (columns: STRUCTURE, ENERGY)
    - cem: Path to saved ClusterExpansion model file
    
    Returns:
    - fig: matplotlib figure object with convex hull comparison plot
    - ax: matplotlib axes object with scatter and hull lines
    - hull: Dictionary containing ConvexHull objects and energy arrays:
        * 'dft_hull': ConvexHull for DFT calculated mixing energies
        * 'cluster_hull': ConvexHull for CEM predicted mixing energies
        * 'dft_energies_per_site': Raw DFT energies (normalized per anion site)
        * 'cluster_energies_per_site': Raw CEM energies (normalized per anion site)
        * 'mixing_energies_dft': DFT energies relative to O/F endmembers (normalized per anion site)
        * 'mixing_energies_cluster': CEM energies relative to O/F endmembers (normalized per anion site)
    """
    
    # Load all unrelaxed structures and corresponding energies
    num_of_structures = len(os.listdir('./unrelaxed_structures_all'))
    struct_list = []
    for idx in range(num_of_structures):
        struct_list.append(ase.io.read("./unrelaxed_structures_all/POSCAR_{}".format(idx)))
   
    # Read saved cluster expansion model
    ce = ClusterExpansion.read(cem)
    df = pd.read_csv(energies, sep='\s+', usecols=['STRUCTURE', 'ENERGY'])
    
    # Initialize data dictionary to store composition, DFT and CEM energies
    data = {'concentration': [], 'dft_energy': [], 'cluster_energy': []}
    sizes = []

    # Compute F concentration, normalize energies per anion site, and predict with CEM
    for idx, structure in enumerate(struct_list):
        conc = structure.get_chemical_symbols().count('F') / (structure.get_chemical_symbols().count('O') + structure.get_chemical_symbols().count('F'))
        data['concentration'].append(conc)
        n_sites = len(structure) - structure.get_chemical_symbols().count('Fe')
        data['dft_energy'].append(df['ENERGY'][idx] / n_sites) 
        sizes.append(len(structure) / 6)   
        data['cluster_energy'].append(ce.predict(structure))

    # Extract arrays for convex hull construction
    raw_energies_dft = np.array(data['dft_energy'])
    raw_energies_cluster = np.array(data['cluster_energy'])
    concentrations = np.array(data['concentration'])
    hull = dict()

    # Reference endmember energies (FeO2 at idx=0, FeF2 at idx=5 from size_1)
    FeO2_energy_dft = raw_energies_dft[0]
    FeF2_energy_dft = raw_energies_dft[5]
    FeO2_energy_cluster = raw_energies_cluster[0]
    FeF2_energy_cluster = raw_energies_cluster[5]

    # Compute mixing energies as deviation from linear interpolation between endmembers
    mixing_energies_dft = raw_energies_dft - ((1 - concentrations) * FeO2_energy_dft + (concentrations) * FeF2_energy_dft)
    hull1 = ConvexHull(concentrations, mixing_energies_dft)
    hull['dft_hull'] = hull1
    hull['dft_energies_per_site'] = raw_energies_dft
    hull['mixing_energies_dft'] = mixing_energies_dft
    
    # Compute CEM mixing energies and convex hull
    mixing_energies_cluster = raw_energies_cluster - ((1 - concentrations) * FeO2_energy_cluster + (concentrations) * FeF2_energy_cluster)
    hull2 = ConvexHull(concentrations, mixing_energies_cluster)        
    hull['cluster_hull'] = hull2
    hull['cluster_energies_per_site'] = raw_energies_cluster
    hull['mixing_energies_cluster'] = mixing_energies_cluster
    
    # Create comparison plot
    fig, ax = plt.subplots(figsize=(8, 6))
    fig.text(0.01, 0.05, "FeO2")
    fig.text(0.95, 0.05, "FeF2")
    ax.set_xlabel(r'F content')
    ax.set_ylabel(r'Mixing Energy (meV/atom)')
    ax.set_xlim([0, 1])
    ax.set_ylim([-400, 15])
    
    # Plot DFT structures and hull
    ax.scatter(concentrations, 1e3 * mixing_energies_dft, marker='x', label="DFT structures")
    ax.plot(hull1.concentrations, 1e3 * hull1.energies, '-o', color='green', label="DFT hull")
    
    # Plot CEM structures and hull
    ax.scatter(concentrations, 1e3 * mixing_energies_cluster, marker='x', label='CEM structures')
    ax.plot(hull2.concentrations, 1e3 * hull2.energies, '-o', color='purple', label="CEM hull")
    
    ax.legend(loc='best')
    return fig, ax, hull




"""def get_structures_with_equal_F_O(workdir):
    structures_list = []  # Initialize an empty list to store dictionaries for each structure
    for idx, filename in enumerate(os.listdir(workdir)) :  # Use enumerate for index and filename
        if "POSCAR" in filename:  # Check if the filename contains "POSCAR"
            structure = ase.io.read(f"{workdir}/{filename}")  # Read the structure file
            if structure.get_chemical_symbols().count('F') == structure.get_chemical_symbols().count('O'):
                struct_info = {  # Create a dictionary for each structure's information
                    'index': idx,
                    'structure': structure,
                    'size': len(structure)/6
                }
                structures_list.append(struct_info)  # Append the dictionary to the list
    
    return structures_list"""


"""def get_50_50_structures(workdir):
    structures_list = []  # Initialize an empty list to store dictionaries for each structure
    for idx, filename in enumerate(os.listdir(workdir)):  # Use enumerate for index and filename
        print(idx,filename)
        if "POSCAR" in filename:  # Check if the filename contains "POSCAR"
            structure = ase.io.read(f"{workdir}/{filename}")  # Read the structure file
            num_F = structure.get_chemical_symbols().count('F')
            num_O = structure.get_chemical_symbols().count('O')
            if num_F == num_O:  # Check if the number of F and O atoms are equal
                struct_info = {  # Create a dictionary for each structure's information
                    'index': idx,
                    'filename': filename,  # Include filename in the structure info
                    'num_atoms': len(structure),  # Store the total number of atoms instead of size
                    'num_F': num_F,  # Store the number of F atoms
                    'num_O': num_O  # Store the number of O atoms
                }
                structures_list.append(struct_info)  # Append the dictionary
    return structures_list

def get_50_50_structures(workdir):
    structures_list = []  # Initialize an empty list to store dictionaries for each structure
    for filename in os.listdir(workdir):  # Loop through filenames in the directory
        if "POSCAR" in filename:  # Check if the filename contains "POSCAR"
            structure = ase.io.read(f"{workdir}/{filename}")  # Read the structure file
            num_F = structure.get_chemical_symbols().count('F')
            num_O = structure.get_chemical_symbols().count('O')
            if num_F == num_O:  # Check if the number of F and O atoms are equal
                # Extract the number from the filename using regular expression
                match = re.search(r'POSCAR_(\d+)', filename)
                if match:  # If a match is found
                    index = int(match.group(1))  # Convert the matched group (number) to an integer
                    struct_info = {  # Create a dictionary for each structure's information
                        'index': index,  # Use the extracted number as the index
                        'filename': filename,
                        'num_atoms': len(structure),
                        'num_F': num_F,
                        'num_O': num_O
                    }
                    structures_list.append(struct_info)  # Append the dictionary to the list
    
    return structures_list

def get_cve_displt(fit_method,sc):
    
    cve = CrossValidationEstimator(fit_data=sc.get_fit_data(key='average_displacement'), fit_method=fit_method)
    cve.validate()
    cve.train()

    return cve
def get_bad_structures(start_struct_idx,end_struct_idx,excluded_list=[]):
    
    warnings_list = [['high_volumetric_strain'], ['high_anisotropic_strain'], ['large_maximum_relaxation_distance'], ['large_average_relaxation_distance']]
    for index in range(start_struct_idx,end_struct_idx):
        reference = ase.io.read("./unrelaxed_structures_all/POSCAR_{}".format(index))
        structure = ase.io.read("./relaxed_structures_revised/CONTCAR_{}".format(index))    
        mapped_structure, info = map_structure_to_reference(structure, reference)
        #print('Maximum displacement: {:.3f} Angstrom'.format(info['drmax']))
        #print('Average displacement: {:.3f} Angstrom'.format(info['dravg']))
    
        if info['warnings'] in warnings_list:
            #print(index, "should be excluded")
            excluded_list.append(index)
    return excluded_list
def displt_expansion(struct,cluster_list,cutoffs,fit_method,max_size):
    struct_list =[]
    size_dict  ={}
    number_of_structures = 0

    
    for i in range(1,max_size+1):
        key = f"size_{i}"
        size_dict[key] = len(os.listdir('./size_{}_unrelaxed'.format(i)))
        number_of_structures += size_dict[key]
    
    displacements = np.zeros(number_of_structures)
    for index in range(number_of_structures):
        reference = ase.io.read("./unrelaxed_structures_all/POSCAR_{}".format(index))
        structure = ase.io.read("./relaxed_structures_revised/CONTCAR_{}".format(index))    
        info = map_structure_to_reference(structure, reference)[1]
        displacements[index] = info['dravg']
    #print(displacements)
    
    cs = ClusterSpace(structure=struct, cutoffs=cutoffs, chemical_symbols=list(itertools.chain(*cluster_list)))
    sc_displacements = StructureContainer(cluster_space=cs)
    
    for j in range(0,number_of_structures):
        struct_list.append(ase.io.read('./unrelaxed_structures_all/POSCAR_{}'.format(j)))
    
    for index,structure in enumerate(struct_list):
            sc_displacements.add_structure(structure, properties = {'average_displacement': displacements[index]})
    
    opt = get_cve_displt(fit_method, sc_displacements)
    ce_site_energies = get_cem_new(opt,cs)
    return opt,ce_site_energies"""