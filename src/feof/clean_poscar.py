import numpy as np
from ase import io, Atoms
import sys

def generate_vasp_poscar(input_file, output_file):
    """
    Generate a VASP POSCAR file from an input file.

    This function reads atomic structure data from an input file, extracts 
    the lattice vectors, atomic symbols, and atomic positions, and writes 
    them to an output file in the VASP POSCAR format.

    Parameters
    ----------
    input_file : str
        Path to the input file containing atomic structure data.
        The file should be in a format readable by ASE (e.g., VASP, CIF).
        
    output_file : str
        Path to the output file where the VASP POSCAR will be written.
        
    Raises
    ------
    IOError
        If the input file cannot be read or the output file cannot be written.
        
    ValueError
        If the input file does not contain valid atomic structure data.
        
    Examples
    --------
    Generate a POSCAR file from an existing VASP file:
    
    >>> generate_vasp_poscar('input.vasp', 'output_POSCAR.vasp')
    
    Notes
    -----
    - The POSCAR format includes a comment line, a scaling factor, 
      lattice vectors, atomic symbols and their counts, and atomic 
      positions in Cartesian coordinates.
    - The function assumes the input file contains valid atomic structure 
      data that ASE can parse.
    
    """
    # Read the atoms from the input file
    atoms = io.read(input_file)
    
    # Extract data
    cell = atoms.get_cell()               # Lattice vectors
    symbols = np.array(atoms.get_chemical_symbols())
    positions = atoms.get_positions()
    
    # Count unique symbols and their occurrences
    unique_symbols, counts = np.unique(symbols, return_counts=True)
    
    # Write the VASP POSCAR file
    with open(output_file, 'w') as f:
        f.write('Generated POSCAR\n')     # Comment line
        f.write('1.0\n')                  # Scaling factor
        
        # Write lattice vectors
        for vector in cell:
            f.write('  '.join(map(str, vector)) + '\n')
        
        # Write unique symbols
        f.write('  '.join(unique_symbols) + '\n')
        
        # Write counts of each symbol
        f.write('  '.join(map(str, counts)) + '\n')
        
        f.write('Cartesian\n')            # Positions type
        
        # Write atomic positions
        for symbol in unique_symbols:
            for pos in positions[symbols == symbol]:
                f.write('  '.join(map(str, pos)) + '\n')

def main():
    """CLI entry point for cleaning POSCAR files."""
    import argparse
    parser = argparse.ArgumentParser(
        description="Clean and reformat VASP POSCAR files for VESTA visualization."
    )
    parser.add_argument("input_file", help="Path to input structure file.")
    parser.add_argument("output_file", help="Path to output POSCAR file.")
    args = parser.parse_args()
    
    try:
        generate_vasp_poscar(args.input_file, args.output_file)
        print(f"VASP POSCAR file generated successfully: {args.output_file}")
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
