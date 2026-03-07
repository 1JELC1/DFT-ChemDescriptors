"""
Code to calculate global and local properties for a set of molecules sharing structural similarity
"""

import itertools
from pathlib import Path
import os
import glob
import shutil
import numpy as np
import warnings
warnings.filterwarnings(
    'ignore',
    message='DataFrame is highly fragmented'
)
import re
import pandas as pd
import FragmentFinder as Ff
import subprocess
import threading
import psutil
import queue
import tempfile
import csv
from collections import defaultdict
import other_desc as desc
from scipy.optimize import fsolve
import concurrent.futures
from rdkit import Chem
from rdkit.Chem import rdDetermineBonds


''' Section 1: Function declarations '''

# Global lists to store file paths
neutral_file_list = []
anion_file_list = []
cation_file_list = []

def run_multiwfn(inputs, work_dir):
    """Execute Multiwfn with given input file in specified directory"""
    with open(inputs, 'r') as f:
        subprocess.run(['Multiwfn'], stdin=f, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, cwd=work_dir)

def run_multiwfn_output(inputs, work_dir, output_file):
    """Execute Multiwfn and save output to specified file"""
    with open(inputs, 'r') as f:
        with open(output_file, 'w') as out:
            subprocess.run(['Multiwfn'], stdin=f, stdout=out, stderr=subprocess.DEVNULL, cwd=work_dir)

def save_csv(df, filename, work_path):
    """Save DataFrame to CSV file with error handling and sorting"""
    while True:
        try:
            # Check for duplicate columns
            duplicate_columns = df.columns[df.columns.duplicated()]
            if not duplicate_columns.empty:
                print(f"Warning: Found duplicate columns: {list(duplicate_columns)}")
                print("Renaming duplicate columns.")

            # Replace empty strings with NaN
            df.replace("", pd.NA, inplace=True)
            # Drop columns where all values are NaN
            df.dropna(axis=1, how='all', inplace=True)

            # Rename duplicate columns
            for col in duplicate_columns:
                mask = df.columns.duplicated(keep='first')
                new_cols = df.columns.where(~mask, df.columns + '_duplicate')
                df.columns = new_cols

            # Ensure 'Molecule' column exists
            if 'Molecule' not in df.columns:
                raise ValueError("Column 'Molecule' not found in DataFrame.")

            # Sort DataFrame by 'Molecule'
            sorted_df = df.sort_values(by=['Molecule'])

            # Try to convert 'Molecule' to numeric if possible
            try:
                sorted_df['Molecule'] = pd.to_numeric(sorted_df['Molecule'])
                sorted_df = sorted_df.sort_values(by=['Molecule'])
            except ValueError:
                pass

            # Create the full path
            full_path = Path(work_path) / filename

            # Save sorted DataFrame to the output folder
            sorted_df.to_csv(full_path, index=False)
            
            # Verify save by reading from the output folder
            df_check = pd.read_csv(full_path)
            df_check = df_check.sort_values(by=['Molecule'])
            df_check.to_csv(full_path, index=False)

            print(f"--> File {filename} generated successfully")
            break
        except Exception as e:
            resp = input(f"Error saving file '{filename}'.\nError: {str(e)}"
                         f"\n(Try again = ENTER, Cancel = c)")
            if resp.lower() == 'c':
                break

def merge_dataframes(global_df, local_df):
    """Merge global and local DataFrames using 'Molecule' as key"""
    try:
        # Verify both dataframes have 'Molecule' column
        if 'Molecule' not in global_df.columns or 'Molecule' not in local_df.columns:
            raise ValueError("Both dataframes must have a 'Molecule' column.")

        # Sort both dataframes by 'Molecule'
        global_df = global_df.sort_values('Molecule').reset_index(drop=True)
        local_df = local_df.sort_values('Molecule').reset_index(drop=True)

        # Merge dataframes using 'Molecule' as key
        combined_df = pd.merge(global_df, local_df, on='Molecule', how='outer')

        # Remove duplicate columns
        duplicate_columns = combined_df.columns[combined_df.columns.duplicated()]
        for col in duplicate_columns:
            if col != 'Molecule':
                combined_df = combined_df.loc[:, ~combined_df.columns.duplicated(keep='first')]

        return combined_df

    except Exception as e:
        print(f"Error merging dataframes: {str(e)}")
        return None

def extract_AIM_charges(output_file, output):
    """Extract AIM charges from Multiwfn output file"""
    charges = []
    with open(output_file, 'r') as f:
        content = f.read()
        # Find section containing atomic charges
        charges_section = re.search(r'The atomic charges after normalization and atomic volumes:(.*?)Integrating basins',
                                   content, re.DOTALL)

        if charges_section:
            lines = charges_section.group(1).strip().split('\n')
            for line in lines:
                # Regex to capture atom info, charge and volume
                match = re.match(r'\s*(\d+\s*\([A-Za-z]{1,2}\s*\))\s+Charge:\s+([-]?\d+\.\d+)\s+Volume:', line)

                if match:
                    atom_info, charge = match.groups()
                    # Format output with atom and charge
                    charges.append(f"{atom_info:<10} Charge: {charge:<12}")

    # Write output to file
    if charges:
        with open(output, 'w') as f:
            f.write('\n'.join(charges))
    else:
        print(f"ERROR: No charges found in {output_file}")

def calculate_charges():
    """Worker function for charge calculation queue"""
    while True:
        index = task_queue.get()
        if index is None:
            break
        try:
            charge_calculation(index)
        finally:
            task_queue.task_done()

def get_molecule_name(file_path, suffix):
    """Extract molecule name from file path by removing suffix and extension"""
    base_name = Path(file_path).stem
    if suffix and base_name.endswith(suffix):
        return base_name[:-len(suffix)]
    return base_name

def check_files():
    """
    Check and categorize files by charge state, supporting .fchk, .wfn, .wfx.
    Identifies missing states (N, N+1, N-1) based on user selection.
    """
    # Ensure source folder exists
    if not fchk_folder.exists():
        print(f"Error: Folder {fchk_folder} does not exist.")
        return None

    extensions = ['*.fchk', '*.wfn', '*.wfx']
    all_files = []
    for ext in extensions:
        all_files.extend(fchk_folder.glob(ext))

    neutral_files = []
    anion_files = []
    cation_files = []

    # Categorize raw files based on selection
    for file in all_files:
        stem = file.stem
        if anion_extension and stem.endswith(anion_extension):
            if process_anion:
                anion_files.append(str(file))
        elif cation_extension and stem.endswith(cation_extension):
            if process_cation:
                cation_files.append(str(file))
        else:
            # Fallback to neutral
            if process_neutral:
                # Check explicit extension if provided
                if not neutral_extension or stem.endswith(neutral_extension):
                    neutral_files.append(str(file))

    neutral_files.sort()
    anion_files.sort()
    cation_files.sort()

    if process_neutral: print(f'N state molecule files   : {len(neutral_files)}')
    if process_anion:   print(f'N+1 state molecule files : {len(anion_files)}')
    if process_cation:  print(f'N-1 state molecule files : {len(cation_files)}')

    # Map base names to files
    def _get_base(path_str, ext):
        stem = Path(path_str).stem
        if ext and stem.endswith(ext):
            return stem[:-len(ext)]
        return stem

    mol_N = {_get_base(f, neutral_extension): f for f in neutral_files} if process_neutral else {}
    mol_Ani = {_get_base(f, anion_extension): f for f in anion_files} if process_anion else {}
    mol_Cat = {_get_base(f, cation_extension): f for f in cation_files} if process_cation else {}

    # Determine ALL molecules (union of all selected states)
    todas = set()
    if process_neutral: todas |= set(mol_N.keys())
    if process_anion:   todas |= set(mol_Ani.keys())
    if process_cation:  todas |= set(mol_Cat.keys())
    
    # Intersection of SELECTED states (complete molecules for CDFT)
    complete_molecules = None
    if process_neutral:
        complete_molecules = set(mol_N.keys()) if complete_molecules is None else complete_molecules & set(mol_N.keys())
    if process_anion:
        complete_molecules = set(mol_Ani.keys()) if complete_molecules is None else complete_molecules & set(mol_Ani.keys())
    if process_cation:
        complete_molecules = set(mol_Cat.keys()) if complete_molecules is None else complete_molecules & set(mol_Cat.keys())
    
    if complete_molecules is None:
        complete_molecules = set()

    # ALL molecules enter the pipeline (union)
    valid_molecules = todas

    incomplete = sorted(todas - complete_molecules)

    if incomplete:
        print("\nNote: Molecules with incomplete states (CDFT will be skipped, local properties will still be calculated):")
        for mol in incomplete:
            faltan = []
            if process_neutral and mol not in mol_N:   faltan.append("N")
            if process_anion and mol not in mol_Ani:   faltan.append("N+1")
            if process_cation and mol not in mol_Cat:  faltan.append("N-1")
            present = []
            if mol in mol_N:   present.append("N")
            if mol in mol_Ani: present.append("N+1")
            if mol in mol_Cat: present.append("N-1")
            print(f"  - {mol}: has [{', '.join(present)}], missing [{', '.join(faltan)}]")
    else:
        print("\nAll molecules have the selected states.")

    # Check extension consistency per molecule (only for states that exist)
    final_valid = set()
    mixed_extension = set()
    
    for mol in valid_molecules:
        exts = []
        if process_neutral and mol in mol_N:   exts.append(Path(mol_N[mol]).suffix)
        if process_anion and mol in mol_Ani:   exts.append(Path(mol_Ani[mol]).suffix)
        if process_cation and mol in mol_Cat:  exts.append(Path(mol_Cat[mol]).suffix)
        
        if len(set(exts)) <= 1:
            final_valid.add(mol)
        else:
            mixed_extension.add(mol)
    
    if mixed_extension:
        print("\nWarning: Molecules with mixed extensions found (will be ignored):")
        for mol in sorted(mixed_extension):
            print(f"  - {mol}")

    if not final_valid:
        print("\nNo valid molecules found for selection.")
        return None

    # Return aligned lists
    final_neutral = [mol_N.get(m) for m in sorted(final_valid)] if process_neutral else [None]*len(final_valid)
    final_anion = [mol_Ani.get(m) for m in sorted(final_valid)] if process_anion else [None]*len(final_valid)
    final_cation = [mol_Cat.get(m) for m in sorted(final_valid)] if process_cation else [None]*len(final_valid)
    
    return final_neutral, final_anion, final_cation, complete_molecules

def charge_calculation(item):
    """Calculate atomic charges for given molecule and method"""
    index, method, suffix = item
    file_path = Path(all_files[index])
    file_suffix = file_path.suffix
    if suffix == 'EEM':
        if file_path.stem.endswith(anion_extension):
            charge = "-1"
        elif file_path.stem.endswith(cation_extension):
            charge = "1"
        else:
            charge = "0"
        method = method.replace("charge", charge)

    if not Path(f"{Path(charges_folder, file_path.stem)}_{suffix}.chg").exists():
        temp_dir = tempfile.mkdtemp()
        temp_dir_path = Path(temp_dir)
        input_file = temp_dir_path / "inputs.txt"
        try:

            # Execute Multiwfn in temporary directory
            if suffix == 'AIM':
                input_content = f"""
            {file_path}
            17
            {method}
            """
                out_calc = temp_dir_path / f"out_{file_path.stem}_{suffix}.chg"
                output = temp_dir_path / f"{file_path.stem}.chg"

                with open(input_file, "w") as f:
                    f.write(input_content)

                run_multiwfn_output(input_file, temp_dir_path, out_calc)
                extract_AIM_charges(out_calc, output)
            else:
                file = open(input_file, "w")
                file.write(f"""
                {str(all_files[index])}
                7
                {method}
                """)
                file.close()
                run_multiwfn(input_file, temp_dir_path)

            if Path(temp_dir_path, f"{file_path.stem}.chg").exists():

                shutil.move(Path(temp_dir_path, f"{file_path.stem}.chg"),
                            Path(charges_folder, Path(f"{Path(charges_folder, file_path.stem)}_{suffix}.chg")))
                with print_lock:
                    print(index + 1,
                          f"--> File {file_path.stem}_{suffix}.chg generated")
            else:
                print('File not generated')
                errors.append(f"{file_path.stem}_{suffix}.chg")
        finally:
            shutil.rmtree(temp_dir)  # Remove temporary directory

    else:
        with print_lock:
            print(index + 1, f"File {file_path.stem}_{suffix}.chg already exists")

def generate_and_execute(index, folder):
    """Generate input file and execute CDFT calculation"""
    neutral_stem = Path(neutral_file_list[index]).stem
    if not Path(folder, f"{neutral_stem}_CDFT.txt").exists():
        temp_dir = tempfile.mkdtemp()
        try:
            temp_dir_path = Path(temp_dir)
            file = open(temp_dir_path / "inputs.txt", "w")
            file.write(f"""
{str(neutral_file_list[index])}
22
2
{str(neutral_file_list[index])}
{str(anion_file_list[index])}
{str(cation_file_list[index])}
0
q
""")
            file.close()

            # Execute Multiwfn in temporary directory
            run_multiwfn(temp_dir_path/"inputs.txt", temp_dir_path)

            if (temp_dir_path/"CDFT.txt").exists():
                shutil.move(temp_dir_path/"CDFT.txt",
                            Path(folder, f"{neutral_stem}_CDFT.txt"))
                with print_lock:
                    print(index + 1,
                          f"--> File {neutral_stem}_CDFT.txt generated successfully ")
            else:
                errors.append(neutral_stem)
        finally:
            shutil.rmtree(temp_dir)

    else:
        with print_lock:
            print(index + 1, f"File {neutral_stem}_CDFT.txt already exists")

# Create task queue
task_queue = queue.Queue()

def calculate_cdft(folder):
    """Worker function for CDFT calculation queue"""
    while True:
        index = task_queue.get()
        if index is None:
            break
        try:
            generate_and_execute(index, folder)
        finally:
            task_queue.task_done()

def create_xyz_input(file_path, output_folder):
    """Create input file for XYZ generation"""
    base_name = Path(file_path).stem
    input_path = Path(output_folder, f"{base_name}_input.txt")

    with open(input_path, "w") as file:
        file.write(f"""
{str(file_path)}
100
2
2

0
q
""")
    return input_path

def generate_xyz_files(file_list, xyz_folder, strip_ext=""):
    """Generate XYZ files from input files, stripping suffix if present"""
    errors = []

    for file_path in file_list:
        stem = Path(file_path).stem
        if strip_ext and stem.endswith(strip_ext):
            base_name = stem[:-len(strip_ext)]
        else:
            base_name = stem
            
        xyz_name = f"{base_name}.xyz"
        xyz_path = Path(xyz_folder, xyz_name)

        if not xyz_path.exists():
            input_path = create_xyz_input(file_path, xyz_folder)
            run_multiwfn(input_path, xyz_folder)            
            generated_xyz = Path(xyz_folder, f"{stem}.xyz")
            if generated_xyz.exists():
                if generated_xyz != xyz_path:
                    try:
                        os.rename(generated_xyz, xyz_path)
                    except Exception as e:
                        print(f"Error renaming {generated_xyz} to {xyz_path}: {e}")
                        errors.append(base_name)
                        continue

                print(f"--> File {xyz_name} generated")
                # Remove temporary input file
                if os.path.exists(input_path):
                    os.remove(input_path)
            else:
                errors.append(base_name)

    return errors

def xyz_to_mol(xyz_file, output_directory, charge=0):
    """Convert XYZ file to MOL format using RDKit"""
    # Read XYZ file
    with open(xyz_file, 'r') as f:
        xyz_data = f.readlines()

    # XYZ file data
    num_atoms = int(xyz_data[0])
    atoms = []
    coordinates = []
    for line in xyz_data[2:]:
        parts = line.split()
        atoms.append(parts[0])
        coordinates.append([float(x) for x in parts[1:4]])

    # Create RDKit molecule
    mol = Chem.rdchem.EditableMol(Chem.rdchem.Mol())
    for atom in atoms:
        mol.AddAtom(Chem.rdchem.Atom(atom))

    mol = mol.GetMol()

    # Add conformer
    conf = Chem.rdchem.Conformer(num_atoms)
    for i, coord in enumerate(coordinates):
        conf.SetAtomPosition(i, coord)
    mol.AddConformer(conf)

    # Determine bonds
    ex = None
    for charge_try in [charge] + list(range(-1, -11, -1)):
        try:
            Chem.rdDetermineBonds.DetermineBonds(mol, charge=charge_try)
            break
        except Exception as e:
            ex = e
    else:
        raise ex

    # Create MOL content
    mol_content = Chem.MolToMolBlock(mol)

    # Write .MOL file
    mol_name = os.path.splitext(os.path.basename(xyz_file))[0] + ".mol"
    mol_path = os.path.join(output_directory, mol_name)
    with open(mol_path, 'w') as f:
        f.write(mol_content)

    return mol_path

# Create lock for output synchronization
print_lock = threading.Lock()

def request_paths():
    """
    Request paths from user:
      - FCHK files folder
      - LOG files folder (optional)
      - Output folder
    """
    while True:
        fchk_path = input("Enter path to folder containing input files (.fchk, .wfn, .wfx): ").strip()
        fchk_path = Path(fchk_path).expanduser().resolve()

        if not fchk_path.exists() or not fchk_path.is_dir():
            print("Path is not a valid directory. Try again.")
            continue

        if not (any(fchk_path.glob("*.fchk")) or any(fchk_path.glob("*.wfn")) or any(fchk_path.glob("*.wfx"))):
            print("Directory is empty or contains no .fchk/.wfn/.wfx files. Please enter a valid path.")
            continue

        break

    # Request LOG files folder path
    while True:
        logs_path = input(
            "Enter path to folder containing .log files (optional, press Enter to skip): ").strip()
        if not logs_path:
            logs_path = None
            break

        logs_path = Path(logs_path).expanduser().resolve()
        if not logs_path.exists() or not logs_path.is_dir():
            print("Path is not a valid directory. Try again.")
            continue

        if not any(logs_path.glob("*.log")):
            print("Log directory is empty or contains no .log files. Try again.")
            continue

        break

    # Request output folder path, create if doesn't exist
    output_path = input("Enter path for output folder (will be created if doesn't exist): ").strip()
    output_path = Path(output_path).expanduser().resolve()
    if not output_path.exists():
        output_path.mkdir(parents=True, exist_ok=True)

    return fchk_path, logs_path, output_path

# Request paths from user and assign global variables
fchk_folder, logs_folder, work_path = request_paths()

# Request processing options
def ask_process(name):
    resp = input(f"Process {name} state? (y/n, default y): ").strip().lower()
    return resp != 'n'

print("\nSelect states to process:")
process_neutral = ask_process("Neutral (N)")
process_anion = ask_process("Anion (N+1)")
process_cation = ask_process("Cation (N-1)")

# Request extensions for FCHK files in selected states
print("\nEnter file extensions (suffixes) for the selected states:")
neutral_extension = input('Enter extension for neutral files (press Enter for default ""): ') or "" if process_neutral else None
anion_extension = input('Enter extension for anion files (press Enter for default "-ani"): ') or "-ani" if process_anion else None
cation_extension = input('Enter extension for cation files (press Enter for default "-cat"): ') or "-cat" if process_cation else None

# Verify at least one state is selected
if not (process_neutral or process_anion or process_cation):
    print("Error: No states selected for processing.")
    exit()

# Verify all corresponding files are present
result = None
while result is None:
    result = check_files()

neutral_file_list, anion_file_list, cation_file_list, complete_molecules = result
errors = []

# Create a global map of molecule base name to file paths for each state
molecule_file_map = {}
# Populate molecule map
for i in range(len(neutral_file_list)):
    n_file = neutral_file_list[i]
    a_file = anion_file_list[i]
    c_file = cation_file_list[i]

    # Get base name from available file
    if n_file:
        base_name = get_molecule_name(n_file, neutral_extension)
    elif a_file:
        base_name = get_molecule_name(a_file, anion_extension)
    elif c_file:
        base_name = get_molecule_name(c_file, cation_extension)
    else:
        continue 
    
    molecule_file_map[base_name] = {
        'N': n_file,
        'N+1': a_file,
        'N-1': c_file
    }

# Create XYZ and MOL folders
xyz_folder = Path(work_path, "xyz")
xyz_folder.mkdir(parents=True, exist_ok=True)
mol_folder = Path(work_path, "mol")
mol_folder.mkdir(parents=True, exist_ok=True)

# Select target list for structure generation
target_files_list = []
target_extensions_list = []
target_extension = "" 

for i in range(len(neutral_file_list)):
    n_file = neutral_file_list[i]
    a_file = anion_file_list[i]
    c_file = cation_file_list[i]
    
    if n_file and process_neutral:
        target_files_list.append(n_file)
        target_extensions_list.append(neutral_extension)
        if not target_extension: target_extension = neutral_extension
    elif a_file and process_anion:
        target_files_list.append(a_file)
        target_extensions_list.append(anion_extension)
        if not target_extension: target_extension = anion_extension
    elif c_file and process_cation:
        target_files_list.append(c_file)
        target_extensions_list.append(cation_extension)
        if not target_extension: target_extension = cation_extension

# Generate XYZ files
print(f"Generating XYZ files in /{Path(xyz_folder).name} ...")
errors = []
for idx_f, file_path in enumerate(target_files_list):
    ext = target_extensions_list[idx_f] if idx_f < len(target_extensions_list) else target_extension
    errs = generate_xyz_files([file_path], xyz_folder, strip_ext=ext)
    errors.extend(errs)

# Generate MOL files
print(f"Generating MOL files in /{Path(mol_folder).name} ...")
error_molecules = []
xyz_directory = work_path / xyz_folder
# Iterate through target files
for idx_f, file_path in enumerate(target_files_list):
    ext = target_extensions_list[idx_f] if idx_f < len(target_extensions_list) else target_extension
    name = Path(file_path).name
    # Simple extraction using get_molecule_name to handle suffix
    base_name = get_molecule_name(file_path, ext)

    xyz_file = os.path.join(xyz_directory, f"{base_name}.xyz")
    mol_name = os.path.splitext(os.path.basename(xyz_file))[0] + ".mol"
    mol_path = os.path.join(mol_folder, mol_name)
    if not Path(mol_path).exists():
        if os.path.exists(xyz_file):
            try:
                mol_file = xyz_to_mol(xyz_file, mol_folder)
                print(f"--> File {base_name}.mol generated")
            except Exception as e:
                error_molecules.append(base_name)
                print(f"Error converting {base_name}: {e}")
        else:
            print(f"File {xyz_file} not found")

if error_molecules:
    print("\nErrors generating MOL files for following molecules:")
    for mol in error_molecules:
        print(mol)


''' Section 3: DFT properties calculation (Generate CDFT files) '''

# Check if full triad is available for CDFT
process_cdft = (process_neutral and process_anion and process_cation) and len(complete_molecules) > 0

cdft_folder = Path(work_path, "CDFT")
cdft_folder.mkdir(parents=True, exist_ok=True)

# Get number of CPUs and calculate upper limit
num_cpus = os.cpu_count()
max_calculations = max(1, num_cpus // 4)

while True:
    try:
        n = int(input(f"\nEnter number of simultaneous calculations to perform (default {max_calculations}, maximum {max_calculations})"
                      f"\n(recommended to test with number that doesn't saturate CPU usage over 90%): ") or max_calculations)
        if 0 < n <= max_calculations:
            break
        else:
            print(f"Please enter an integer greater than 0 and less than or equal to {max_calculations}.")
    except ValueError:
        print("Please enter a valid integer.")

print(f"\nPerforming {n} simultaneous calculations.")

if process_cdft:

    errors = []

    print("\nSoftness units are eV^-1, sum_zero-point, dU, dH and dG are eV/particle and others in eV")
    print(f"Generating {len(neutral_file_list)} global and local CDFT property files in /{Path(cdft_folder).name} ...")

    # Create and launch calculation threads
    threads = []
    for _ in range(n):
        t = threading.Thread(target=calculate_cdft, args=(cdft_folder,))
        t.start()
        threads.append(t)

    # Put only complete molecules in queue
    for i in range(len(neutral_file_list)):
        # Skip molecules that don't have all 3 states
        n_file = neutral_file_list[i]
        a_file = anion_file_list[i]
        c_file = cation_file_list[i]
        if n_file and a_file and c_file:
            task_queue.put(i)

    # Wait for all tasks to complete
    task_queue.join()

    # Stop calculations
    for _ in range(n):
        task_queue.put(None)
    for t in threads:
        t.join()

    if len(errors) > 0:
        print(f"Error occurred for files: {errors}\n")
else:
    print("\nSkipping CDFT calculation (requires Neutral, Anion, and Cation states).")

''' Section 4: Extract global reactivity parameters (Extract from CDFT files) '''

if process_cdft:
    # Conversion factor for Hartree to eV energies
    fconv = 27.2113862459

    print("Extracting global properties and generating .csv file ... ")

def check_logs(logs_folder, file_list, extension):
    """
    Verify that all corresponding .log files exist in logs folder
    """
    if logs_folder is None or not Path(logs_folder).exists() or not any(Path(logs_folder).glob("*.log")):
        print("Logs folder doesn't exist or contains no .log files")
        return
    logs_path = Path(logs_folder)
    missing_logs = set()
    for file_path in file_list:
        if not file_path: continue
        base_name = get_molecule_name(file_path, extension)
        log_file = logs_path / f"{base_name}.log"
        if not log_file.exists():
            missing_logs.add(f"{base_name}.log")

    if missing_logs:
        print("Not all log files found in logs_folder. Missing:", ", ".join(sorted(missing_logs)))
        return False

    return True

def extract_log_data(base_name, logs_folder, fconv, energies):
    """
    Extract log data for a molecule
    """
    log_path = Path(logs_folder) / f"{base_name}.log"

    sum_zero_point = None
    dU = None
    dH = None
    dG = None
    dipole_moment = None
    dipole_flag = False

    with open(log_path, 'r') as file:
        for line in file:
            # Energies
            if "Sum of electronic and zero-point" in line:
                sum_zero_point = line.split()[-1]
            elif "Sum of electronic and thermal Energies" in line:
                dU = line.split()[-1]
            elif "Sum of electronic and thermal Enthalpies" in line:
                dH = line.split()[-1]
            elif "Sum of electronic and thermal Free Energies" in line:
                dG = line.split()[-1]

            # Detection of Dipole moment block
            elif "Dipole moment (field-independent basis, Debye):" in line:
                dipole_flag = True
            elif dipole_flag and "Tot=" in line:
                idx = line.find("Tot=")
                if idx != -1:
                    val_str = line[idx+4:].split()[0]
                    try:
                        dipole_moment = float(val_str)
                    except ValueError:
                        dipole_moment = None
                dipole_flag = False

    energies['sum_zero-point'].append(
        float(sum_zero_point) if sum_zero_point is not None else None
    )
    energies['dU'].append(
        float(dU) * fconv if dU is not None else None
    )
    energies['dH'].append(
        float(dH) * fconv if dH is not None else None
    )
    energies['dG'].append(
        float(dG) * fconv if dG is not None else None
    )
    energies['Dipole_Moment'].append(
        dipole_moment
    )


def extract_properties(cdft_name, cdft_file, logs_folder, fconv, energies, logs_found):
    """
    Process CDFT file to extract properties and extract .log data if available
    """
    base_name = cdft_name.split("_CDFT")[0]

    with open(cdft_file, 'r') as file:
        lines = file.readlines()

    found_values = {
        'Nucleophilicity': None, 'Electrophilicity': None, 'Softness': None,
        'Hardness': None, 'ChemPotential': None, 'MullikenElectronegativity': None,
        'EA': None, 'IP': None, 'E_Cation': None, 'E_Anion': None, 'E_Neutral': None
    }

    # Extract global reactivity data (units in eV)
    for line in lines:
        parts = line.replace(',', ' ').split()
        if not parts: continue
        
        if "Nucleophilicity" in parts:
            found_values['Nucleophilicity'] = parts[-2]
        elif "Electrophilicity" in parts:
            found_values['Electrophilicity'] = parts[-2]
        elif "Softness:" in parts:
            found_values['Softness'] = parts[-2]
        elif "Hardness" in parts:
            found_values['Hardness'] = parts[-2]
        elif "Chemical" in parts and "potential:" in line:
            found_values['ChemPotential'] = parts[-2]
        elif "Mulliken" in parts:
            found_values['MullikenElectronegativity'] = parts[-2]
        elif "EA:" in parts:
            found_values['EA'] = parts[-2]
        elif "IP:" in parts:
            found_values['IP'] = parts[-2]
        elif "E(N-1):" in parts:
            try: found_values['E_Cation'] = float(parts[-2]) * fconv
            except: pass
        elif "E(N+1):" in parts:
            try: found_values['E_Anion'] = float(parts[-2]) * fconv
            except: pass
        elif "E(N):" in parts:
            try: found_values['E_Neutral'] = float(parts[-2]) * fconv
            except: pass

    # Append found values
    energies['Molecule'].append(base_name)
    for key, value in found_values.items():
        if key in energies:
            energies[key].append(value)

    # Extract .log information if all logs available
    if logs_found:
        extract_log_data(base_name, logs_folder, fconv, energies)

# Verify if all log files exist
if process_cdft:
    logs_found = check_logs(logs_folder, target_files_list, target_extension)
    energies = {
        'Molecule': [],
        'Nucleophilicity': [],
        'Electrophilicity': [],
        'Softness': [],
        'Hardness': [],
        'ChemPotential': [],
        'MullikenElectronegativity': [],
        'EA': [],
        'IP': [],
        'E_Cation': [],
        'E_Anion': [],
        'E_Neutral': []
    }
    if logs_found:
        energies.update({
            'sum_zero-point': [],
            'dU': [],
            'dH': [],
            'dG': [],
            'Dipole_Moment': []
        })

    # Get base names of files
    valid_bases = set()
    for file_path in target_files_list:
        base_name = get_molecule_name(file_path, target_extension)
        valid_bases.add(base_name)

    # Process valid CDFT files
    for file in Path(cdft_folder).glob("*_CDFT.txt"):
        stem = file.stem
        temp_base = stem.split("_CDFT")[0]
        
        if temp_base in valid_bases:
            extract_properties(stem, file, logs_folder, fconv, energies, logs_found)
            continue
        
        if neutral_extension and temp_base.endswith(neutral_extension):
            stripped_base = temp_base[:-len(neutral_extension)]
            if stripped_base in valid_bases:
                extract_properties(stem, file, logs_folder, fconv, energies, logs_found)

    print("Extraction completed.")

if process_cdft:
    try:
        global_df = pd.DataFrame(energies)
    except ValueError as e:
        print("Error creating DataFrame:", e)
        print("Attempting to align columns and remove rows with incomplete data...")
        global_df = pd.DataFrame({key: pd.Series(value) for key, value in energies.items()})
        global_df.replace("", pd.NA, inplace=True)
        global_df.dropna(inplace=True)

    global_df['IP'] = pd.to_numeric(global_df['IP'], errors='coerce')
    global_df['EA'] = pd.to_numeric(global_df['EA'], errors='coerce')

    # Calculate 'w+' and 'w-'
    global_df['w+'] = ((3 * global_df['IP'] + global_df['EA']) ** 2) / (16 * (global_df['IP'] - global_df['EA']))
    global_df['w-'] = ((global_df['IP'] + 3 * global_df['EA']) ** 2) / (16 * (global_df['IP'] - global_df['EA']))
    print(global_df)
    save_csv(global_df, "global_properties.csv", work_path)
else:
    print("\nSkipping Global Properties extraction (no CDFT files generated).")

''' Section 5: Base fragment search and atoms for property calculation in each molecule '''
expected_xyz_names = set()
for file_path in target_files_list:
    base_name = get_molecule_name(file_path, target_extension)
    expected_xyz_names.add(f"{base_name}.xyz")

# Filter XYZ files in folder
xyz_files_list = [i for i in Path(xyz_folder).glob("*.xyz") if i.name in expected_xyz_names]

# Check for missing XYZ files
missing_xyz = expected_xyz_names - {i.name for i in xyz_files_list}
if missing_xyz:
    print("Warning: Missing following XYZ files:", ", ".join(sorted(missing_xyz)))

while True:
    choice = input(
        f"\nTo extract local properties, enter filename of one of your {len(target_files_list)} molecules "
        f"to select the fragment common to all your molecules (d = choose first in list): ")
    if choice == 'd':
        first_base = get_molecule_name(target_files_list[0], target_extension)
        file = Path(xyz_folder) / f"{first_base}.xyz"
        print(f"Openning file {file}")
        print(f"\nCommon fragment and atoms for property extraction will be selected using molecule {first_base}")
        break
    else:
        file = Path(xyz_folder) / f"{choice}.xyz"
        if file.name in [f.name for f in xyz_files_list]:
            break
        else:
            print("\nEntered file not found in your xyz folder, choose another")

# Define fragment search specificity level
print("\nDefine fragment search specificity:")
print("'0': Connectivity only (Matches based on internal bonds of the fragment).")
print("'1': Specificity (Matches require matching neighbor environment).")
specificity = input("Enter specificity (0/1, default 0): ").strip()
if specificity not in ['0', '1']:
    specificity = "0"

results, atoms_of_interest, neighbor_dict = Ff.start(file, specificity)

print(f"ATOMS OF INTEREST: {atoms_of_interest}")
# Keep only first match per molecule
multi_match_count = 0
for key, matches in results.items():
    if len(matches) > 1:
        multi_match_count += 1
        results[key] = [matches[0]]

if multi_match_count > 0:
    print(f"\nWARNING! Files with >1 match: {multi_match_count}")
    print("Define the common fragment more robustly.")
    print("Otherwise, the first match will be taken for files with more than one match.\n")

''' Section 6: Extraction and calculation of local properties (Extract from CDFT files) '''

cdft_file_names = {p.name for p in Path(cdft_folder).glob("*_CDFT.txt")}

# Initialize dictionary to store data
atom_properties = []

# Extract CDFT local properties for molecules that have CDFT files
for file, match_list in results.items():
    target_filename = f"{file}_CDFT.txt"
    if target_filename in cdft_file_names:
        molecule_name = file
        properties = {'Molecule': molecule_name}
        for element in match_list:
            search_atoms = element['selected_atoms']
            with open(Path(cdft_folder, f"{file}_CDFT.txt"), 'r') as file_obj:
                lines = file_obj.readlines()
                for i, atom in enumerate(search_atoms):
                    count = 0
                    for line in lines:
                        parts = line.split()
                        if parts:
                            part_idx = parts[0].split('(')[0]
                            atom_idx = atom.split('(')[0]
                            
                            if part_idx == atom_idx:
                                if count == 0:
                                    # Atomic properties
                                    count += 1
                                elif count == 1:
                                    properties.update(
                                        {f'{atoms_of_interest[i]}_Electrophilicity_Hirshfeld': parts[-2],
                                         f'{atoms_of_interest[i]}_Nucleophilicity_Hirshfeld': parts[-1]}
                                    )
                                    count += 1
                                elif count == 2:
                                    # Handle variable columns (s(2) might be present or absent)
                                    # Count float values from end
                                    vals = []
                                    for p in reversed(parts):
                                        try:
                                            float(p)
                                            vals.insert(0, p)
                                        except ValueError:
                                            break
                                    
                                    if len(vals) >= 6:
                                        # s(2) is present
                                        s_minus = vals[0]
                                        s_plus = vals[1]
                                        s0 = vals[2]
                                        ratio_plus_minus = vals[3]
                                        ratio_minus_plus = vals[4]
                                        s2 = vals[5]
                                        has_s2 = True
                                    else:
                                        # s(2) is absent
                                        s_minus = vals[0]
                                        s_plus = vals[1]
                                        s0 = vals[2]
                                        ratio_plus_minus = vals[3]
                                        ratio_minus_plus = vals[4]
                                        has_s2 = False
                                    
                                    update_dict = {
                                        f'{atoms_of_interest[i]}_s-_Hirshfeld': s_minus,
                                        f'{atoms_of_interest[i]}_s+_Hirshfeld': s_plus,
                                        f'{atoms_of_interest[i]}_s0_Hirshfeld': s0,
                                        f'{atoms_of_interest[i]}_s+/s-_Hirshfeld': ratio_plus_minus,
                                        f'{atoms_of_interest[i]}_s-/s+_Hirshfeld': ratio_minus_plus
                                    }
                                    
                                    if has_s2:
                                        update_dict[f'{atoms_of_interest[i]}_s(2)_Hirshfeld'] = s2
                                        
                                    properties.update(update_dict)
                                    count += 1
                                    break
        atom_properties.append(properties)

for file, match_list in results.items():
    target_filename = f"{file}_CDFT.txt"
    if target_filename not in cdft_file_names:
        atom_properties.append({'Molecule': file})

# Print dictionary with data and save to CSV
local_df = pd.DataFrame(atom_properties)
print("\nExtracted local properties (CDFT-based):")
print(local_df)

save_csv(local_df, "local_properties.csv", work_path)

global_df = None
local_df = None
if (work_path / 'global_properties.csv').exists():
    try: global_df = pd.read_csv(work_path / 'global_properties.csv')
    except: pass
if (work_path / 'local_properties.csv').exists():
    try: local_df = pd.read_csv(work_path / 'local_properties.csv')
    except: pass

if global_df is not None and local_df is not None:
    combined_df = merge_dataframes(global_df, local_df)
    if combined_df is not None:
        save_csv(combined_df, 'properties.csv', work_path)
        print("Merged global and local properties saved in 'properties.csv'")
elif global_df is not None:
     save_csv(global_df, 'properties.csv', work_path)
     print("Saved global properties to 'properties.csv' (no local properties found)")
elif local_df is not None:
     save_csv(local_df, 'properties.csv', work_path)
     print("Saved local properties to 'properties.csv' (no global properties found)")
else:
     print("No properties generated to save in 'properties.csv'")


# Atomic charge calculation and CPs file extraction

# List of available charge methods 
available_method_suffixes = [
    ("""1
1
y
0
q""", "Hirshfeld"),
    ("""2
1
y
0
q""", "Voronoi"),
    ("""5
1
y
0
0
q""", "Mulliken"),
    ("""6
    
y
0
q""", "Lowdin"),
    ("""7
y
0
q""", "Mk-RS"),
    ("""8
y
0
q""", "Mk-SP"),
    ("""9
y
0
q""", "Mk-Bh"),
    ("""10
0
y
0
q""", "Becke"),
    ("""11
1
y
0
q""", "ADCH"),
    ("""16
1
y
0
q""", "CM5"),
    (f"""17
g
2
charge
0
y
-1
0
q""", "EEM"),
    ("""19
y
0
q""", "GAST"),
    ("""18
1
y
0
0
q""", "RESP"),
    ("""13
1
y
0
0
q""", "MK"),
    ("""12
1
y
0
0
q""", "CHELPG"),
    ("""20
1
y
0
q""", "STOCK"),
    ("""20
1
1
2
7
2
1
-10
q""", "AIM")
]

print("Select one or more atomic charge calculation methods:")
for idx, (input_str, method_name) in enumerate(available_method_suffixes, start=1):
    print(f"{idx}. {method_name}")

while True:
    selection = input(
        "Enter corresponding numbers separated by commas (default 1 for Hirshfeld): ").strip()
    if not selection:
        selection = "1"
    tokens = [t.strip() for t in selection.split(',') if t.strip()]
    valid = True
    selected_indices = []
    for token in tokens:
        if token.isdigit():
            num = int(token)
            if 1 <= num <= len(available_method_suffixes):
                selected_indices.append(num)
            else:
                print(f"Number {token} out of range.")
                valid = False
                break
        else:
            print(f"Invalid entry: {token} is not a number.")
            valid = False
            break
    if valid and selected_indices:
        selected_indices = sorted(set(selected_indices))
        method_suffixes = [available_method_suffixes[i - 1] for i in selected_indices]
        break
    else:
        print("Invalid entry. Try again.")

print("Selected charge methods:")
for m in method_suffixes:
    print(m[1])

# Create folder to save charge files
charges_folder = os.path.join(work_path, "charges")
os.makedirs(charges_folder, exist_ok=True)

# List of all files (neutral, anions, cations)
all_files = [f for f in (neutral_file_list + anion_file_list + cation_file_list) if f]

print(f"Calculating charges for {len(method_suffixes)} method(s)...")

# Create task queue for calculations
task_queue = queue.Queue()
threads = []
charge_threads = n
for _ in range(charge_threads):
    t = threading.Thread(target=calculate_charges)
    t.start()
    threads.append(t)
for method, suffix in method_suffixes:
    for i in range(len(all_files)):
        task_queue.put((i, method, suffix))
task_queue.join()
for _ in range(charge_threads):
    task_queue.put(None)
for t in threads:
    t.join()

print(f"Errors in charge calculation: {errors}")

def extract_charges(suffix):
    """Extract charges and save to CSV"""
    # Iterate using index
    for index in range(len(neutral_file_list)):
        neutral_file = neutral_file_list[index]
        anion_file = anion_file_list[index]
        cation_file = cation_file_list[index]

        # Determine base name
        if neutral_file:
            base_name = get_molecule_name(neutral_file, neutral_extension)
            neutral_stem = Path(neutral_file).stem
        elif anion_file:
            base_name = get_molecule_name(anion_file, anion_extension)
            neutral_stem = ""
        elif cation_file:
            base_name = get_molecule_name(cation_file, cation_extension)
            neutral_stem = ""
        else:
            continue
        
        # Determine stems for paths if file exists
        n_stem = Path(neutral_file).stem if neutral_file else ""
        a_stem = Path(anion_file).stem if anion_file else ""
        c_stem = Path(cation_file).stem if cation_file else ""

        neutral_file_path = os.path.join(charges_folder, f"{n_stem}_{suffix}.chg") if n_stem else ""
        anion_file_path = os.path.join(charges_folder, f"{a_stem}_{suffix}.chg") if a_stem else ""
        cation_file_path = os.path.join(charges_folder, f"{c_stem}_{suffix}.chg") if c_stem else ""

        # Check existence
        has_n = neutral_file_path and os.path.exists(neutral_file_path)
        has_a = anion_file_path and os.path.exists(anion_file_path)
        has_c = cation_file_path and os.path.exists(cation_file_path)

        if has_n or has_a or has_c:
            try:
                # Open available files
                file_neutral = open(neutral_file_path, 'r') if has_n else None
                file_anion = open(anion_file_path, 'r') if has_a else None
                file_cation = open(cation_file_path, 'r') if has_c else None
                
                rows = []

                # Read lines
                lines_neutral = file_neutral.readlines() if file_neutral else []
                lines_anion = file_anion.readlines() if file_anion else []
                lines_cation = file_cation.readlines() if file_cation else []
                
                # Close files
                if file_neutral: file_neutral.close()
                if file_anion: file_anion.close()
                if file_cation: file_cation.close()

                # Use the longest available line list as driver
                driver_lines = lines_neutral or lines_anion or lines_cation
                
                for i, line in enumerate(driver_lines):
                    def get_val(lines_list, index, method_suffix):
                        if index >= len(lines_list): return None
                        l_str = lines_list[index]
                        if method_suffix == "AIM":
                            parts = l_str.split()
                            if "Charge:" in l_str:
                                return float(parts[parts.index("Charge:")+1])
                            return None
                        else:
                            pat = r'(\w+)\s+([-]?\d+\.\d+)\s+([-]?\d+\.\d+)\s+([-]?\d+\.\d+)\s*([-]?\d+\.\d+)'
                            match = re.match(pat, l_str)
                            if match: return float(match.groups()[4]) # index 4 is charge
                            return None

                    atom = None
                    if suffix == "AIM":
                        match = re.match(r'\s*(\d+\s*\([A-Za-z]{1,2}\s*\))\s+Charge:', line)
                        if match: atom = match.group(1).split()[0]
                    else:
                        atom = line.split()[0]

                    if not atom: continue

                    val_n = get_val(lines_neutral, i, suffix) if has_n else None
                    val_a = get_val(lines_anion, i, suffix) if has_a else None
                    val_c = get_val(lines_cation, i, suffix) if has_c else None
                    
                    rows.append({
                        "Atom": atom,
                        "q(N)": val_n,
                        "q(N+1)": val_a,
                        "q(N-1)": val_c
                    })

                # Create DataFrame
                df_charges = pd.DataFrame(rows)
                
                # Drop columns with all NaN or all Zeros
                df_charges.replace({"": pd.NA, None: pd.NA}, inplace=True)
                
                # Drop columns that are all NaN
                df_charges.dropna(axis=1, how='all', inplace=True)
                numeric_cols = df_charges.select_dtypes(include=[np.number]).columns
                cols_to_drop = [c for c in numeric_cols if (df_charges[c].fillna(0) == 0).all()]
                df_charges.drop(columns=cols_to_drop, inplace=True)

                # Save if not empty
                if not df_charges.empty:
                    df_charges.to_csv(os.path.join(charges_folder, f"{base_name}_{suffix}_charges.csv"), index=False)

            except Exception as e:
                print(f"Error processing charges for {base_name} with suffix {suffix}: {e}")
        else:
            # print(f"Missing charge files for {base_name} with suffix {suffix}")
            pass

for method, suffix in method_suffixes:
    extract_charges(suffix)

# Calculate local Fukui descriptors in each charge file
charge_csv_files = glob.glob(os.path.join(charges_folder, '*.csv'))
for file in charge_csv_files:
    try:
        df = pd.read_csv(file)
        has_qN = 'q(N)' in df.columns
        has_qNp1 = 'q(N+1)' in df.columns
        has_qNm1 = 'q(N-1)' in df.columns

        # Fukui indices require at least two states
        if has_qN and has_qNm1:
            df['f-'] = abs(df['q(N)'] - df['q(N-1)'])
        if has_qNp1 and has_qN:
            df['f+'] = abs(df['q(N+1)'] - df['q(N)'])
        if 'f+' in df.columns and 'f-' in df.columns:
            df['CDD'] = df['f+'] - df['f-']
            df['f0'] = (df['f-'] + df['f+']) * 0.5
            df['f+f-'] = df['f+'] * df['f-']
        if has_qN and 'f+' in df.columns:
            df['q(N)/f+'] = df['q(N)'] / df['f+'].replace(0, np.nan)
        if has_qN and 'CDD' in df.columns:
            df['q(N)/CDD'] = df['q(N)'] / abs(df['f+'] - df['f-']).replace(0, np.nan)
        if has_qN and 'f-' in df.columns:
            df['q(N)/f-'] = df['q(N)'] / df['f-'].replace(0, np.nan)
        if has_qNp1 and 'f+' in df.columns:
            df['q(N+1)/f+'] = df['q(N+1)'] / df['f+'].replace(0, np.nan)
        if has_qNp1 and 'CDD' in df.columns:
            df['q(N+1)/CDD'] = df['q(N+1)'] / abs(df['f+'] - df['f-']).replace(0, np.nan)
        if has_qNp1 and 'f-' in df.columns:
            df['q(N+1)/f-'] = df['q(N+1)'] / df['f-'].replace(0, np.nan)
        if has_qNm1 and 'f+' in df.columns:
            df['q(N-1)/f+'] = df['q(N-1)'] / df['f+'].replace(0, np.nan)
        if has_qNm1 and 'CDD' in df.columns:
            df['q(N-1)/CDD'] = df['q(N-1)'] / abs(df['f+'] - df['f-']).replace(0, np.nan)
        if has_qNm1 and 'f-' in df.columns:
            df['q(N-1)/f-'] = df['q(N-1)'] / df['f-'].replace(0, np.nan)
        # Save directly to same file
        df.to_csv(file, index=False)
    except Exception as e:
        print(f"Error calculating descriptors in {file}: {e}")

# Extract local properties from charge files
# Extract properties for each method for atoms of interest
for input_str, method in method_suffixes:
    atom_properties = []
    for file, match_list in results.items():
        base = file
        charge_file = os.path.join(charges_folder, f"{base}_{method}_charges.csv")
        if os.path.exists(charge_file):
            df_charges = pd.read_csv(charge_file)

            molecule_name = base
            properties = {'Molecule': molecule_name}

            for element in match_list:
                search_indices = element['interest_atom_indices']
                for i, index in enumerate(search_indices):

                    row_idx = index - 1

                    if 0 <= row_idx < len(df_charges):
                        row = df_charges.iloc[row_idx]
                        for col in df_charges.columns[1:]:
                            value = row[col]
                            properties[f'{atoms_of_interest[i]}_{col}_{method}'] = value

            atom_properties.append(properties)
        else:
            print(f'File {base}_{method}_charges.csv does not exist')

    local_method_df = pd.DataFrame(atom_properties)
    print(local_method_df)
    save_csv(local_method_df, f"local_properties_{method}.csv", work_path)

# Combine local properties from different methods using "Molecule" key
local_file_path = work_path / 'local_properties.csv'
if local_file_path.exists():
    try:
        local_combined_df = pd.read_csv(local_file_path)
    except Exception as e:
        print(f"Error reading existing local_properties.csv: {e}")
        local_combined_df = pd.DataFrame({'Molecule': list(results.keys())})
else:
    local_combined_df = pd.DataFrame({'Molecule': list(results.keys())})
for input_str, method in method_suffixes:
    try:
        method_df = pd.read_csv(work_path / f'local_properties_{method}.csv')
        local_combined_df = pd.merge(local_combined_df, method_df, on='Molecule', how='outer')
    except Exception as e:
        print(f"Error merging properties for method {method}: {e}")

local_combined_df = local_combined_df.dropna(axis=1, how='any')
local_combined_df = local_combined_df.loc[:, (local_combined_df.sum() != 0)]
save_csv(local_combined_df, "local_properties.csv", work_path)

# Calculate atomic and bond critical points (CPs) for each state (N, N+1, N-1)
# Create folder to store CP property files
cps_folder = Path(work_path, "cps")
cps_folder.mkdir(exist_ok=True)

# List of properties to extract from CPs
cps_properties_list = [
    'Density of all electrons', 'Density of Alpha electrons', 'Density of Beta electrons',
    'Lagrangian kinetic energy G(r)', 'Hamiltonian kinetic energy K(r)', 'Potential energy density V(r)',
    'Energy density E(r) or H(r)', 'Laplacian of electron density', 'Electron localization function (ELF)',
    'Localized orbital locator (LOL)', 'Local information entropy', 'Sign(lambda2)*rho', 
    'Sign(lambda2)*rho with promolecular approximation', 'Average local ionization energy (ALIE)',
    'Delta-g (under promolecular approximation)', 'Delta-g (under Hirshfeld partition)', 'ESP from nuclear charges',
    'ESP from electrons', 'Total ESP'
]

print("Calculating atomic and bond CP properties for N, N+1 and N-1 states")

def calculate_cps(full_file_path, name, atom_indices, selected_atoms, neighbor_dict):
    """
    Calculate and extract atomic and bond CP properties for molecule 'name'
    using Multiwfn. Generate two file types:
      - {name}_cps_atomic.txt (atomic CPs, type "2")
      - {name}_cps_{atom1}-{atom2}_bond.txt (bond CPs, type "3")
    """
    molecule_file = Path(full_file_path)
    if not molecule_file.exists():
        raise FileNotFoundError(f"Input file not found: {molecule_file}")
    
    print(f"Processing {name}")
    # Convert indices to strings and join
    indices_str = [str(i) for i in atom_indices]
    nuclear_indices = ",".join(indices_str)

    def write_input(indices, cp_type, input_path):
        with open(input_path, "w") as f:
            f.write(f"""
        {molecule_file}
        2
        -1
        10
        1
        {indices}
        0
        {cp_type}
        7
        0
        -10
        q
        """)

    # --- Atomic CPs (type "2") ---
    tmp_dir = tempfile.mkdtemp()
    try:
        tmp_path = Path(tmp_dir)
        input_path = tmp_path / "inputs.txt"
        write_input(nuclear_indices, "2", input_path)

        atomic_output = cps_folder / f"{name}_cps_atomic.txt"
        if not atomic_output.exists():
            try:
                run_multiwfn(input_path, tmp_path)
                cp_file = tmp_path / "CPprop.txt"
                if cp_file.exists():
                    shutil.move(str(cp_file), str(atomic_output))
                    with print_lock:
                        print(f"--> File {atomic_output.name} generated")
                else:
                    errors.append(atomic_output.name)
                    print(f"Error: {atomic_output.name} not generated")
            except Exception as e:
                errors.append(atomic_output.name)
                print(f"Error in atomic CPs of {name}: {e}")
        else:
            print(f"File {atomic_output.name} already exists")
    finally:
        shutil.rmtree(tmp_dir)

    # --- Bond CPs (type "3") ---
    print("Searching for BCPs")
    cp_type = "3"
    atom_list = selected_atoms
    atom_list_neighbors = [a for a, _ in neighbor_dict.items()]
    neighbors_list = [v for _, v in neighbor_dict.items()]
    for i, idx1 in enumerate(indices_str):
        atom1 = atom_list[i]
        for j, idx2 in enumerate(indices_str):
            atom2 = atom_list[j]
            if j > i and atom2 in neighbors_list[i]:
                indices_pair = f"{idx1},{idx2}"
                tmp_dir = tempfile.mkdtemp()
                try:
                    tmp_path = Path(tmp_dir)
                    input_path = tmp_path / "inputs.txt"
                    write_input(indices_pair, cp_type, input_path)
                    bond_output = cps_folder / f"{name}_cps_{atoms_of_interest[i]}-{atoms_of_interest[j]}_bond.txt"
                    if not bond_output.exists():
                        try:
                            run_multiwfn(input_path, tmp_path)
                            cp_file = tmp_path / "CPprop.txt"
                            if cp_file.exists() and cp_file.stat().st_size > 0:
                                shutil.move(str(cp_file), str(bond_output))
                                with print_lock:
                                    print(f"--> File {bond_output.name} generated")
                            else:
                                if cp_file.exists():
                                    cp_file.unlink()
                                errors.append(bond_output.name)
                                print(f"Error: {bond_output.name} not generated or empty")
                        except Exception as e:
                            errors.append(bond_output.name)
                            print(f"Error in bond CPs {bond_output.name}: {e}")
                    else:
                        print(f"File {bond_output.name} already exists")
                finally:
                    shutil.rmtree(tmp_dir)

def process_molecule(molecule_name, match_list, neutral_extension, fchk_folder, cps_folder, atoms_of_interest,
                      cps_properties_list, calculation_extension):
    """
    Process molecule in given state (ext_calculo = neutral_extension, anion_extension, cation_extension).
    """
    # Determine state for property labeling
    if calculation_extension == neutral_extension:
        state = 'N'
    elif calculation_extension == anion_extension:
        state = 'N+1'
    elif calculation_extension == cation_extension:
        state = 'N-1'
    else:
        state = 'N'

    if molecule_name in molecule_file_map:
        full_file_path = molecule_file_map[molecule_name].get(state)
        if not full_file_path:
            return []
        # Construct output specific name using base + suffix; {base_name}{suffix}
        output_name = f"{molecule_name}{calculation_extension}"
    else:
        print(f"Error: Molecule {molecule_name} not found in file map.")
        return []

    molecule_properties = []

    for element in match_list:
        atom_indices = element['interest_atom_indices']
        neighbor_dict = element['neighbor_dict']
        selected_atoms = element['selected_atoms']

        # Calculate and extract CPs
        try:
            calculate_cps(full_file_path, output_name, atom_indices, selected_atoms, neighbor_dict)
        except Exception as e:
            print(f"Error calculating CPs for {output_name}: {e}")

        # Initialize dictionary for this molecule
        props = {'Molecule': molecule_name}

        # Extract properties from atomic file
        atomic_cp_file = cps_folder / f"{output_name}_cps_atomic.txt"
        if atomic_cp_file.exists():
            with open(atomic_cp_file, 'r') as cp_file:
                lines = cp_file.readlines()
                current_atom = None
                for line in lines:
                    if 'Corresponding nucleus:' in line:
                        parts = "".join(line.split()[2:])
                        try:
                            atom_index = int(parts.split("(")[0].strip())
                            atom_idx = atom_indices.index(atom_index)
                            current_atom = atoms_of_interest[atom_idx]
                        except Exception:
                            current_atom = None

                    if current_atom:
                        for prop in cps_properties_list:
                            if line.strip().startswith(prop):
                                value = line.split(':')[-1].split()[0].strip()
                                props[f'{prop}_{state}_{current_atom}'] = value
        else:
            print(f"File {atomic_cp_file} not found for property extraction")

        molecule_properties.append(props)

    return molecule_properties

# Calculate atomic and bond CP properties for neutral state
processes = max(1, num_cpus // 2)
if process_neutral:
    cps_atom_properties = []
    calculation_extension = neutral_extension
    with concurrent.futures.ThreadPoolExecutor(max_workers=processes) as executor:
        futures = []
        for file, match_list in results.items():
            future = executor.submit(process_molecule, file, match_list, neutral_extension, fchk_folder,
                                     cps_folder, atoms_of_interest, cps_properties_list, calculation_extension)
            futures.append(future)

        for fut in concurrent.futures.as_completed(futures):
            cps_atom_properties.extend(fut.result())

    cps_atom_properties_df = pd.DataFrame(cps_atom_properties)
    save_csv(cps_atom_properties_df, 'local_properties_cps_atoms_N.csv', work_path)
    print(f"Errors: {errors}")

# Calculate atomic and bond CP properties for anion state
if process_anion:
    cps_atom_properties = []
    calculation_extension = anion_extension
    with concurrent.futures.ThreadPoolExecutor(max_workers=processes) as executor:
        futures = []
        for file, match_list in results.items():
            future = executor.submit(process_molecule, file, match_list, neutral_extension, fchk_folder,
                                     cps_folder, atoms_of_interest, cps_properties_list, calculation_extension)
            futures.append(future)

        for fut in concurrent.futures.as_completed(futures):
            cps_atom_properties.extend(fut.result())

    cps_atom_properties_df = pd.DataFrame(cps_atom_properties)
    save_csv(cps_atom_properties_df, 'local_properties_cps_atoms_N+1.csv', work_path)
    print(f"Errors: {errors}")

# Calculate atomic and bond CP properties for cation state
if process_cation:
    cps_atom_properties = []
    calculation_extension = cation_extension
    with concurrent.futures.ThreadPoolExecutor(max_workers=processes) as executor:
        futures = []
        for file, match_list in results.items():
            future = executor.submit(process_molecule, file, match_list, neutral_extension, fchk_folder,
                                     cps_folder, atoms_of_interest, cps_properties_list, calculation_extension)
            futures.append(future)

        for fut in concurrent.futures.as_completed(futures):
            cps_atom_properties.extend(fut.result())

    cps_atom_properties_df = pd.DataFrame(cps_atom_properties)
    save_csv(cps_atom_properties_df, 'local_properties_cps_atoms_N-1.csv', work_path)
    print(f"Errors: {errors}")

# Merge into local_properties file
local_df = pd.read_csv(work_path / 'local_properties.csv')

if process_neutral:
    try:
        neutral_atom_df = pd.read_csv(work_path / 'local_properties_cps_atoms_N.csv')
        local_df = merge_dataframes(local_df, neutral_atom_df)
    except Exception: pass

if process_anion:
    try:
        anion_atom_df = pd.read_csv(work_path / 'local_properties_cps_atoms_N+1.csv')
        local_df = merge_dataframes(local_df, anion_atom_df)
    except Exception: pass

if process_cation:
    try:
        cation_atom_df = pd.read_csv(work_path / 'local_properties_cps_atoms_N-1.csv')
        local_df = merge_dataframes(local_df, cation_atom_df)
    except Exception: pass

print(local_df)
save_csv(local_df, 'local_properties.csv', work_path)

def extract_bond_cps_properties(cps_folder, cps_properties_list, state):
    """
    Extract bond critical point (3,-1) properties for molecule
    in specified state (N, N+1 or N-1).
    """
    molecule_data = []
    # Build list of .txt files with bond
    if state == "N-1":
        charge = cation_extension
        path_list = list(Path(cps_folder).glob(f'*{charge}_cps_*_bond.txt'))
        molecule_names = [p.stem.split(charge)[0] for p in path_list]
    elif state == "N+1":
        charge = anion_extension
        path_list = list(Path(cps_folder).glob(f'*{charge}_cps_*_bond.txt'))
        molecule_names = [p.stem.split(charge)[0] for p in path_list]
    else:
        # State N
        anion_list = set(Path(p) for p in glob.glob(str(Path(cps_folder) / f'*{anion_extension}_cps_*_bond.txt')))
        cation_list = set(Path(p) for p in glob.glob(str(Path(cps_folder) / f'*{cation_extension}_cps_*_bond.txt')))
        path_list = [Path(p) for p in glob.glob(str(Path(cps_folder) / f'*{neutral_extension}_cps_*_bond.txt'))
                       if Path(p) not in anion_list and Path(p) not in cation_list]
        molecule_names = [p.stem.split('_cps')[0] for p in path_list]

    for i, file in enumerate(path_list):
        molecule_name = molecule_names[i]
        # name_cps_At1-At2_bond.txt --> At1-At2
        try:
            bond_atoms = file.stem.split("_cps_")[1].replace("_bond", "").split("-")
        except IndexError:
            print(f"Malformed bond filename: {file}")
            continue

        properties2 = {'Molecule': molecule_name}
        with open(file, 'r') as file_in:
            for line in file_in:
                line = line.strip()
                for prop in cps_properties_list:
                    if line.startswith(prop):
                        key = f'{prop}_{state}_{bond_atoms[0]}-{bond_atoms[1]}'
                        value = line.split(':')[-1].split()[0].strip()
                        properties2[key] = value

        existing = next((x for x in molecule_data if x['Molecule'] == molecule_name), None)
        if existing:
            existing.update(properties2)
        else:
            molecule_data.append(properties2)

    return molecule_data

# Extract bond properties for N, N+1 and N-1
if process_neutral:
    try:
        props = extract_bond_cps_properties(cps_folder, cps_properties_list, "N")
        bond_df = pd.DataFrame(props)
        save_csv(bond_df, "local_properties_cps_bond_N.csv", work_path)
    except Exception: pass

if process_anion:
    try:
        props = extract_bond_cps_properties(cps_folder, cps_properties_list, "N+1")
        bond_df = pd.DataFrame(props)
        save_csv(bond_df, "local_properties_cps_bond_N+1.csv", work_path)
    except Exception: pass

if process_cation:
    try:
        props = extract_bond_cps_properties(cps_folder, cps_properties_list, "N-1")
        bond_df = pd.DataFrame(props)
        save_csv(bond_df, "local_properties_cps_bond_N-1.csv", work_path)
    except Exception: pass

# Merge with local_properties
local_df = pd.read_csv(work_path / 'local_properties.csv')
# local_df = local_df.dropna(axis=1, how='any')

if process_neutral:
    try:
        neutral_bond_df = pd.read_csv(work_path / 'local_properties_cps_bond_N.csv')
        local_df = merge_dataframes(local_df, neutral_bond_df)
    except Exception: pass

if process_anion:
    try:
        anion_bond_df = pd.read_csv(work_path / 'local_properties_cps_bond_N+1.csv')
        local_df = merge_dataframes(local_df, anion_bond_df)
    except Exception: pass

if process_cation:
    try:
        cation_bond_df = pd.read_csv(work_path / 'local_properties_cps_bond_N-1.csv')
        local_df = merge_dataframes(local_df, cation_bond_df)
    except Exception: pass

# Remove columns that are completely empty
local_df = local_df.dropna(axis=1, how='all')
numeric_cols = local_df.select_dtypes(include='number').columns
local_df = local_df.drop(columns=[c for c in numeric_cols if c != 'Molecule' and (local_df[c].sum() == 0)])
save_csv(local_df, "local_properties.csv", work_path)

# Function to calculate differences between anion, cation and neutral states for CPs
def calculate_cps_differences(df, property_name, atom1, atom2=None):
    suffix = f"_{atom1}-{atom2}" if atom2 else f"_{atom1}"
    try:
        df[f'{property_name}_(N)-(N-1){suffix}'] = (
            df[f'{property_name}_N{suffix}'] - df[f'{property_name}_N-1{suffix}']
        )
        df[f'{property_name}_(N+1)-(N){suffix}'] = (
            df[f'{property_name}_N+1{suffix}'] - df[f'{property_name}_N{suffix}']
        )
        df[f'{property_name}_(N+1)-(N-1){suffix}'] = (
            df[f'{property_name}_N+1{suffix}'] - df[f'{property_name}_N-1{suffix}']
        )
    except Exception as error:
        pass
        #print(f"Error calculating CP differences for {property_name}{suffix}: {error}")


# Calculate CP differences between states
local_df = pd.read_csv(work_path / 'local_properties.csv')
neighbors_list = [neighbors for _, neighbors in neighbor_dict.items()]

# CP properties
for prop_name in cps_properties_list:
    # Individual atoms
    for atom in atoms_of_interest:
        calculate_cps_differences(local_df, prop_name, atom)
    # Atom pairs
    for i, at1 in enumerate(atoms_of_interest):
        for j, at2 in enumerate(atoms_of_interest):
            if i > j and at2 in neighbors_list[i]:
                calculate_cps_differences(local_df, prop_name, at1, at2)

# Save updated local_properties
save_csv(local_df, "local_properties.csv", work_path)

# Merge of global and local properties
try:
    if (work_path / 'global_properties.csv').exists():
        global_df = pd.read_csv(work_path / 'global_properties.csv')
        global_df['Molecule'] = global_df['Molecule'].astype(str)
    else:
        global_df = pd.DataFrame(columns=['Molecule'])

    if (work_path / 'local_properties.csv').exists():
        local_df = pd.read_csv(work_path / 'local_properties.csv')
        local_df['Molecule'] = local_df['Molecule'].astype(str)
    else:
        local_df = pd.DataFrame(columns=['Molecule'])

    if global_df.empty and not local_df.empty:
        final_df = local_df
    elif local_df.empty and not global_df.empty:
        final_df = global_df
    elif local_df.empty and global_df.empty:
        final_df = pd.DataFrame(columns=['Molecule'])
    else:
        final_df = pd.merge(global_df, local_df, on='Molecule', how='outer')

    print("--> Combined global and local property files in 'properties.csv'")
    save_csv(final_df, 'properties.csv', work_path)
except Exception as e:
    print(f"Error merging global and local properties: {e}")

# Module to calculate molecular descriptors (rdkit, Mordred and PaDEL)
mol_files_folder = os.path.join(work_path, "mol")
mol_files = glob.glob(os.path.join(mol_files_folder, '*.mol'))
results2 = desc.start(mol_files)

if results2:
    descriptors_df = pd.DataFrame(results2)
    if 'Molecule' not in descriptors_df.columns:
        print("Warning: 'Molecule' column not in descriptors. Extracting from filename.")
        descriptors_df['Molecule'] = [Path(f).stem for f in mol_files]
    cols = ['Molecule'] + [col for col in descriptors_df.columns if col != 'Molecule']
    descriptors_df = descriptors_df[cols]
    descriptors_df = descriptors_df.dropna(axis=1, how='any')
    descriptors_df = descriptors_df.loc[:, (descriptors_df.sum() != 0)]
    csv_file_name = 'molecular_descriptors_rdkit_mordred_padel.csv'
    save_csv(descriptors_df, csv_file_name, work_path)
    print(f"Descriptor results saved in '{csv_file_name}'")
    print("\nFirst rows:")
    print(descriptors_df.head())
else:
    print("Could not process any molecules correctly in descriptors.")


if (work_path / 'molecular_descriptors_rdkit_mordred_padel.csv').exists():
    rdkit_df = pd.read_csv(work_path / 'molecular_descriptors_rdkit_mordred_padel.csv')
    rdkit_df['Molecule'] = rdkit_df['Molecule'].astype(str)
    
    if 'Molecule' in final_df.columns:
        final_df['Molecule'] = final_df['Molecule'].astype(str)
        
    final_combined_df = pd.merge(final_df, rdkit_df, on='Molecule', how='outer')
    
    final_combined_df.dropna(axis=1, how='all', inplace=True)
    numeric_cols = final_combined_df.select_dtypes(include=[np.number]).columns
    cols_to_drop = [c for c in numeric_cols if (final_combined_df[c].fillna(0) == 0).all()]
    final_combined_df.drop(columns=cols_to_drop, inplace=True)

    save_csv(final_combined_df, 'properties.csv', work_path)
else:
    print("Skipping merging of RDKit/Mordred descriptors (file not found).")
    save_csv(final_df, 'properties.csv', work_path)

''' Calculation of atomic properties in terms of neighboring atoms '''

def calculate_derived_descriptor(atom, atom_charge, neighbor_charges, atom_label):
    descriptors = {}
    atom = atom_label
    def div(numerator, denominator):
        if pd.isna(numerator) or pd.isna(denominator):
            return np.nan
        try:
            val = float(numerator) / float(denominator) if denominator != 0 else np.nan
            return val
        except (ValueError, TypeError):
            return np.nan

    def get_values(dict_list, key):
        vals = []
        for d in dict_list:
            v = d.get(key)
            if v is None or pd.isna(v) or v == '':
                vals.append(np.nan)
            else:
                try: 
                    vals.append(float(v))
                except ValueError: 
                    vals.append(np.nan)
        return vals

    def get_scalar(d, key):
        v = d.get(key)
        if v is None or pd.isna(v) or v == '':
            return np.nan
        try: return float(v)
        except: return np.nan

    num_neighbors = len(neighbor_charges)

    # Initialize lists
    neighbor_N_charges = get_values(neighbor_charges, 'q(N)')
    f_plus_neighbors = get_values(neighbor_charges, 'f+')
    f_minus_neighbors = get_values(neighbor_charges, 'f-')
    f0_neighbors = get_values(neighbor_charges, 'f0')
    CDD_neighbors = get_values(neighbor_charges, 'CDD')
    
    def safe_sum(dict_list, key):
        return sum(get_values(dict_list, key))

    # Sums of neighbor properties
    sum_q_N_neighbors = sum(neighbor_N_charges)
    sum_f_plus_neighbors = sum(f_plus_neighbors)
    sum_q_Np1_neighbors = safe_sum(neighbor_charges, 'q(N+1)')
    sum_q_Nm1_neighbors = safe_sum(neighbor_charges, 'q(N-1)')
    sum_f_minus_neighbors = sum(f_minus_neighbors)
    sum_CDD_neighbors = sum(CDD_neighbors)
    
    # f+ * f- product
    fprod_neighbors = []
    for v in neighbor_charges:
        vp = get_scalar(v, 'f+')
        vm = get_scalar(v, 'f-')
        fprod_neighbors.append(vp * vm)
    sum_fprod_neighbors = sum(fprod_neighbors)
    
    sum_f0_neighbors = sum(f0_neighbors)

    # Averages of neighbor properties
    avg_q_N_neighbors = np.mean(neighbor_N_charges) if num_neighbors != 0 else np.nan
    avg_f_plus_neighbors = np.mean(f_plus_neighbors) if num_neighbors != 0 else np.nan
    avg_f_minus_neighbors = np.mean(f_minus_neighbors) if num_neighbors != 0 else np.nan
    avg_f0_neighbors = np.mean(f0_neighbors) if num_neighbors != 0 else np.nan
    avg_CDD_neighbors = np.mean(CDD_neighbors) if num_neighbors != 0 else np.nan

    # Standard deviations of neighbor properties
    std_q_N_neighbors = np.std(neighbor_N_charges) if num_neighbors > 1 else np.nan
    std_f_plus_neighbors = np.std(f_plus_neighbors) if num_neighbors > 1 else np.nan
    std_f_minus_neighbors = np.std(f_minus_neighbors) if num_neighbors > 1 else np.nan

    # Maximum and minimum of neighbor properties
    max_f_plus_neighbors = max(f_plus_neighbors) if num_neighbors != 0 else np.nan
    max_f_minus_neighbors = max(f_minus_neighbors) if num_neighbors != 0 else np.nan

    # Atomic values
    atom_qN = get_scalar(atom_charge, 'q(N)')
    atom_qNp1 = get_scalar(atom_charge, 'q(N+1)')
    atom_qNm1 = get_scalar(atom_charge, 'q(N-1)')
    atom_f_plus = get_scalar(atom_charge, 'f+')
    atom_f_minus = get_scalar(atom_charge, 'f-')
    atom_f0 = get_scalar(atom_charge, 'f0')

    atom_cdd = get_scalar(atom_charge, 'CDD')

    # Charge relationships between atom and its neighbors
    descriptors.update({
        f"{atom}_q(N)_div_Sum_q(N)_neighbors": div(atom_qN, sum_q_N_neighbors),
        f"{atom}_q(N)_res_Sum_q(N)_neighbors": atom_qN - sum_q_N_neighbors,
        f"{atom}_q(N+1)_div_Sum_q(N+1)_neighbors": div(atom_qNp1, sum_q_Np1_neighbors),
        f"{atom}_q(N-1)_div_Sum_q(N-1)_neighbors": div(atom_qNm1, sum_q_Nm1_neighbors),
        f"{atom}_Sum_q(N)_neighbors_div_q(N)": div(sum_q_N_neighbors, atom_qN),
        f"{atom}_Sum_q(N+1)_neighbors_div_q(N+1)": div(sum_q_Np1_neighbors, atom_qNp1),
        f"{atom}_Sum_q(N-1)_neighbors_div_q(N-1)": div(sum_q_Nm1_neighbors, atom_qNm1),
    })

    # Fukui function differences between atom and average of neighbors
    descriptors.update({
        f"{atom}_Delta_f+": atom_f_plus - avg_f_plus_neighbors,
        f"{atom}_Delta_f-": atom_f_minus - avg_f_minus_neighbors,
        f"{atom}_Delta_f0": atom_f0 - avg_f0_neighbors,
    })

    # Product of Fukui functions between atom and neighbors
    descriptors.update({
        f"{atom}_f+_atom_x_Sum_f-_neighbors": atom_f_plus * sum_f_minus_neighbors,
        f"{atom}_f-_atom_x_Sum_f+_neighbors": atom_f_minus * sum_f_plus_neighbors,
    })

    # Relationship between CDD and sum of neighbor CDD
    descriptors.update({
        f"{atom}_CDD_div_Sum_CDD_neighbors": div(atom_cdd, sum_CDD_neighbors),
    })

    # Cross-relationships between charges and Fukui functions
    descriptors.update({
        f"{atom}_q(N-1)_div_f+": div(atom_qNm1, atom_f_plus),
        f"{atom}_q(N-1)_div_f-": div(atom_qNm1, atom_f_minus),
        f"{atom}_q(N+1)_div_f+": div(atom_qNp1, atom_f_plus),
        f"{atom}_q(N+1)_div_f-": div(atom_qNp1, atom_f_minus),
    })

    # Sums of neighbor Fukui functions
    descriptors.update({
        f"{atom}_Sum_f+_neighbors": sum_f_plus_neighbors,
        f"{atom}_Sum_f-_neighbors": sum_f_minus_neighbors,
    })

    # Average of neighbor charges
    descriptors.update({
        f"{atom}_Avg_q(N)_neighbors": avg_q_N_neighbors,
    })

    # Ratio of atom Fukui product to sum of neighbor products
    fprod_atom = atom_f_plus * atom_f_minus
    descriptors.update({
        f"{atom}_f+f-_atom_div_Sum_f+f-_neighbors": div(fprod_atom, sum_fprod_neighbors),
    })

    descriptors.update({
        f"{atom}_Std_q(N)_neighbors": std_q_N_neighbors,
        f"{atom}_Std_f+_neighbors": std_f_plus_neighbors,
        f"{atom}_Std_f-_neighbors": std_f_minus_neighbors,
        f"{atom}_Max_f+_neighbors": max_f_plus_neighbors,
        f"{atom}_Max_f-_neighbors": max_f_minus_neighbors,
    })

    # Absolute differences between atom and neighbor averages
    descriptors.update({
        f"{atom}_Abs_Diff_q(N)": abs(atom_qN - avg_q_N_neighbors),
        f"{atom}_Abs_Diff_f+": abs(atom_f_plus - avg_f_plus_neighbors),
        f"{atom}_Abs_Diff_f-": abs(atom_f_minus - avg_f_minus_neighbors),
    })

    # Product of atom charges with average neighbor charges
    descriptors.update({
        f"{atom}_q(N)_x_Avg_q(N)_neighbors": atom_qN * avg_q_N_neighbors,
    })

    # Ratio of atom Fukui functions to neighbor maximums
    descriptors.update({
        f"{atom}_f+_div_Max_f+_neighbors": div(atom_f_plus, max_f_plus_neighbors),
        f"{atom}_f-_div_Max_f-_neighbors": div(atom_f_minus, max_f_minus_neighbors),
    })

    # Number of neighbors
    descriptors.update({
        f"{atom}_Num_neighbors": num_neighbors,
    })

    # Shannon entropy of neighbor charges
    def calculate_entropy(values):
        if any(pd.isna(v) for v in values):
            return np.nan
        total = sum(values)
        if total == 0:
            return np.nan
        probabilities = [v / total for v in values if v != 0]
        return -sum(p * np.log2(p) for p in probabilities)

    entropy_q_N_neighbors = calculate_entropy([abs(v) for v in neighbor_N_charges])
    entropy_f_plus_neighbors = calculate_entropy([abs(v) for v in f_plus_neighbors])
    entropy_f_minus_neighbors = calculate_entropy([abs(v) for v in f_minus_neighbors])

    descriptors.update({
        f"{atom}_Entropy_q(N)_neighbors": entropy_q_N_neighbors,
        f"{atom}_Entropy_f+_neighbors": entropy_f_plus_neighbors,
        f"{atom}_Entropy_f-_neighbors": entropy_f_minus_neighbors,
    })

    # Electronic Proximity Index
    elec_prox = sum(
        atom_f_plus * f_m + atom_f_minus * f_p 
        for f_m, f_p in zip(f_minus_neighbors, f_plus_neighbors)
    )
    descriptors.update({
        f"{atom}_Elec_prox": elec_prox,
    })

    return descriptors

def extract_atomic_properties(base_path, molecule, method, atoms_of_interest, neighbor_dict, atoms_of_interest_labels):
    file_path = base_path / "charges" / f"{molecule}_{method}_charges.csv"

    if not file_path.exists():
        print(f"Warning: File {file_path} not found")
        return None, None

    df = pd.read_csv(file_path)

    properties = {}
    new_descriptors = {}

    for i, atom in enumerate(atoms_of_interest):
        atom_label = atoms_of_interest_labels[i]
        index = int(atom.split('(')[0])
        index = index - 1
        if index < len(df):
            atom_properties = df.iloc[index].to_dict()
            properties[atom] = {
                'properties': atom_properties,
                'neighbors': {}
            }
            neighbors = neighbor_dict[atom]
            neighbor_charges = []

            for neighbor in neighbors:
                neighbor_index = int(neighbor.split('(')[0])
                neighbor_index = neighbor_index - 1
                if neighbor_index <= len(df):
                    neighbor_properties = df.iloc[neighbor_index].to_dict()
                    properties[atom]['neighbors'][neighbor] = neighbor_properties
                    neighbor_properties['Atom'] = neighbor
                    neighbor_charges.append(neighbor_properties)
                else:
                    print(f"Warning: No properties found for neighbor {neighbor} in {file_path}")

            try:
                new_descriptors.update(
                    calculate_derived_descriptor(atom, atom_properties, neighbor_charges, atom_label)
                )
            except Exception:
                pass
        else:
            print(f"Warning: No properties found for atom {atom} in {file_path}")

    return properties, new_descriptors

def calculate_atomic_descriptors(results_dict, charge_methods, atoms_of_interest_labels):
    descriptors = {}
    global_new_descriptors = {}

    for molecule, data in results_dict.items():
        descriptors[molecule] = {}
        global_new_descriptors[molecule] = {}

        for datum in data:
            real_indices = datum['fragment_indices']
            atoms_of_interest = datum['selected_atoms']
            fragment_atoms = datum['fragment_atoms']
            fragment_atoms_formatted = [f"{real_indices[i]}({fragment_atoms[i]})" for i in range(len(atoms_of_interest))]
            neighbor_dict = datum['neighbor_dict']

            for method in charge_methods:
                properties, new_descriptors = extract_atomic_properties(
                    work_path,
                    molecule,
                    method,
                    atoms_of_interest,
                    neighbor_dict, atoms_of_interest_labels)
                if properties:
                    if method not in descriptors[molecule]:
                        descriptors[molecule][method] = []
                    descriptors[molecule][method].append(properties)

                    # Add new descriptors to global dictionary
                    for descriptor, value in new_descriptors.items():
                        global_new_descriptors[molecule][f"{descriptor}_{method}"] = value

    return descriptors, global_new_descriptors

electronic_states = ['N', 'N+1', 'N-1']
charge_methods = [method for (i, method) in method_suffixes]

# Calculate atomic descriptors with charges
atomic_descriptors, new_local_descriptors = calculate_atomic_descriptors(
    results, charge_methods, atoms_of_interest)

new_descriptors_df = pd.DataFrame.from_dict(new_local_descriptors, orient='index')
new_descriptors_df.reset_index(inplace=True)
new_descriptors_df.rename(columns={'index': 'Molecule'}, inplace=True)

# Remove columns that are all zeros
numeric_cols = new_descriptors_df.select_dtypes(include=[np.number]).columns
cols_to_drop = [c for c in numeric_cols if (new_descriptors_df[c].fillna(0) == 0).all()]
new_descriptors_df.drop(columns=cols_to_drop, inplace=True)

save_csv(new_descriptors_df, 'derived_local_descriptors.csv', work_path)

# Load properties file
try:
    derived_local_properties_df = pd.read_csv(work_path / 'derived_local_descriptors.csv')
    properties_df = pd.read_csv(work_path / 'properties.csv')
except FileNotFoundError:
    print("Error loading properties.csv file")
    properties_df = pd.DataFrame({'Molecule': list(results.keys())})

properties_df['Molecule'] = properties_df['Molecule'].astype(str)
derived_local_properties_df['Molecule'] = derived_local_properties_df['Molecule'].astype(str)

# Merge DataFrames
updated_properties_df = pd.merge(properties_df, derived_local_properties_df, on='Molecule', how='left')
# Save updated DataFrame
save_csv(updated_properties_df, 'properties.csv', work_path)

def calculate_G_div_V_ratios_atomic(df: pd.DataFrame, atoms_of_interest: list[str]) -> list[str]:
    """
    For each state and atom of interest, calculate -G(r)/V(r)
    """
    new_cols = []
    for state in ('N', 'N+1', 'N-1'):
        for atom in atoms_of_interest:
            col_G = f"Lagrangian kinetic energy G(r)_{state}_{atom}"
            col_V = f"Potential energy density V(r)_{state}_{atom}"
            if col_G in df.columns and col_V in df.columns:
                denom = df[col_V].replace({0: pd.NA})
                col_ratio = f"G_div_V_{state}_{atom}"
                df[col_ratio] = -df[col_G] / denom
                new_cols.append(col_ratio)
    return new_cols

def calculate_G_div_V_ratios_bond(df: pd.DataFrame, atoms_of_interest: list[str]) -> list[str]:
    """
    For each state and BOND, calculate G(r)/V(r).
    Only considers bonds where BOTH atoms are in atoms_of_interest
    """
    new_cols = []
    # Find all G(r) bond columns
    G_bonds = {}
    
    for col in df.columns:
        m = BOND_PATTERN.match(col)
        if m and m['prop'] == "Lagrangian kinetic energy G(r)":
             # Check if bond is within fragment (both atoms)
             if m['atom1'] in atoms_of_interest and m['atom2'] in atoms_of_interest:
                 key = (m['state'], m['atom1'], m['atom2'])
                 G_bonds[key] = col
    
    for (state, a1, a2), col_G in G_bonds.items():
        # Prop_State_A1-A2
        col_V = f"Potential energy density V(r)_{state}_{a1}-{a2}"
        col_V_rev = f"Potential energy density V(r)_{state}_{a2}-{a1}"
        
        target_V = None
        if col_V in df.columns:
            target_V = col_V
        elif col_V_rev in df.columns:
            target_V = col_V_rev
            
        if target_V:
             denom = df[target_V].replace({0: pd.NA})
             col_ratio = f"G_div_V_{state}_{a1}-{a2}"
             df[col_ratio] = -df[col_G] / denom
             new_cols.append(col_ratio)
             
    return new_cols

# 3(C)_q(N)_Hirshfeld
CHARGE_PATTERN = re.compile(
    r'^(?P<atom_id>\d+\([A-Za-z]+\))_(?P<prop>.+)_(?P<method>[^_]+)$')
# Density of Alpha electrons_N_6(N)
CP_PATTERN = re.compile(
    r'^(?P<prop>.+)_(?P<state>N\+1|N-1|N)_(?P<atom_id>\d+\([A-Za-z]+\))$')
# Density of Alpha electrons_N_1(C)-2(H)
BOND_PATTERN = re.compile(
    r'^(?P<prop>.+)_(?P<state>N\+1|N-1|N|Delta_.+)_(?P<atom1>\d+\([A-Za-z]+\))-(?P<atom2>\d+\([A-Za-z]+\))$')

def calculate_bond_cp_differences(df: pd.DataFrame) -> None:
    """
    Calculate differences for bond CP properties between states:
    (N+1 - N), (N - N-1), (N+1 - N-1)
    """
    # Identify unique bond identifiers and properties
    bond_map = defaultdict(dict)
    
    for col in df.columns:
        m = BOND_PATTERN.match(col)
        if m:
            # Identifier: prop and atoms
            key = (m['prop'], m['atom1'], m['atom2'])
            bond_map[key][m['state']] = col

    # Calculate differences
    for (prop, a1, a2), states in bond_map.items():
        def get_col(s): return states.get(s)
        
        col_N = get_col('N')
        col_Np1 = get_col('N+1')
        col_Nm1 = get_col('N-1')
        
        # N+1 - N
        if col_Np1 and col_N:
            new_col = f"{prop}_Delta_(N+1)-(N)_{a1}-{a2}"
            df[new_col] = df[col_Np1] - df[col_N]
            
        # N - N-1
        if col_N and col_Nm1:
            new_col = f"{prop}_Delta_(N)-(N-1)_{a1}-{a2}"
            df[new_col] = df[col_N] - df[col_Nm1]

        # N+1 - N-1
        if col_Np1 and col_Nm1:
            new_col = f"{prop}_Delta_(N+1)-(N-1)_{a1}-{a2}"
            df[new_col] = df[col_Np1] - df[col_Nm1]

def calculate_fragment_descriptors_csv(work_path, atoms_of_interest):
    csv_path = work_path / "properties.csv"
    df = pd.read_csv(csv_path)
    if df.empty:
        raise ValueError("properties.csv is empty")

    # Calculate Bond CP Differences
    calculate_bond_cp_differences(df)

    try:
        local_csv_path = work_path / "local_properties.csv"
        if local_csv_path.exists():
            local_df = pd.read_csv(local_csv_path)
            calculate_bond_cp_differences(local_df)
            save_csv(local_df, "local_properties.csv", work_path)
    except Exception as e:
        print(f"Error updating local_properties.csv with bond differences: {e}")

    # Sum fragment properties
    column_groups = defaultdict(list)
    for col in df.columns:
        # Check Atomic Charges
        m = CHARGE_PATTERN.match(col)
        if m and m['atom_id'] in atoms_of_interest:
            key = f"sum_frag_{m['prop']}_{m['method']}"
            column_groups[key].append(col)
            continue
        
        # Check Atomic CPs
        m = CP_PATTERN.match(col)
        if m and m['atom_id'] in atoms_of_interest:
            key = f"sum_frag_{m['prop']}_{m['state']}"
            column_groups[key].append(col)
            continue
            
        # Check Bond CPs
        m = BOND_PATTERN.match(col)
        if m:
            # Check if atoms are in the fragment
            if m['atom1'] in atoms_of_interest and m['atom2'] in atoms_of_interest:
                key = f"sum_frag_bond_{m['prop']}_{m['state']}"
                column_groups[key].append(col)
            continue

    for new_col, cols in column_groups.items():
        df[new_col] = df[cols].sum(axis=1, min_count=1)

    # Calculate G/V (Atomic)
    ratio_cols_atomic = calculate_G_div_V_ratios_atomic(df, atoms_of_interest)

    # Calculate G/V (Bond)
    ratio_cols_bond = calculate_G_div_V_ratios_bond(df, atoms_of_interest)
    
    # Sum global G/V by state (Atomic)
    for state in ('N', 'N+1', 'N-1'):
        state_cols = [c for c in ratio_cols_atomic if c.startswith(f"G_div_V_{state}_")]
        if state_cols:
            df[f"sum_frag_G_div_V_{state}"] = df[state_cols].sum(axis=1, min_count=1)

    # Sum global G/V by state (Bond)
    for state in ('N', 'N+1', 'N-1'):
        # G_div_V_{state}_{a1}-{a2}
        state_cols = [c for c in ratio_cols_bond if c.startswith(f"G_div_V_{state}_")]
        if state_cols:
            df[f"sum_frag_bond_G_div_V_{state}"] = df[state_cols].sum(axis=1, min_count=1)

    # Remove columns that are all zeros
    df = df.loc[:, (df != 0).any(axis=0)]
    
    # Save CSV
    save_csv(df, "properties.csv", work_path)

    try:
        derived_csv_path = work_path / "derived_local_descriptors.csv"
        if derived_csv_path.exists():
            derived_df = pd.read_csv(derived_csv_path)
            derived_df['Molecule'] = derived_df['Molecule'].astype(str)
            new_derived_cols = [c for c in df.columns if c.startswith('sum_frag_') or c.startswith('G_div_V_')]
            if 'Molecule' in df.columns:
                df['Molecule'] = df['Molecule'].astype(str)
                subset_new = df[['Molecule'] + new_derived_cols].copy()
                for c in new_derived_cols:
                    if c in derived_df.columns:
                        derived_df.drop(columns=[c], inplace=True)
                
                derived_df = pd.merge(derived_df, subset_new, on='Molecule', how='left')
                numeric_cols = derived_df.select_dtypes(include=[np.number]).columns
                cols_to_drop = [c for c in numeric_cols if (derived_df[c].fillna(0) == 0).all()]
                derived_df.drop(columns=cols_to_drop, inplace=True)
                save_csv(derived_df, "derived_local_descriptors.csv", work_path)
            else:
                print("Warning: 'Molecule' column missing in properties.csv, cannot update derived_local_descriptors.csv")
    except Exception as e:
        print(f"Error updating derived_local_descriptors.csv: {e}")

    # Return fragment results
    output_cols = [c for c in df.columns if c.startswith("sum_frag_")]
    if not df.empty:
      data = df.iloc[0][output_cols].to_dict()
      return data
    return {}

calculate_fragment_descriptors_csv(work_path, atoms_of_interest)

# Create a clean version of properties.csv without any missing data
try:
    final_csv_path = work_path / "properties.csv"
    if final_csv_path.exists():
        final_df_clean = pd.read_csv(final_csv_path)
        # Drop columns that have any NaN values
        final_df_clean = final_df_clean.dropna(axis=1, how='any')
        save_csv(final_df_clean, "properties_clean.csv", work_path)
        print("\nCreated 'properties_clean.csv' containing only columns with complete data.")
except Exception as e:
    print(f"\nError creating properties_clean.csv: {e}")
