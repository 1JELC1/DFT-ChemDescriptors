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
import pandas as pd
import re
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

            # Save sorted DataFrame
            sorted_df.to_csv(filename, index=False)
            # Verify save
            df_check = pd.read_csv(filename)
            df_check = df_check.sort_values(by=['Molecule'])
            df_check.to_csv(work_path/filename, index=False)

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

        # Verify molecules match in both dataframes
        if not global_df['Molecule'].equals(local_df['Molecule']):
            raise ValueError("Molecules in both dataframes don't match or are not in same order.")

        # Merge dataframes using 'Molecule' as key
        combined_df = pd.merge(global_df, local_df, on='Molecule', how='inner')

        # Verify no rows were lost in merge
        if len(combined_df) != len(global_df) or len(combined_df) != len(local_df):
            raise ValueError("Rows were lost during merge. Verify all molecules match.")

        # Remove duplicate columns (except 'Molecule')
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

def get_molecule_name(file_path, extension):
    """Extract molecule name from file path"""
    base_name = os.path.basename(file_path)
    return base_name.replace(extension, "").replace(".fchk", "")

def check_files():
    """Check and categorize FCHK files by charge state"""
    # Ensure FCHK folder exists (create if not)
    if not fchk_folder.exists():
        fchk_folder.mkdir(parents=True, exist_ok=True)

    # List all .fchk files in folder using pathlib
    fchk_files = list(fchk_folder.glob("*.fchk"))

    # Separate files by charge state extension
    neutral_fchk_list = []
    anion_fchk_list = []
    cation_fchk_list = []

    for file in fchk_files:
        if file.stem.endswith(anion_extension):
            anion_fchk_list.append(str(file))
        elif file.stem.endswith(cation_extension):
            cation_fchk_list.append(str(file))
        else:
            neutral_fchk_list.append(str(file))

    # Sort lists alphabetically
    neutral_fchk_list.sort()
    anion_fchk_list.sort()
    cation_fchk_list.sort()

    print(f'Neutral state molecule files: {len(neutral_fchk_list)}')
    print(f'N+1 state molecule files: {len(anion_fchk_list)}')
    print(f'N-1 state molecule files: {len(cation_fchk_list)}')

    # Create dictionaries mapping molecule name to file path
    neutral_molecules = {get_molecule_name(f, neutral_extension): f for f in neutral_fchk_list}
    anion_molecules = {get_molecule_name(f, anion_extension): f for f in anion_fchk_list}
    cation_molecules = {get_molecule_name(f, cation_extension): f for f in cation_fchk_list}

    # Get set of all found molecules
    all_molecules = set(neutral_molecules) | set(anion_molecules) | set(cation_molecules)
    # Complete molecules are those present in all three states
    complete_molecules = set(neutral_molecules) & set(anion_molecules) & set(cation_molecules)
    # Incomplete molecules are the complement of complete within total
    incomplete_molecules = all_molecules - complete_molecules

    if incomplete_molecules:
        print("\nFound molecules with missing files:")
        for mol in sorted(incomplete_molecules):
            print(f"\nMolecule: {mol}")
            if mol not in neutral_molecules:
                print("  Neutral: Missing")
            if mol not in anion_molecules:
                print("  Anion: Missing")
            if mol not in cation_molecules:
                print("  Cation: Missing")

        # If at least one complete molecule exists, offer option to skip incomplete ones
        if complete_molecules:
            option = input("\nContinue only with complete molecules and skip incomplete ones? (Y/N): ")
            if option.lower() == "y":
                # Filter lists to include only complete molecules
                neutral_fchk_list = [neutral_molecules[m] for m in sorted(complete_molecules)]
                anion_fchk_list = [anion_molecules[m] for m in sorted(complete_molecules)]
                cation_fchk_list = [cation_molecules[m] for m in sorted(complete_molecules)]
                return neutral_fchk_list, anion_fchk_list, cation_fchk_list

        print("\nPlease complete missing files or remove incomplete files.")
        input("Press Enter when done to recheck...")
        return None
    else:
        print("\nAll files present for all molecules.")
        return neutral_fchk_list, anion_fchk_list, cation_fchk_list

def charge_calculation(item):
    """Calculate atomic charges for given molecule and method"""
    index, method, suffix = item
    fchk_file = Path(all_fchks[index])
    if suffix == 'EEM':
        if Path(all_fchks[index]).stem.endswith(anion_extension):
            charge = "-1"
        elif Path(all_fchks[index]).stem.endswith(cation_extension):
            charge = "1"
        else:
            charge = "0"
        method = method.replace("charge", charge)

    if not Path(f"{Path(charges_folder, Path(all_fchks[index]).stem)}_{suffix}.chg").exists():
        temp_dir = tempfile.mkdtemp()
        temp_dir_path = Path(temp_dir)
        input_file = temp_dir_path / "inputs.txt"
        try:

            # Execute Multiwfn in temporary directory
            if suffix == 'AIM':
                input_content = f"""{fchk_file}
            17
            {method}
            """
                out_calc = temp_dir_path / f"out_{fchk_file.stem}_{suffix}.chg"
                output = temp_dir_path / f"{fchk_file.stem}.chg"

                with open(input_file, "w") as f:
                    f.write(input_content)

                run_multiwfn_output(input_file, temp_dir_path, out_calc)
                extract_AIM_charges(out_calc, output)
            else:
                file = open(input_file, "w")
                file.write(f"""{str(all_fchks[index])}
                7
                {method}
                """)
                file.close()
                run_multiwfn(input_file, temp_dir_path)

            if Path(temp_dir_path, f"{Path(all_fchks[index]).stem}.chg").exists():

                shutil.move(Path(temp_dir_path, f"{Path(all_fchks[index]).stem}.chg"),
                            Path(charges_folder, Path(f"{Path(charges_folder, Path(all_fchks[index]).stem)}_{suffix}.chg")))
                with print_lock:
                    print(index + 1,
                          f"--> File {Path(all_fchks[index]).stem}_{suffix}.chg generated")
            else:
                print('File not generated')
                errors.append(f"{Path(all_fchks[index]).stem}_{suffix}.chg")
        finally:
            shutil.rmtree(temp_dir)  # Remove temporary directory

    else:
        with print_lock:
            print(index + 1, f"File {Path(all_fchks[index]).stem}_{suffix}.chg already exists")

def generate_and_execute(index, folder):
    """Generate input file and execute CDFT calculation"""
    if not Path(folder, f"{Path(neutral_fchk_list[index]).stem}_CDFT.txt").exists():
        temp_dir = tempfile.mkdtemp()
        try:
            temp_dir_path = Path(temp_dir)
            file = open(temp_dir_path / "inputs.txt", "w")
            file.write(f"""{str(neutral_fchk_list[index])}
22
2
{str(neutral_fchk_list[index])}
{str(anion_fchk_list[index])}
{str(cation_fchk_list[index])}
0
q
""")
            file.close()

            # Execute Multiwfn in temporary directory
            run_multiwfn(temp_dir_path/"inputs.txt", temp_dir_path)

            if (temp_dir_path/"CDFT.txt").exists():
                shutil.move(temp_dir_path/"CDFT.txt",
                            Path(folder, f"{Path(neutral_fchk_list[index]).stem}_CDFT.txt"))
                with print_lock:
                    print(index + 1,
                          f"--> File {Path(folder, Path(neutral_fchk_list[index]).stem).name}_CDFT.txt generated successfully ")
            else:
                errors.append(Path(neutral_fchk_list[index]).stem)
        finally:
            shutil.rmtree(temp_dir)

    else:
        with print_lock:
            print(index + 1, f"File {Path(folder, Path(neutral_fchk_list[index]).stem).name}_CDFT.txt already exists")

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

def create_xyz_input(fchk_path, output_folder):
    """Create input file for XYZ generation"""
    base_name = Path(fchk_path).stem
    input_path = Path(output_folder, f"{base_name}_input.txt")

    with open(input_path, "w") as file:
        file.write(f"""
{str(fchk_path)}
100
2
2

0
q
""")
    return input_path

def generate_xyz_files(neutral_fchk_list, xyz_folder):
    """Generate XYZ files from FCHK files"""
    errors = []

    for fchk_path in neutral_fchk_list:
        xyz_name = f"{Path(fchk_path).stem}.xyz"
        xyz_path = Path(xyz_folder, xyz_name)

        if not xyz_path.exists():
            input_path = create_xyz_input(fchk_path, xyz_folder)
            run_multiwfn(input_path, xyz_folder)

            # Check if XYZ file was generated
            if xyz_path.exists():
                print(f"--> File {xyz_name} generated")
                # Remove temporary input file
                os.remove(input_path)
            else:
                errors.append(Path(fchk_path).stem)

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
      - FCHK files folder (must exist, be directory, contain .fchk files)
      - LOG files folder (optional; if provided, must exist, be directory, contain .log files)
      - Output folder (created if doesn't exist and set as work path)
    """
    # Request FCHK folder path until valid and not empty
    while True:
        fchk_path = input("Enter path to folder containing .fchk files: ").strip()
        fchk_path = Path(fchk_path).expanduser().resolve()

        if not fchk_path.exists() or not fchk_path.is_dir():
            print("Path is not a valid directory. Try again.")
            continue

        # Check for .fchk files in folder
        if not any(fchk_path.glob("*.fchk")):
            print("Directory is empty or contains no .fchk files. Please enter a valid path.")
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

# Request extensions for FCHK files in each state
neutral_extension = input('Enter extension for neutral files (press Enter for default ""): ') or ""
anion_extension = input('Enter extension for anion files (press Enter for default "-ani"): ') or "-ani"
cation_extension = input('Enter extension for cation files (press Enter for default "-cat"): ') or "-cat"

# Verify all corresponding FCHK files are present
result = None
while result is None:
    result = check_files()

neutral_fchk_list, anion_fchk_list, cation_fchk_list = result
errors = []

# Create XYZ and MOL folders
xyz_folder = Path(work_path, "xyz")
xyz_folder.mkdir(parents=True, exist_ok=True)
mol_folder = Path(work_path, "mol")
mol_folder.mkdir(parents=True, exist_ok=True)

# Generate XYZ files
print(f"Generating XYZ files in /{Path(xyz_folder).name} ...")
errors = generate_xyz_files(neutral_fchk_list, xyz_folder)

# Generate MOL files
print(f"Generating MOL files in /{Path(mol_folder).name} ...")
error_molecules = []
xyz_directory = work_path / xyz_folder
# Iterate through each FCHK file in neutral_fchk_list
for fchk_file in neutral_fchk_list:
    name = Path(fchk_file).name
    suffix = f"{neutral_extension}.fchk"
    base_name = name[:-len(suffix)]

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

cdft_folder = Path(work_path, "CDFT")
if not cdft_folder.exists():
    os.makedirs(cdft_folder)

errors = []

# Get number of CPUs and calculate upper limit
num_cpus = os.cpu_count()
max_calculations = max(1, num_cpus // 4)

while True:
    try:
        n = int(input(f"\nEnter number of simultaneous calculations to perform (default 1, maximum {max_calculations})"
                      f"\n(recommended to test with number that doesn't saturate CPU usage over 90%): ") or 1)
        if 0 < n <= max_calculations:
            break
        else:
            print(f"Please enter an integer greater than 0 and less than or equal to {max_calculations}.")
    except ValueError:
        print("Please enter a valid integer.")

print(f"Generating {len(neutral_fchk_list)} global and local CDFT property files in /{Path(cdft_folder).name} ...")
print(f"Performing {n} simultaneous calculations.")

# Create and launch calculation threads
threads = []
for _ in range(n):
    t = threading.Thread(target=calculate_cdft, args=(cdft_folder,))
    t.start()
    threads.append(t)

# Put all tasks in queue
for i in range(len(neutral_fchk_list)):
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

''' Section 4: Extract global reactivity parameters (Extract from CDFT files) '''

# Conversion factor for Hartree to eV energies
fconv = 27.2113862459

print("Extracting global properties and generating .csv file ... ")

def check_logs(logs_folder, neutral_fchk_list, neutral_extension):
    """
    Verify that all corresponding .log files exist in logs folder
    """
    if logs_folder is None or not Path(logs_folder).exists() or not any(Path(logs_folder).glob("*.log")):
        print("Logs folder doesn't exist or contains no .log files")
        return
    logs_path = Path(logs_folder)
    missing_logs = set()
    for fchk_file in neutral_fchk_list:
        fchk_name = Path(fchk_file).name
        suffix = f"{neutral_extension}.fchk"
        if fchk_name.endswith(suffix):
            base_name = fchk_name[:-len(suffix)]
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

    # Append to each list
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
        file_list = list(enumerate(file, start=1))

    # Add molecule name to table
    energies['Molecule'].append(base_name)

    # Extract global reactivity data (units in eV)
    for i in range(1, 16):
        current_line = file_list[-i][1].split()
        if "Nucleophilicity" in current_line:
            energies['Nucleophilicity'].append(current_line[-2])
        elif "Electrophilicity" in current_line:
            energies['Electrophilicity'].append(current_line[-2])
        elif "Softness:" in current_line:
            energies['Softness'].append(current_line[-2])
        elif "Hardness" in current_line:
            energies['Hardness'].append(current_line[-2])
        elif "Chemical" in current_line:
            energies['ChemPotential'].append(current_line[-2])
        elif "Mulliken" in current_line:
            energies['MullikenElectronegativity'].append(current_line[-2])
        elif "EA:" in current_line:
            energies['EA'].append(current_line[-2])
        elif "IP:" in current_line:
            energies['IP'].append(current_line[-2])
        elif "E(N-1):" in current_line:
            energies['E_Cation'].append(float(current_line[-2]) * fconv)
        elif "E(N+1):" in current_line:
            energies['E_Anion'].append(float(current_line[-2]) * fconv)
        elif "E(N):" in current_line:
            energies['E_Neutral'].append(float(current_line[-2]) * fconv)

    # Extract .log information if all logs available
    if logs_found:
        extract_log_data(base_name, logs_folder, fconv, energies)

# Verify if all log files exist
logs_found = check_logs(logs_folder, neutral_fchk_list, neutral_extension)

# Always include properties extracted from CDFT; log properties only added if available
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

# Get base names of FCHK files
valid_bases = set()
for fchk_file in neutral_fchk_list:
    fchk_name = Path(fchk_file).name
    suffix = f"{neutral_extension}.fchk"
    if fchk_name.endswith(suffix):
        base_name = fchk_name[:-len(suffix)]
        valid_bases.add(base_name)

# Process valid CDFT files
for file in Path(cdft_folder).glob("*_CDFT.txt"):
    stem = file.stem
    base = stem.split("_CDFT")[0]
    if base in valid_bases:
        extract_properties(stem, file, logs_folder, fconv, energies, logs_found)

print("Extraction completed.")

print("\nSoftness units are eV^-1, sum_zero-point, dU, dH and dG are eV/particle and others in eV")

try:
    # Try to create DataFrame directly
    global_df = pd.DataFrame(energies)
except ValueError as e:
    print("Error creating DataFrame:", e)
    print("Attempting to align columns and remove rows with incomplete data...")
    global_df = pd.DataFrame({key: pd.Series(value) for key, value in energies.items()})
    global_df.replace("", pd.NA, inplace=True)
    # Remove rows with any NaN values (incomplete data)
    global_df.dropna(inplace=True)

# Convert 'IP' and 'EA' to numeric type
global_df['IP'] = pd.to_numeric(global_df['IP'], errors='coerce')
global_df['EA'] = pd.to_numeric(global_df['EA'], errors='coerce')

# Calculate 'w+' and 'w-'
global_df['w+'] = ((3 * global_df['IP'] + global_df['EA']) ** 2) / (16 * (global_df['IP'] - global_df['EA']))
global_df['w-'] = ((global_df['IP'] + 3 * global_df['EA']) ** 2) / (16 * (global_df['IP'] - global_df['EA']))
print(global_df)

save_csv(global_df, "global_properties.csv", work_path)

''' Section 5: Base fragment search and atoms for property calculation in each molecule '''
# Filter XYZ file list to contain only those corresponding to neutral FCHK files
expected_xyz_names = set()
for fchk in neutral_fchk_list:
    # Extract full name without extension: {molecule_name}{neutral_extension}
    base = Path(fchk).stem
    # Remove part corresponding to neutral_extension to get {molecule_name}
    if neutral_extension and base.endswith(neutral_extension):
        base_name = base[:-len(neutral_extension)]
        # If removing neutral extension results in empty, keep original name
        if not base_name:
            base_name = base
    else:
        base_name = base
    expected_xyz_names.add(f"{base_name}.xyz")

# Filter XYZ files in folder that are in expected_xyz_names
xyz_files_list = [i for i in Path(xyz_folder).glob("*.xyz") if i.name in expected_xyz_names]

# Check for missing XYZ files
missing_xyz = expected_xyz_names - {i.name for i in xyz_files_list}
if missing_xyz:
    print("Warning: Missing following XYZ files:", ", ".join(sorted(missing_xyz)))

while True:
    choice = input(
        f"\nTo extract local properties, enter filename of one of your {len(neutral_fchk_list)} molecules "
        f"to select the fragment common to all your molecules (d = choose first in list): ")
    if choice == 'd':
        first_base = Path(neutral_fchk_list[0]).stem
        if neutral_extension and first_base.endswith(neutral_extension):
            first_base = first_base[:-len(neutral_extension)]
        file = Path(xyz_folder) / f"{first_base}.xyz"
        print(f"\nCommon fragment and atoms for property extraction will be selected using molecule {first_base}")
        break
    else:
        file = Path(xyz_folder) / f"{choice}.xyz"
        if file.name in [f.name for f in xyz_files_list]:
            break
        else:
            print("\nEntered file not found in your xyz folder, choose another")

# Define fragment search specificity level
specificity = "0"
results, atoms_of_interest, neighbor_dict = Ff.start(file, specificity)

print(f"ATOMS OF INTEREST: {atoms_of_interest}")
# Keep only first match per molecule
for key, matches in results.items():
    if len(matches) > 1:
        results[key] = [matches[0]]

''' Section 6: Extraction and calculation of local properties (Extract from CDFT files) '''

cdft_files_list = [path for path in Path(cdft_folder).glob("*_CDFT.txt")]

# Initialize dictionary to store data
atom_properties = []
for file, match_list in results.items():
    # Verify corresponding CDFT file exists
    if Path(cdft_folder, f"{file}_CDFT.txt") in cdft_files_list:
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
                        if parts and atom.startswith(parts[0]):
                            if count == 0:
                                # Atomic properties
                                pass
                            elif count == 1:
                                properties.update(
                                    {f'{atoms_of_interest[i]}_Electrophilicity_Hirshfeld': parts[-2],
                                     f'{atoms_of_interest[i]}_Nucleophilicity_Hirshfeld': parts[-1]}
                                )
                                count += 1
                            elif count == 2:
                                properties.update(
                                    {f'{atoms_of_interest[i]}_s-_Hirshfeld': parts[-5],
                                     f'{atoms_of_interest[i]}_s+_Hirshfeld': parts[-4],
                                     f'{atoms_of_interest[i]}_s0_Hirshfeld': parts[-3],
                                     f'{atoms_of_interest[i]}_s+/s-_Hirshfeld': parts[-2],
                                     f'{atoms_of_interest[i]}_s-/s+_Hirshfeld': parts[-1]}
                                )
                                count += 1
                                break
        atom_properties.append(properties)
    else:
        print(f'File {file}_CDFT.txt does not exist')

# Print dictionary with data and save to CSV
local_df = pd.DataFrame(atom_properties)
print("\nExtracted local properties:")
print(local_df)

save_csv(local_df, "local_properties.csv", work_path)


# Merge global and local properties

try:
    global_df = pd.read_csv('global_properties.csv')
    local_df = pd.read_csv('local_properties.csv')
except Exception as e:
    raise RuntimeError("Error reading 'global_properties.csv' or 'local_properties.csv'") from e

combined_df = merge_dataframes(global_df, local_df)
if combined_df is not None:
    save_csv(combined_df, 'properties.csv', work_path)
    print("Merged global and local properties saved in 'properties.csv'")


# Section B: Atomic charge calculation and CPs file extraction

# List of available methods (those in available_method_suffixes)
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

# List of all FCHK files (neutral, anions, cations)
all_fchks = neutral_fchk_list + anion_fchk_list + cation_fchk_list

print(f"Calculating charges for {len(method_suffixes)} method(s)...")

# Create task queue for calculations
task_queue = queue.Queue()
threads = []
n = 8
for _ in range(n):
    t = threading.Thread(target=calculate_charges)
    t.start()
    threads.append(t)
for method, suffix in method_suffixes:
    for i in range(len(all_fchks)):
        task_queue.put((i, method, suffix))
task_queue.join()
for _ in range(n):
    task_queue.put(None)
for t in threads:
    t.join()

print(f"Errors in charge calculation: {errors}")

def extract_charges(suffix):
    """Extract charges and save to CSV"""
    for fchk in neutral_fchk_list:
        if neutral_extension:
            parts = Path(fchk).stem.split(neutral_extension)
            base_name = parts[0] if parts[0] else Path(fchk).stem
        else:
            base_name = Path(fchk).stem

        neutral_file = os.path.join(charges_folder, f"{base_name}{neutral_extension}_{suffix}.chg")
        anion_file = os.path.join(charges_folder, f"{base_name}{anion_extension}_{suffix}.chg")
        cation_file = os.path.join(charges_folder, f"{base_name}{cation_extension}_{suffix}.chg")

        if os.path.exists(neutral_file) and os.path.exists(anion_file) and os.path.exists(cation_file):
            try:
                with open(neutral_file, 'r') as file_neutral, \
                        open(anion_file, 'r') as file_anion, \
                        open(cation_file, 'r') as file_cation, \
                        open(os.path.join(charges_folder, f"{base_name}_{suffix}_charges.csv"), 'w',
                             newline='') as csvfile:

                    csv_writer = csv.writer(csvfile)
                    csv_writer.writerow(["Atom", "q(N)", "q(N+1)", "q(N-1)"])
                    lines_neutral = file_neutral.readlines()
                    lines_anion = file_anion.readlines()
                    lines_cation = file_cation.readlines()

                    for ln, la, lc in zip(lines_neutral, lines_anion, lines_cation):
                        if suffix == "AIM":
                            neutral_charge = float(ln.split()[-1])
                            anion_charge = float(la.split()[-1])
                            cation_charge = float(lc.split()[-1])
                        else:
                            pattern = r'(\w+)\s+([-]?\d+\.\d+)\s+([-]?\d+\.\d+)\s+([-]?\d+\.\d+)\s*([-]?\d+\.\d+)'
                            match1 = re.match(pattern, ln)
                            match2 = re.match(pattern, la)
                            match3 = re.match(pattern, lc)
                            if match1 and match2 and match3:
                                atom, x, y, z, charge = match1.groups()
                                atom2, x2, y2, z2, charge2 = match2.groups()
                                atom3, x3, y3, z3, charge3 = match3.groups()
                                neutral_charge = float(charge)
                                anion_charge = float(charge2)
                                cation_charge = float(charge3)
                            else:
                                raise ValueError(f"Could not extract data from:\n{ln}\n{la}\n{lc}")
                        atom = ln.split()[0]
                        csv_writer.writerow([atom, neutral_charge, anion_charge, cation_charge])
            except Exception as e:
                print(f"Error processing charges for {base_name} with suffix {suffix}: {e}")
        else:
            print(f"Missing charge files for {base_name} with suffix {suffix}")

for method, suffix in method_suffixes:
    extract_charges(suffix)

# Calculate local Fukui descriptors in each charge file
charge_csv_files = glob.glob(os.path.join(charges_folder, '*.csv'))
for file in charge_csv_files:
    try:
        df = pd.read_csv(file)
        df['f-'] = abs(df['q(N)'] - df['q(N-1)'])
        df['f+'] = abs(df['q(N+1)'] - df['q(N)'])
        df['CDD'] = df['f+'] - df['f-']
        df['f0'] = (df['f-'] + df['f+']) * 0.5
        df['f+f-'] = df['f+'] * df['f-']
        df['q(N)/f+'] = df['q(N)'] / df['f+']
        df['q(N)/CDD'] = df['q(N)'] / abs(df['f+'] - df['f-'])
        df['q(N)/f-'] = df['q(N)'] / df['f-']
        df['q(N+1)/f+'] = df['q(N+1)'] / df['f+']
        df['q(N+1)/CDD'] = df['q(N+1)'] / abs(df['f+'] - df['f-'])
        df['q(N+1)/f-'] = df['q(N+1)'] / df['f-']
        df['q(N-1)/f+'] = df['q(N-1)'] / df['f+']
        df['q(N-1)/CDD'] = df['q(N-1)'] / abs(df['f+'] - df['f-'])
        df['q(N-1)/f-'] = df['q(N-1)'] / df['f-']
        # Save directly to same file
        df.to_csv(file, index=False)
    except Exception as e:
        print(f"Error calculating descriptors in {file}: {e}")

# Section C: Extract local properties from charge files

# Extract properties for each method for atoms of interest, merging by 'Molecule' key
for input_str, method in method_suffixes:
    atom_properties = []
    for file, match_list in results.items():
        # Extract base name from FCHK using neutral_extension
        if neutral_extension:
            parts = Path(file).stem.split(neutral_extension)
            base = parts[0] if parts[0] else Path(file).stem
        else:
            base = Path(file).stem

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
                        # Iterate through all columns except first (Atom)
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
local_combined_df = pd.read_csv('local_properties.csv')
for input_str, method in method_suffixes:
    try:
        method_df = pd.read_csv(f'local_properties_{method}.csv')
        local_combined_df = pd.merge(local_combined_df, method_df, on='Molecule', how='outer')
    except Exception as e:
        print(f"Error merging properties for method {method}: {e}")

local_combined_df = local_combined_df.dropna(axis=1, how='any')
local_combined_df = local_combined_df.loc[:, (local_combined_df.sum() != 0)]
save_csv(local_combined_df, "local_properties.csv", work_path)

''' Calculate atomic and bond critical points (CPs) for each state (N, N+1, N-1) '''

# Create folder to store CP property files
cps_folder = Path(work_path, "cps")
cps_folder.mkdir(exist_ok=True)

# List of properties to extract from CPs
cps_properties_list = [
    'Density of all electrons', 'Density of Alpha electrons', 'Density of Beta electrons',
    'Lagrangian kinetic energy G(r)', 'Hamiltonian kinetic energy K(r)', 'Potential energy density V(r)',
    'Energy density E(r) or H(r)', 'Laplacian of electron density', 'Electron localization function (ELF)',
    'Localized orbital locator (LOL)', 'Local information entropy', 'Interaction region indicator (IRI)',
    'Reduced density gradient (RDG)', 'Reduced density gradient with promolecular approximation',
    'Sign(lambda2)*rho', 'Sign(lambda2)*rho with promolecular approximation', 'Average local ionization energy (ALIE)',
    'Delta-g (under promolecular approximation)', 'Delta-g (under Hirshfeld partition)', 'ESP from nuclear charges',
    'ESP from electrons', 'Total ESP'
]

print("Calculating atomic and bond CP properties for N, N+1 and N-1 states")

def calculate_cps(name, atom_indices, selected_atoms, neighbor_dict):
    """
    Calculate and extract atomic and bond CP properties for molecule 'name'
    using Multiwfn. Generate two file types:
      - {name}_cps_atomic.txt (atomic CPs, type "2")
      - {name}_cps_{atom1}-{atom2}_bond.txt (bond CPs, type "3")
    """
    molecule_file = Path(fchk_folder, f"{name}.fchk")
    if not molecule_file.exists():
        raise FileNotFoundError(f"FCHK file not found: {molecule_file}")

    print(f"Processing {name}")
    # Convert indices to strings and join
    indices_str = [str(i) for i in atom_indices]
    nuclear_indices = ",".join(indices_str)

    def write_input(indices, cp_type, input_path):
        with open(input_path, "w") as f:
            f.write(f"""{molecule_file}
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

def process_molecule(file, match_list, neutral_extension, fchk_folder, cps_folder, atoms_of_interest,
                      cps_properties_list, calculation_extension):
    """
    Process molecule in given state (ext_calculo = neutral_extension, anion_extension, cation_extension).
    """
    # Adjust base name
    if neutral_extension:
        file_base = "".join(Path(file).stem.split(neutral_extension)[:-1])
        if not file_base:
            file_base = Path(file).stem
    else:
        file_base = Path(file).stem

    molecule_name = file_base
    molecule_properties = []

    for element in match_list:
        atom_indices = element['interest_atom_indices']
        neighbor_dict = element['neighbor_dict']
        selected_atoms = element['selected_atoms']

        # Calculate and extract CPs
        try:
            calculate_cps(f"{file_base}{calculation_extension}", atom_indices, selected_atoms, neighbor_dict)
        except Exception as e:
            print(f"Error calculating CPs for {file_base}{calculation_extension}: {e}")

        # Initialize dictionary for this molecule
        props = {'Molecule': molecule_name}

        # Determine state for property labeling
        if calculation_extension == neutral_extension:
            state = 'N'
        elif calculation_extension == anion_extension:
            state = 'N+1'
        elif calculation_extension == cation_extension:
            state = 'N-1'
        else:
            state = 'N'  # Default

        # Extract properties from atomic file
        atomic_cp_file = cps_folder / f"{file_base}{calculation_extension}_cps_atomic.txt"
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

# 1) Calculate atomic and bond CP properties for neutral state
cps_atom_properties = []
processes = 8
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

# 2) Calculate atomic and bond CP properties for anion state
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

# 3) Calculate atomic and bond CP properties for cation state
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
local_df = pd.read_csv('local_properties.csv')
neutral_atom_df = pd.read_csv('local_properties_cps_atoms_N.csv')
anion_atom_df = pd.read_csv('local_properties_cps_atoms_N+1.csv')
cation_atom_df = pd.read_csv('local_properties_cps_atoms_N-1.csv')

local_df = merge_dataframes(local_df, neutral_atom_df)
local_df = merge_dataframes(local_df, anion_atom_df)
local_df = merge_dataframes(local_df, cation_atom_df)
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
        # example: name_cps_At1-At2_bond.txt --> At1-At2
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

        # Check if dictionary already exists for this molecule
        existing = next((x for x in molecule_data if x['Molecule'] == molecule_name), None)
        if existing:
            existing.update(properties2)
        else:
            molecule_data.append(properties2)

    return molecule_data

# Extract bond properties for N, N+1 and N-1
for state, fname in [("N", "local_properties_cps_bond_N.csv"),
                      ("N+1", "local_properties_cps_bond_N+1.csv"),
                      ("N-1", "local_properties_cps_bond_N-1.csv")]:
    props = extract_bond_cps_properties(cps_folder, cps_properties_list, state)
    bond_df = pd.DataFrame(props)
    save_csv(bond_df, fname, work_path)

# Merge with local_properties
local_df = pd.read_csv(work_path / 'local_properties.csv')
local_df = local_df.dropna(axis=1, how='any')
neutral_bond_df = pd.read_csv(work_path / 'local_properties_cps_bond_N.csv')
anion_bond_df = pd.read_csv(work_path / 'local_properties_cps_bond_N+1.csv')
cation_bond_df = pd.read_csv(work_path / 'local_properties_cps_bond_N-1.csv')

local_df = merge_dataframes(local_df, neutral_bond_df)
local_df = merge_dataframes(local_df, anion_bond_df)
local_df = merge_dataframes(local_df, cation_bond_df)

# Remove columns with incomplete or empty data
local_df = local_df.dropna(axis=1, how='any')
local_df = local_df.loc[:, (local_df.sum() != 0)]
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
local_df = pd.read_csv('local_properties.csv')
neighbors_list = [neighbors for _, neighbors in neighbor_dict.items()]

# For CP properties
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

global_df = pd.read_csv('global_properties.csv')
local_df = pd.read_csv('local_properties.csv')
final_df = pd.merge(global_df, local_df, on='Molecule', how='outer')
print("--> Combined global and local property files in 'properties.csv'")
save_csv(final_df, 'properties.csv', work_path)

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

rdkit_df = pd.read_csv('molecular_descriptors_rdkit_mordred_padel.csv')
final_combined_df = pd.merge(final_df, rdkit_df, on='Molecule', how='outer')
final_combined_df = final_combined_df.dropna(axis=1, how='any')
final_combined_df = final_combined_df.loc[:, (final_combined_df.sum() != 0)]
save_csv(final_combined_df, 'properties.csv', work_path)

''' Calculation of atomic properties in terms of neighboring atoms '''

def calculate_new_descriptor(atom, atom_charge, neighbor_charges, atom_label):
    descriptors = {}
    atom = atom_label
    # Avoid division by zero
    def div(numerator, denominator):
        return numerator / denominator if denominator != 0 else 0.0

    num_neighbors = len(neighbor_charges)

    # Initialize lists
    neighbor_N_charges = [v['q(N)'] for v in neighbor_charges]
    f_plus_neighbors = [v['f+'] for v in neighbor_charges]
    f_minus_neighbors = [v['f-'] for v in neighbor_charges]
    f0_neighbors = [v['f0'] for v in neighbor_charges]
    CDD_neighbors = [v['CDD'] for v in neighbor_charges]

    # Sums of neighbor properties
    sum_q_N_neighbors = sum(neighbor_N_charges)
    sum_f_plus_neighbors = sum(f_plus_neighbors)
    sum_q_Np1_neighbors = sum(v['q(N+1)'] for v in neighbor_charges)
    sum_q_Nm1_neighbors = sum(v['q(N-1)'] for v in neighbor_charges)
    sum_f_minus_neighbors = sum(f_minus_neighbors)
    sum_CDD_neighbors = sum(CDD_neighbors)
    sum_fprod_neighbors = sum(v['f+'] * v['f-'] for v in neighbor_charges)
    sum_f0_neighbors = sum(f0_neighbors)

    # Averages of neighbor properties
    avg_q_N_neighbors = np.mean(neighbor_N_charges) if num_neighbors != 0 else 0.0
    avg_f_plus_neighbors = np.mean(f_plus_neighbors) if num_neighbors != 0 else 0.0
    avg_f_minus_neighbors = np.mean(f_minus_neighbors) if num_neighbors != 0 else 0.0
    avg_f0_neighbors = np.mean(f0_neighbors) if num_neighbors != 0 else 0.0
    avg_CDD_neighbors = np.mean(CDD_neighbors) if num_neighbors != 0 else 0.0

    # Standard deviations of neighbor properties
    std_q_N_neighbors = np.std(neighbor_N_charges) if num_neighbors > 1 else 0.0
    std_f_plus_neighbors = np.std(f_plus_neighbors) if num_neighbors > 1 else 0.0
    std_f_minus_neighbors = np.std(f_minus_neighbors) if num_neighbors > 1 else 0.0

    # Maximum and minimum of neighbor properties
    max_f_plus_neighbors = max(f_plus_neighbors) if num_neighbors != 0 else 0.0
    max_f_minus_neighbors = max(f_minus_neighbors) if num_neighbors != 0 else 0.0

    # Charge relationships between atom and its neighbors
    descriptors.update({
        f"{atom}_q(N)_div_Sum_q(N)_neighbors": div(atom_charge['q(N)'], sum_q_N_neighbors),
        f"{atom}_q(N)_res_Sum_q(N)_neighbors": atom_charge['q(N)'] - sum_q_N_neighbors,
        f"{atom}_q(N+1)_div_Sum_q(N+1)_neighbors": div(atom_charge['q(N+1)'], sum_q_Np1_neighbors),
        f"{atom}_q(N-1)_div_Sum_q(N-1)_neighbors": div(atom_charge['q(N-1)'], sum_q_Nm1_neighbors),
        f"{atom}_Sum_q(N)_neighbors_div_q(N)": div(sum_q_N_neighbors, atom_charge['q(N)']),
        f"{atom}_Sum_q(N+1)_neighbors_div_q(N+1)": div(sum_q_Np1_neighbors, atom_charge['q(N+1)']),
        f"{atom}_Sum_q(N-1)_neighbors_div_q(N-1)": div(sum_q_Nm1_neighbors, atom_charge['q(N-1)']),
    })

    # Fukui function differences between atom and average of neighbors
    descriptors.update({
        f"{atom}_Delta_f+": atom_charge['f+'] - avg_f_plus_neighbors,
        f"{atom}_Delta_f-": atom_charge['f-'] - avg_f_minus_neighbors,
        f"{atom}_Delta_f0": atom_charge['f0'] - avg_f0_neighbors,
    })

    # Product of Fukui functions between atom and neighbors
    descriptors.update({
        f"{atom}_f+_atom_x_Sum_f-_neighbors": atom_charge['f+'] * sum_f_minus_neighbors,
        f"{atom}_f-_atom_x_Sum_f+_neighbors": atom_charge['f-'] * sum_f_plus_neighbors,
    })

    # Relationship between CDD and sum of neighbor CDD
    descriptors.update({
        f"{atom}_CDD_div_Sum_CDD_neighbors": div(atom_charge['CDD'], sum_CDD_neighbors),
    })

    # Cross-relationships between charges and Fukui functions
    descriptors.update({
        f"{atom}_q(N)_div_f+": div(atom_charge['q(N)'], atom_charge['f+']),
        f"{atom}_q(N)_div_f-": div(atom_charge['q(N)'], atom_charge['f-']),
        f"{atom}_q(N-1)_div_f+": div(atom_charge['q(N-1)'], atom_charge['f+']),
        f"{atom}_q(N-1)_div_f-": div(atom_charge['q(N-1)'], atom_charge['f-']),
        f"{atom}_q(N+1)_div_f+": div(atom_charge['q(N+1)'], atom_charge['f+']),
        f"{atom}_q(N+1)_div_f-": div(atom_charge['q(N+1)'], atom_charge['f-']),
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
    fprod_atom = atom_charge['f+'] * atom_charge['f-']
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
        f"{atom}_Abs_Diff_q(N)": abs(atom_charge['q(N)'] - avg_q_N_neighbors),
        f"{atom}_Abs_Diff_f+": abs(atom_charge['f+'] - avg_f_plus_neighbors),
        f"{atom}_Abs_Diff_f-": abs(atom_charge['f-'] - avg_f_minus_neighbors),
    })

    # Product of atom charges with average neighbor charges
    descriptors.update({
        f"{atom}_q(N)_x_Avg_q(N)_neighbors": atom_charge['q(N)'] * avg_q_N_neighbors,
    })

    # Ratio of atom Fukui functions to neighbor maximums
    descriptors.update({
        f"{atom}_f+_div_Max_f+_neighbors": div(atom_charge['f+'], max_f_plus_neighbors),
        f"{atom}_f-_div_Max_f-_neighbors": div(atom_charge['f-'], max_f_minus_neighbors),
    })

    # Number of neighbors
    descriptors.update({
        f"{atom}_Num_neighbors": num_neighbors,
    })

    # Entropy of neighbor charges
    def calculate_entropy(values):
        total = sum(values)
        if total == 0:
            return 0.0
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
        atom_charge['f+'] * v['f-'] + atom_charge['f-'] * v['f+'] for v in neighbor_charges
    )
    descriptors.update({
        f"{atom}_Elec_prox": elec_prox,
    })

    # Return descriptors
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

            new_descriptors.update(
                calculate_new_descriptor(atom, atom_properties, neighbor_charges, atom_label)
            )
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

# Define electronic states handled
electronic_states = ['N', 'N+1', 'N-1']

# Assuming 'results' and 'method_suffixes' are defined
charge_methods = [method for (i, method) in method_suffixes]

# Calculate atomic descriptors with charges
atomic_descriptors, new_local_descriptors = calculate_atomic_descriptors(
    results, charge_methods, atoms_of_interest)

new_descriptors_df = pd.DataFrame.from_dict(new_local_descriptors, orient='index')
new_descriptors_df.reset_index(inplace=True)
new_descriptors_df.rename(columns={'index': 'Molecule'}, inplace=True)
save_csv(new_descriptors_df, 'derived_local_descriptors.csv', work_path)

# Load properties file
try:
    derived_local_properties_df = pd.read_csv(work_path / 'derived_local_descriptors.csv')
    properties_df = pd.read_csv(work_path / 'properties.csv')
except FileNotFoundError:
    print("Error loading properties.csv file")
    properties_df = pd.DataFrame({'Molecule': list(results.keys())})

# Verify congruence in row count (as strings)
properties_df['Molecule'] = properties_df['Molecule'].astype(str)
derived_local_properties_df['Molecule'] = derived_local_properties_df['Molecule'].astype(str)

# Merge DataFrames using 'Molecule' as key
updated_properties_df = pd.merge(properties_df, derived_local_properties_df, on='Molecule', how='left')
# Save updated DataFrame
save_csv(updated_properties_df, 'properties.csv', work_path)

def calculate_G_div_V_ratios_atomic(df: pd.DataFrame, atoms_of_interest: list[str]) -> list[str]:
    """
    For each state and atom of interest, calculate G(r)/V(r), create column
    "G_div_V_<state>_<atom>" in DataFrame and return list of new columns.
    """
    new_cols = []
    for state in ('N', 'N+1', 'N-1'):
        for atom in atoms_of_interest:
            col_G = f"Lagrangian kinetic energy G(r)_{state}_{atom}"
            col_V = f"Potential energy density V(r)_{state}_{atom}"
            if col_G in df.columns and col_V in df.columns:
                denom = df[col_V].replace({0: pd.NA})
                col_ratio = f"G_div_V_{state}_{atom}"
                df[col_ratio] = df[col_G] / denom
                new_cols.append(col_ratio)
    return new_cols

#   3(C)_q(N)_Hirshfeld
CHARGE_PATTERN = re.compile(
    r'^(?P<atom_id>\d+\([A-Za-z]+\))_(?P<prop>.+)_(?P<method>[^_]+)$')
#   Density of Alpha electrons_N_6(N)
CP_PATTERN = re.compile(
    r'^(?P<prop>.+)_(?P<state>N\+1|N-1|N)_(?P<atom_id>\d+\([A-Za-z]+\))$')

def calculate_fragment_descriptors_csv(work_path, atoms_of_interest):
    csv_path = work_path / "properties.csv"
    df = pd.read_csv(csv_path)
    if df.empty:
        raise ValueError("properties.csv is empty")

    # Sum fragment properties
    column_groups = defaultdict(list)
    for col in df.columns:
        m = CHARGE_PATTERN.match(col)
        if m and m['atom_id'] in atoms_of_interest:
            key = f"sum_frag_{m['prop']}_{m['method']}"
            column_groups[key].append(col)
            continue
        m = CP_PATTERN.match(col)
        if m and m['atom_id'] in atoms_of_interest:
            key = f"sum_frag_{m['prop']}_{m['state']}"
            column_groups[key].append(col)
    for new_col, cols in column_groups.items():
        df[new_col] = df[cols].sum(axis=1)

    # Calculate G/V
    ratio_cols = calculate_G_div_V_ratios_atomic(df, atoms_of_interest)

    # Sum global G/V by state
    for state in ('N', 'N+1', 'N-1'):
        state_cols = [c for c in ratio_cols if c.startswith(f"G_div_V_{state}_")]
        if state_cols:
            df[f"sum_frag_G_div_V_{state}"] = df[state_cols].sum(axis=1)

    # Save CSV
    save_csv(df, "properties.csv", work_path)

    # Return fragment results
    output_cols = [c for c in df.columns if c.startswith("sum_frag_")]
    data = df.iloc[0][output_cols].to_dict()
    return data

calculate_fragment_descriptors_csv(work_path, atoms_of_interest)
