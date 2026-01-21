"""
Fragment Finder

This module provides utilities to identify and extract a common fragment from a set of
molecules.  From `.xyz` file and a specificity level, it identifies
atom indices belonging to a fragment that is shared among molecules in the same
directory.
The workflow broadly consists of:

1. Reading molecules from `.xyz` files and computing their chemical connectivity.
2. Selecting a fragment interactively from a reference molecule using a 3D viewer.
3. Finding matches of this fragment in other molecules by graph isomorphism.
4. Reporting atoms of interest and their neighbors for downstream calculations.
"""

import os
from pathlib import Path
import numpy as np
import networkx as nx
from networkx.algorithms import isomorphism
from ase.io import read
from vedo import Sphere, Tube, Plotter, Text3D, Assembly, Text2D
import vedo
from collections import Counter, defaultdict
from ase.neighborlist import NeighborList, natural_cutoffs

# Section 1: Connectivity and Graph
import numpy as np
from ase.data import covalent_radii, atomic_numbers

# Default maximum valences
# Elements not in this list will default to 7 (accommodating octahedral structures)
MAX_VALENCE_DEF = {
    'H': 1, 'C': 4, 'N': 4, 'O': 3,
    'F': 1, 'Cl': 1, 'Br': 1, 'I': 1,
    'P': 5, 'S': 6, 'B': 4, 'Si': 4,
    'Fe': 6, 'Co': 6, 'Ni': 6, 'Cu': 6, 'Zn': 4,
    'Pd': 4, 'Pt': 6, 'Au': 4, 'Hg': 4, 'Al': 6,
}

def calculate_connectivity_matrix(
    mol,
    mult: float = 1.25,
    max_valence: dict | None = None,
    allow_HH: bool = False,
    iter_max: int = 4,
    debug: bool = False,
    base_scale: float = 1.10,
    pair_scale: dict | None = None,
):
    """
    Computes the connectivity matrix using a robust strategy:
    1. Broad Phase: Detect all potential bonds using a generous distance threshold
       (natural_cutoffs * mult). This captures elongated bonds.
    2. Pruning Phase: Enforce maximum valence constraints by keeping the strongest
       (shortest) bonds and removing the excess.
    """
    if max_valence is None:
        max_valence = MAX_VALENCE_DEF

    cutoffs = natural_cutoffs(mol, mult=mult)
    
    S = mol.get_chemical_symbols()
    n = len(mol)

    D = mol.get_all_distances(mic=True)
    np.fill_diagonal(D, np.inf)

    A = np.zeros((n, n), dtype=int)
    
    # 1. Broad Detection Phase
    for i in range(n):
        ri = cutoffs[i]
        si = S[i]
        for j in range(i+1, n):
            sj = S[j]
            # skip H-H contacts unless allowed
            if not allow_HH and si == 'H' and sj == 'H':
                continue
            
            rj = cutoffs[j]
            # Threshold is sum of cutoff radii (which already include scale)
            threshold = ri + rj
            
            if D[i, j] <= threshold:
                A[i, j] = 1
                A[j, i] = 1

    # 2. Pruning Phase (Valence Enforcement)
    for _ in range(iter_max):
        changes = 0
        for i in range(n):
            vmax = max_valence.get(S[i], 7)
            neighbors = np.where(A[i] == 1)[0]
            deg = len(neighbors)
            if deg > vmax:
                # sort neighbors by increasing distance
                order = neighbors[np.argsort(D[i, neighbors])]
                # We keep the 'vmax' closest ones.
                to_remove = order[vmax:]
                
                for j in to_remove:
                    if A[i, j] == 1:
                        A[i, j] = 0
                        A[j, i] = 0
                        changes += 1
                        
                if debug and len(to_remove) > 0:
                    kept = order[:vmax]
                    print(f"[VALENCE PRUNING] {S[i]}{i+1}: deg={deg} > max={vmax}. "
                          f"Retaining nearest: {[k+1 for k in kept]}, "
                          f"Removing furthest: {[k+1 for k in to_remove]}")
        
        if changes == 0:
            break

    return A

def matrix_to_graph(matrix, symbols):
    """Convert a connectivity matrix and a list of symbols into a NetworkX graph."""
    G = nx.Graph()
    n = len(matrix)
    for i in range(n):
        G.add_node(i, label=symbols[i])
    for i in range(n):
        for j in range(i + 1, n):
            if matrix[i, j] == 1:
                G.add_edge(i, j)
    return G

# Section 2: Fragmentation and Matching
def remove_duplicate_matches(matches: list[list[int]]) -> list[list[int]]:
    """
    Remove duplicate fragment matches regardless of ordering.
    """
    unique_matches: list[list[int]] = []
    seen: set[frozenset[int]] = set()
    for match in matches:
        key = frozenset(match)
        if key not in seen:
            seen.add(key)
            unique_matches.append(match)
    return unique_matches


def match_fragment(molecule_matrix: np.ndarray, fragment_matrix: np.ndarray,
                   molecule_symbols: list[str], fragment_symbols: list[str]) -> list[list[int]]:
    """
    Find all occurrences of a fragment within a molecule using graph isomorphism.
    """
    G_molecule = matrix_to_graph(molecule_matrix, molecule_symbols)
    G_fragment = matrix_to_graph(fragment_matrix, fragment_symbols)
    GM = isomorphism.GraphMatcher(
        G_molecule, G_fragment,
        node_match=lambda n1, n2: n1['label'] == n2['label']
    )
    all_matches: list[list[int]] = []
    for mapping in GM.subgraph_isomorphisms_iter():
        ordered = [mol_index for mol_index, frag_index in sorted(mapping.items(), key=lambda kv: kv[1])]
        all_matches.append(ordered)
    return remove_duplicate_matches(all_matches)

# Section 3: Atom and Neighbor Information
def print_unique_atoms_with_neighbors(unique_atoms: list[tuple[int, str]], connectivity_matrix: np.ndarray,
                                      molecule_symbols: list[str]) -> tuple[dict, dict]:
    """
    Print the list of unique atoms along with their neighbors and return dictionaries
    with this information.
    """
    num_neighbors_dict: dict = {}
    neighbor_dict: dict = {}
    print("Unique atoms:", unique_atoms)
    for idx, (atom, symbol) in enumerate(unique_atoms):
        neighbors = [n for n in range(len(connectivity_matrix)) if connectivity_matrix[atom - 1, n] == 1]
        neighbor_atoms = ", ".join(f"{n + 1}({molecule_symbols[n]})" for n in neighbors)
        print(f"{atom}({symbol}) \t {neighbor_atoms}")
        num_neighbors_dict[f"{idx}{molecule_symbols[atom - 1]}"] = [len(neighbors)]
        neighbor_dict[f"{atom}({molecule_symbols[atom - 1]})"] = [f"{n + 1}({molecule_symbols[n]})" for n in neighbors]
    return num_neighbors_dict, neighbor_dict


def include_neighbors(unique_atoms: list[tuple[int, str]], connectivity_matrix: np.ndarray,
                      molecule_symbols: list[str]) -> list[tuple[int, str]]:
    """
    Given a list of unique atoms, include all of their neighbors and return the updated list.
    """
    new_atoms = set(unique_atoms)
    for atom, _ in unique_atoms:
        neighbors = [n for n in range(len(connectivity_matrix)) if connectivity_matrix[atom - 1, n] == 1]
        for neighbor in neighbors:
            new_atoms.add((neighbor + 1, molecule_symbols[neighbor]))
    return sorted(new_atoms)


def calculate_fragment_connectivity_matrix(selected_atoms: list[tuple[int, str]], molecule_matrix: np.ndarray):
    """
    Compute the connectivity matrix for the selected fragment.
    """
    indices = [atom - 1 for atom, _ in selected_atoms]
    fragment_matrix = molecule_matrix[np.ix_(indices, indices)]
    return fragment_matrix, indices


def calculate_neighbor_counts(molecule_matrix: np.ndarray, indices: list[int], fragment_symbols: list[str],
                              molecule_symbols: list[str]) -> tuple[dict, dict]:
    """
    Calculate the number of neighbors for each atom in a fragment match.
    """
    adjusted_indices = [i + 1 for i in indices]
    num_neighbors_dict: dict = {}
    neighbor_dict: dict = {}
    for idx, atom_index in enumerate(adjusted_indices):
        neighbors = [n for n in range(len(molecule_matrix)) if molecule_matrix[atom_index - 1, n] == 1]
        num_neighbors_dict[f"{idx}{fragment_symbols[idx]}"] = [len(neighbors)]
        neighbor_dict[f"{atom_index}({fragment_symbols[idx]})"] = [f"{n + 1}({molecule_symbols[n]})" for n in neighbors]
    return num_neighbors_dict, neighbor_dict

# Section 4: Reading and Searching Molecules
def read_molecules_from_xyz_folder(folder: str, mol: str):
    """
    Read XYZ molecules from the specified folder.
    """
    molecules: list[tuple[str, np.ndarray, list[str], any]] = []
    if mol == 'none':
        for filename in os.listdir(folder):
            if filename.endswith('.xyz'):
                path = os.path.join(folder, filename)
                molecule = read(path)
                matrix = calculate_connectivity_matrix(molecule)
                symbols = molecule.get_chemical_symbols()
                molecules.append((filename, matrix, symbols, molecule))
    else:
        molecule = read(mol)
        matrix = calculate_connectivity_matrix(molecule)
        symbols = molecule.get_chemical_symbols()
        molecules.append((mol, matrix, symbols, molecule))
    return molecules


def search_fragment_in_molecules(molecules: list[tuple[str, np.ndarray, list[str], any]],
                                fragment_matrix: np.ndarray,
                                fragment_symbols: list[str]) -> tuple[list, list, list]:
    """
    Search for a fragment in each molecule and return matches.
    """
    results = []
    found = []
    not_found = []
    for name, molecule_matrix, molecule_symbols, _ in molecules:
        matches = match_fragment(molecule_matrix, fragment_matrix, molecule_symbols, fragment_symbols)
        if matches:
            for match in matches:
                fragment_in_molecule = [molecule_symbols[idx] for idx in match]
                num_dict, neighbor_dict = calculate_neighbor_counts(molecule_matrix, match, fragment_in_molecule,
                                                                    molecule_symbols)
                results.append((name, match, fragment_in_molecule, num_dict, neighbor_dict))
            found.append(name)
        else:
            not_found.append(name)
    return results, found, not_found

# Section 5: 3D Molecular Graphics
def get_element_color(symbol: str) -> str:
    """
    Return a display color for a given chemical element symbol.  Default is ochre.
    """
    colors = {
        'H': '#FFFFFF',
        'C': '#B0B0B0',
        'O': 'red',
        'N': 'navy',
        'Cl': 'limegreen',
        'Br': 'darkorange',
        'P': '#FFA500',
        'F': '#DDA0DD',
        'S': '#CCCC00',
        'I': 'purple'
    }
    return colors.get(symbol, '#CC7722')


def get_element_radius(symbol: str) -> float:
    """
    Return a display radius for a given chemical element symbol.
    Radii are scaled relative to hydrogen as the smallest and heavier elements are larger.
    """
    radii = {
        'H': 0.3,
        'O': 0.35,
        'C': 0.4,
        'N': 0.4,
        'S': 0.4,
        'F': 0.4,
        'Cl': 0.5,
        'Br': 0.6,
        'P': 0.6,
        'I': 0.6
    }
    return radii.get(symbol, 0.4)


def select_atoms_interactive(molecule):
    """
    Visualize a molecule in 3D and allow interactive atom selection.

    Mouse click toggles selection (highlighted in pink).  Keyboard shortcuts:

    * `e` – toggle atom labels
    * `n` – include neighbors of the current selection based on the connectivity matrix
    * `m` – clear the selection
    * `q` – close the window

    """
    positions = molecule.get_positions()
    symbols = molecule.get_chemical_symbols()
    n_atoms = len(symbols)

    # Compute chemical connectivity matrix
    A = calculate_connectivity_matrix(molecule)

    # Bonds (using A)
    bonds = []
    for i in range(n_atoms):
        for j in range(i + 1, n_atoms):
            if A[i, j] == 1:
                bonds.append(Tube([positions[i], positions[j]], r=0.08, c='white'))

    # Sphere + label
    atom_assemblies = []
    for i, (pos, sim) in enumerate(zip(positions, symbols)):
        radius = get_element_radius(sim)
        z_offset = radius + 0.1
        color = get_element_color(sim)

        sp = Sphere(pos=pos, r=radius, c=color).lighting('glossy')
        sp.pickable(True); sp.idx = i

        label_str = f"{sim}{i+1}"
        txt = Text3D(label_str, pos=(pos[0], pos[1], pos[2]+z_offset), s=0.2, c='black', justify='center')
        txt.follow_camera(); txt.lighting('off'); txt.pickable(False); txt.alpha(0)

        assembly = Assembly(sp, txt)
        assembly.pickable(True); assembly.idx = i
        atom_assemblies.append(assembly)

    # Scene
    plt = Plotter(axes=0, title="FragmentFinder")
    info_text = Text2D("Shortcuts: e=labels  n=neighbors  m=clear  q=exit",
                       pos="top-left", c='white', bg='black', alpha=0.7)
    texto_info = Text2D("", pos="bottom-left", c='white', bg='black', alpha=0.7)
    plt.add(info_text)

    selected = []
    labels_visible = False

    # Click
    def callback_click(evt):
        if not evt.actor or not hasattr(evt.actor, 'idx'):
            return
        idx = evt.actor.idx
        sphere = atom_assemblies[idx].unpack(0)
        if idx in selected:
            # deselect: restore the original color
            sphere.color(get_element_color(symbols[idx]))
            selected.remove(idx)
        else:
            # select: highlight in pink
            sphere.color('hotpink')
            selected.append(idx)
        texto_info.text(f"Selected atoms: {[k+1 for k in selected]}")
        plt.render()

    plt.add_callback("mouse click", callback_click)

    # Keyboard
    def key_pressed(evt):
        nonlocal labels_visible, selected
        k = (evt.keypress or "").lower()
        if k == "e":
            labels_visible = not labels_visible
            for assembly in atom_assemblies:
                lab = assembly.unpack(1)
                lab.alpha(1 if labels_visible else 0)
            plt.render()

        elif k == "n":
            if selected:
                # include neighbors of the current selection
                current = set(selected)
                for i in list(current):
                    neighbors = np.where(A[i] == 1)[0]
                    current.update(neighbors.tolist())
                selected[:] = sorted(current)
                for a in atom_assemblies:
                    sphere = a.unpack(0)
                    sphere.color('hotpink' if a.idx in selected else get_element_color(symbols[a.idx]))
                texto_info.text(f"Selected atoms: {[k+1 for k in selected]}")
                plt.render()

        elif k == "m":
            selected[:] = []
            for a in atom_assemblies:
                a.unpack(0).color(get_element_color(symbols[a.idx]))
            texto_info.text("Selected atoms: []")
            plt.render()

        elif k == "q":
            plt.close()

    plt.add_callback("key press", key_pressed)

    # Add and show
    plt.add(bonds); plt.add(atom_assemblies); plt.add(texto_info)
    plt.show(resetcam=True, interactive=True)
    return selected


def select_interest_fragment(molecule, fragment_indices):
    """
    Interface to choose atoms of interest within a fragment (0-based indices).

    Interactively select atoms within the previously selected fragment.  Only operate
    within the fragment:

    * Mouse click – toggle selection (pink)
    * `e` – toggle labels
    * `n` – include neighbors within the fragment
    * `m` – clear selection
    * `q` – close the window

    """
    positions = molecule.get_positions()
    symbols = molecule.get_chemical_symbols()

    # Compute global connectivity and restrict to the fragment
    A = calculate_connectivity_matrix(molecule)
    frag_set = set(fragment_indices)

    # Bonds only within the fragment
    bonds = []
    for i in fragment_indices:
        for j in fragment_indices:
            if j > i and A[i, j] == 1:
                bonds.append(Tube([positions[i], positions[j]], r=0.05, c='gray'))

    atom_assemblies = []
    for i in fragment_indices:
        sim = symbols[i]
        radius = get_element_radius(sim)
        z_offset = radius + 0.1
        color0 = get_element_color(sim)

        sp = Sphere(pos=(0,0,0), r=radius, c=color0).lighting('glossy')
        sp.pickable(True); sp.idx = i

        txt = Text3D(f"{sim}{i+1}", pos=(0,0,z_offset), s=0.2, c='black', justify='center')
        txt.follow_camera(); txt.lighting('off'); txt.pickable(False); txt.alpha(0)

        ass = Assembly(sp, txt)
        ass.pickable(True)
        ass.idx = i
        ass.original_color = color0
        ass.SetPosition(positions[i])
        atom_assemblies.append(ass)

    # Scene
    plt = Plotter(axes=0, title="Select atoms of interest (click). 'q' to exit")
    info_text = Text2D("Shortcuts: e=labels  n=neighbors  m=clear  q=exit",
                       pos="top-left", c='white', bg='black', alpha=0.7)
    texto_info = Text2D("", pos="bottom-left", c='white', bg='black', alpha=0.7)
    plt.add(info_text)

    selected = []
    labels_visible = False

    def callback_click(evt):
        if not evt.actor or not hasattr(evt.actor, 'idx'):
            return
        idx = evt.actor.idx
        # toggle selection only within the fragment
        for ass in atom_assemblies:
            if ass.idx == idx:
                sp = ass.unpack(0)
                if idx in selected:
                    sp.color(ass.original_color)
                    selected.remove(idx)
                else:
                    sp.color('hotpink')
                    selected.append(idx)
                break
        texto_info.text(f"Selected atoms: {[i+1 for i in selected]}")
        plt.render()

    plt.add_callback("mouse click", callback_click)

    def key_pressed(evt):
        nonlocal labels_visible, selected
        k = (evt.keypress or "").lower()
        if k == 'e':
            labels_visible = not labels_visible
            for ass in atom_assemblies:
                lab = ass.unpack(1)
                lab.alpha(1 if labels_visible else 0)
            plt.render()
        elif k == 'n':
            if selected:
                current = set(selected)
                for i in list(current):
                    neighbors = np.where(A[i] == 1)[0]
                    # only include neighbors that are also in the fragment
                    current.update([v for v in neighbors if v in frag_set])
                selected[:] = sorted(current)
                for ass in atom_assemblies:
                    ass.unpack(0).color('hotpink' if ass.idx in selected else ass.original_color)
                texto_info.text(f"Selected atoms: {[i+1 for i in selected]}")
                plt.render()
        elif k == 'm':
            selected[:] = []
            for ass in atom_assemblies:
                ass.unpack(0).color(ass.original_color)
            texto_info.text("Selected atoms: []")
            plt.render()
        elif k == 'q':
            plt.close()

    plt.add_callback("key press", key_pressed)

    plt.add(bonds)
    plt.add(atom_assemblies)
    plt.add(texto_info)
    plt.show(resetcam=True, interactive=True)
    return selected


def neighbor_count_signature(dic_num: dict) -> tuple:
    """
    Convert a dictionary like ``{'0C':[3], '1N':[2], ...}`` into a permutation-invariant signature.
    """
    def symbol_from_key(k: str) -> str:
        i = 0
        while i < len(k) and k[i].isdigit():
            i += 1
        return k[i:]

    signature = Counter()
    for k, v in dic_num.items():
        symbol = symbol_from_key(k)
        degree = v[0] if isinstance(v, (list, tuple)) else int(v)
        signature[(symbol, degree)] += 1
    # Return as a sorted tuple to make it comparable
    return tuple(sorted(signature.items()))


def main(fragment_matrix: np.ndarray, fragment_symbols: list[str], directory: str,
         req: str) -> tuple[list, list, list]:
    """
    Search for a fragment within molecules located in directory.
    """
    if req == 'all':
        mols = read_molecules_from_xyz_folder(directory, mol='none')
    else:
        mols = read_molecules_from_xyz_folder(directory, req)
    res, found, not_found = search_fragment_in_molecules(mols, fragment_matrix, fragment_symbols)
    return res, found, not_found

# Function start()
def start(file_path: str, specificity: str, req: str = 'all', search: bool = True) -> tuple[dict, list, dict]:
    """
    Interactively select a fragment. If search=True, search for it in other molecules.
    If search=False, return the selected atoms directly (Direct Selection Mode).
    """
    directory = Path(file_path).parent
    reference_molecule = read(file_path)
    molecule_symbols = reference_molecule.get_chemical_symbols()

    # -- Direct Selection Mode --
    if not search:
        print("\n[Direct Selection Mode] Select atoms of the fragment.")
        while True:
            print("Select atoms in the 3D view:")
            selected = select_atoms_interactive(reference_molecule)
            if not selected:
                print("You must select at least one atom. Please try again.")
                continue
            break
        
        # 0-based indices from selection
        indices = selected
        # 1-based indices for output
        real_indices = [i + 1 for i in indices]
        
        selected_labels = [f"{i+1}({molecule_symbols[i]})" for i in indices]
        print(f"\nSelected atoms: {selected_labels}")

        # Construct a simple result dict for the current molecule
        key = Path(file_path).stem
        results_dict = {
            key: [{
                'fragment_indices': real_indices,
                'fragment_atoms': [molecule_symbols[i] for i in indices],
                'selected_atoms': selected_labels,
                'neighbor_dict': {}, # Not calculated in this mode
                'interest_atom_indices': real_indices # All selected are "of interest"
            }]
        }
        return results_dict, selected_labels, {}

    # -- Original Search Mode --
    molecule_matrix = calculate_connectivity_matrix(reference_molecule)
    print("Connectivity matrix of the reference molecule:")
    print(molecule_matrix)
    print("Atom symbols of the reference molecule:")
    print(molecule_symbols)

    # Define the base fragment
    while True:
        print("\nSelect atoms in the 3D view (press 'n' to include neighbors):")
        selected = select_atoms_interactive(reference_molecule)
        if not selected:
            print("You must select at least one atom of the fragment. Please try again.")
            continue
        # unique_atoms: list of tuples (1-based index, symbol)
        unique_atoms = [(i + 1, molecule_symbols[i]) for i in selected]
        print("\nSelected fragment:")
        for idx, sym in unique_atoms:
            print(f"{idx}: {sym}")
        break

    # Display information about the base fragment
    neighbor_counts_dict, neighbor_dict_interest = print_unique_atoms_with_neighbors(
        unique_atoms, molecule_matrix, molecule_symbols)
    base_signature = neighbor_count_signature(neighbor_counts_dict)
    # Compute the connectivity matrix of the selected fragment
    new_fragment_matrix, fragment_indices = calculate_fragment_connectivity_matrix(unique_atoms, molecule_matrix)
    print("\nConnectivity matrix of the selected fragment:")
    print(new_fragment_matrix)
    print("\nList of atoms in the fragment:")
    for i, (atom, sym) in enumerate(unique_atoms):
        print(f"{i + 1}: {atom}({sym})")

    fragment_symbols = [sym for _, sym in unique_atoms]
    fragment_indices = [atom_idx - 1 for atom_idx, _ in unique_atoms]

    fragment_labels = [
        f"{idx + 1}({molecule_symbols[idx]})"
        for idx in fragment_indices
    ]
    print("\nBase fragment derived from selection:")
    print(fragment_labels)

    # Selection of atoms of interest within the base fragment
    while True:
        print("\nSelect atoms of interest within the base fragment (highlighted in pink).")
        interest_indices = select_interest_fragment(reference_molecule, fragment_indices)
        if not interest_indices:
            print("You must select at least one atom of interest. Please try again.")
            continue
        break
    try:
        interest_rel = [fragment_indices.index(i) for i in interest_indices if i in fragment_indices]
    except ValueError:
        print("Error: Some selected atoms are not in the base fragment. Check your selection.")
        return None

    # Build the list of labels for atoms of interest using the fragment order
    atoms_of_interest = [fragment_labels[j] for j in interest_rel]
    print("\nSelected atoms of interest:")
    print(atoms_of_interest)

    # Results container
    results_dict: dict = {}

    # Search only in the same XYZ file if req == 'none'
    _req = file_path if str(req).lower() == 'none' else req

    results, found, not_found = main(
        new_fragment_matrix,
        [molecule_symbols[idx] for idx in fragment_indices],
        str(directory),
        req=_req
    )

    def record_match(name: str, indices: list[int], fragment_in_molecule: list[str],
                     dic_num: dict, dic_vec: dict, interest_rel_idx: list[int]):
        real_indices = [i + 1 for i in indices]
        full_fragment_labels = [f"{real_indices[i]}({fragment_in_molecule[i]})" for i in range(len(real_indices))]
        selected_fragment_labels = [full_fragment_labels[i] for i in interest_rel_idx if i < len(full_fragment_labels)]
        selected_fragment_indices = [real_indices[i] for i in interest_rel_idx if i < len(real_indices)]
        ordered_neighbors = {
            label: dic_vec.get(label, [])
            for label in selected_fragment_labels
        }
        key = Path(name).stem
        results_dict.setdefault(key, []).append({
            'fragment_indices': real_indices,
            'fragment_atoms': fragment_in_molecule,
            'selected_atoms': selected_fragment_labels,
            'neighbor_dict': ordered_neighbors,
            'interest_atom_indices': selected_fragment_indices
        })
        print(f"\n---> Molecule: {key}")
        print("Fragment atoms (full):", full_fragment_labels)
        print("Selected fragment atoms:", selected_fragment_labels)
        print(f"Neighbor dictionary: {ordered_neighbors}")

    print("\nResults of the fragment search:")
    matched_files: set[str] = set()
    filtered_results_names = []

    for name, indices, frag_in_molecule, dic_num, dic_vec in results:
        if specificity == '1':
            if neighbor_count_signature(dic_num) != base_signature:
                continue  # discard if the signature (symbol, degree) does not match
        
        record_match(name, indices, frag_in_molecule, dic_num, dic_vec, interest_rel)
        matched_files.add(name)
        filtered_results_names.append(name)

    print(f"\nFragment found in {len(matched_files)} file(s).")
    
    # Statistics for multiple matches
    multi_match_count = 0
    # Combine results and not_found for the CSV report
    all_counts = {}

    # Process found fragments using the filtered list
    match_counts = Counter(filtered_results_names)
    
    for name, count in match_counts.items():
        if count > 1:
            multi_match_count += 1
        all_counts[name] = count

    # Process not found
    for name in not_found:
        all_counts[name] = 0

    if multi_match_count > 0:
        print(f"WARNING! Files with >1 match: {multi_match_count}")
        print("Define the common fragment more robustly.\n"
              "Otherwise, the first match will be taken for files with more than one match.")
    
    # Generate CSV Report
    csv_path = os.path.join(directory, "fragment_counts.csv")
    try:
        with open(csv_path, "w", encoding="utf-8") as f:
            f.write("Molecule,Count\n")
            # Sort by filename for cleaner output
            for name in sorted(all_counts.keys()):
                f.write(f"{name},{all_counts[name]}\n")
        print(f"Search report saved to: {csv_path}")
    except Exception as e:
        print(f"Error saving CSV report: {e}")

    if not_found:
        print("Not found in:", not_found)

    return results_dict, atoms_of_interest, neighbor_dict_interest


if __name__ == '__main__':
    # Entry point when running this module directly.  Prompt the user for a reference `.xyz` file and a specificity
    # level, then initiate the fragment search.
    while True:
        file_path = input("Enter the path to the .xyz file of the reference molecule: ").strip()
        if not os.path.isfile(file_path):
            print("The file does not exist. Please enter a valid path.")
            continue
        break

    # Prompt for specificity (only 0 or 1)
    while True:
        print("'0': Connectivity only (Matches based on internal bonds of the fragment).")
        print("'1': Specificity (Matches require matching neighbor environment).")
        specificity = input("Enter the specificity level (0 or 1): ").strip()
        if specificity not in ["0", "1"]:
            print("Only 0 or 1 are accepted. Please try again.")
            continue
        break

    start(file_path, specificity, req='all')

class InteractiveSession:
    def __init__(self, molecule, cp_data, path_data, analysis_callback):
        # Disable default vedo/VTK keyboard shortcuts globally BEFORE creating Plotter
        vedo.settings.enable_default_keyboard_callbacks = False
        
        self.molecule = molecule
        self.raw_cp_data = cp_data  # List of (cp_index, type, [x,y,z])
        self.path_data = path_data  # { path_id: np.array([[x,y,z]...]) }
        self.analysis_callback = analysis_callback # Function to call for CP detection
        
        # State
        self.fragments = defaultdict(set) # { frag_id (1-based): set(atom_indices 0-based) }
        self.current_frag_id = 1
        self.mode = 'EDIT' # 'EDIT' or 'VIEW'
        self.delete_mode = False # Toggle for deleting CPs by click
        
        # Default active CP types: {1: Atom, 2: Bond, 3: Ring, 4: Cage}
        self.active_cp_types = {1, 2, 3, 4}
        self.ignored_cp_indices = set() # Set of deleted CP indices (from raw_cp_data)
        
        # Graphics
        self.plt = Plotter(axes=0, title='Interactive Fragment Selector - Press h for help')
        self.atom_actors = [] # List of Assembly(Sphere, Text)
        self.bond_actors = []
        self.cp_actors = []
        self.path_actors = []
        self.frag_labels = [] # List of Text2D

        self.txt_info = Text2D('', pos='bottom-left', c='white', bg='black', alpha=0.7)
        self.txt_status = Text2D('', pos='top-left', c='white', bg='black', alpha=0.7)
        
        # Result tracking
        self.visible_path_indices = set()

        # Precompute
        self.symbols = molecule.get_chemical_symbols()
        self.positions = molecule.get_positions()
        self.connectivity = calculate_connectivity_matrix(molecule)
        self.n_atoms = len(self.molecule)

        # Setup
        self._build_scene()
        # Force initial text update to show legend/keys
        self._update_status()

    def _build_scene(self):
        # Bonds
        for i in range(self.n_atoms):
            for j in range(i + 1, self.n_atoms):
                if self.connectivity[i, j] == 1:
                    self.bond_actors.append(Tube([self.positions[i], self.positions[j]], r=0.08, c='white'))
        self.plt.add(self.bond_actors)

        # Atoms
        for i, (pos, sym) in enumerate(zip(self.positions, self.symbols)):
            radius = get_element_radius(sym)
            color = get_element_color(sym)
            
            # Sphere
            sp = Sphere(pos=pos, r=radius, c=color).lighting('glossy')
            sp.pickable(True)
            sp.idx = i
            
            # Label (hidden by default)
            z_offset = radius + 0.1
            txt = Text3D(f'{sym}{i+1}', pos=(pos[0], pos[1], pos[2]+z_offset), s=0.2, c='black', justify='center')
            txt.follow_camera(); txt.lighting('off'); txt.pickable(False); txt.alpha(0)
            
            ass = Assembly(sp, txt)
            ass.idx = i
            ass.original_color = color
            self.atom_actors.append(ass)
        
        self.plt.add(self.atom_actors)
        
        # callbacks
        self.plt.add_callback('mouse click', self._on_click)
        self.plt.add_callback('key press', self._on_key)
        self.plt.add(self.txt_info)
        self.plt.add(self.txt_status)

    def _get_frag_color(self, frag_id):
        # Palette for fragments
        colors = {
            1: 'tomato',      # Red-ish
            2: 'dodgerblue',  # Blue-ish
            3: 'mediumseagreen', # Green-ish
            4: 'gold',
            5: 'slateBlue',
            6: 'hotpink',
            7: 'cyan',
            8: 'orange',
            9: 'magenta'
        }
        return colors.get(frag_id, 'gray')

    def _update_atom_colors(self):
        # Reset all to original
        assignment = {} # atom_idx -> frag_id
        for fid, atoms in self.fragments.items():
            for aidx in atoms:
                assignment[aidx] = fid
        
        for i, ass in enumerate(self.atom_actors):
            sp = ass.unpack(0)
            if i in assignment:
                fid = assignment[i]
                sp.color(self._get_frag_color(fid))
                # Highlight current fragment selection more? maybe
            else:
                sp.color(ass.original_color)
                
        # Update labels whenever colors change
        self._update_frag_labels()

    def _update_frag_labels(self):
        # Clear old labels
        self.plt.remove(self.frag_labels)
        self.frag_labels = []
        
        # Start top-right (slightly lower to avoid overlapping with toolbar buttons if any)
        start_x = 0.90
        start_y = 0.90
        step_y = 0.04
        
        active_frags = sorted([f for f, atoms in self.fragments.items() if atoms])
        
        for i, fid in enumerate(active_frags):
            label_color = self._get_frag_color(fid)
            
            # Create 2D text at fixed screen position
            # pos is (x,y) normalized 0..1
            y_pos = start_y - (i * step_y)
            
            t = Text2D(f"Frag {fid}", pos=(start_x, y_pos), s=1.0, c=label_color)
            self.frag_labels.append(t)
            
        self.plt.add(self.frag_labels)

    def _update_status(self):
        # 1. Top-Left: Status & Filters
        def _box(txt, checked):
            state = "ON" if checked else "off"
            return f"{txt}:{state}"

        t1 = _box("Z:Atom", 1 in self.active_cp_types)
        t2 = _box("X:Bond", 2 in self.active_cp_types)
        t3 = _box("C:Ring", 3 in self.active_cp_types)
        t4 = _box("V:Cage", 4 in self.active_cp_types)
        
        del_st = "ACTIVE (Click CP to remove)" if self.delete_mode else "OFF"
        filters = f"Active CPs: {t1} {t2} {t3} {t4} | Delete Mode(d): {del_st}"
            
        if self.mode == 'EDIT':
            header = f"Editing Fragment {self.current_frag_id}  |  {len(self.fragments[self.current_frag_id])} atoms selected"
            msg = f"{header}\n{filters}"
            
            help_txt = (
                "Mouse: Click atom to select/deselect\n"
                "Keys : [1-9] Change Fragment  |  [n] Neighbors  |  [m] Clear\n"
                "       [k] Select All Rest    |  [f] Analyze    |  [q] Finish"
            )
        else:
            header = "Visualizing Interactions"
            msg = f"{header}\n{filters}"
            
            help_txt = (
                "colors of cps types: Lime=Bond, Purple=Ring, Yellow=Cage\n"
                "Keys  : [z,x,c,v] Toggle Filters  |  [d] Delete Mode (Click CP)  |  [r] Restore All\n"
                "        [f] Return to Edit        |  [q] Finish"
            )

        self.txt_status.text(msg)
        self.txt_info.text(help_txt)

    def _on_click(self, evt):
        if not evt.actor: return
        
        # --- CP DELETION LOGIC (VIEW MODE) ---
        if self.mode == 'VIEW' and self.delete_mode:
            # Check if clicked actor is a CP
            if hasattr(evt.actor, 'cp_idx'):
                cp_idx = evt.actor.cp_idx
                print(f"Deleting CP {cp_idx}")
                self.ignored_cp_indices.add(cp_idx)
                self.plt.remove(evt.actor)
                # Should we remove associated paths? Ideally yes, but refreshing view handles it.
                # Just remove actor for immediate feedback.
                return

        # --- ATOM SELECTION LOGIC (EDIT MODE) ---
        if self.mode == 'EDIT':
            if not hasattr(evt.actor, 'idx'): return
            
            idx = evt.actor.idx
            current_owner = None
            for fid, atoms in self.fragments.items():
                if idx in atoms:
                    current_owner = fid
                    break
            
            if current_owner == self.current_frag_id:
                self.fragments[self.current_frag_id].remove(idx)
            else:
                if current_owner is not None:
                    self.fragments[current_owner].remove(idx)
                self.fragments[self.current_frag_id].add(idx)
                
            self._update_atom_colors()
            self._update_status()
            self.plt.render()

    def _on_key(self, evt):
        k = (evt.keypress or '').lower()
        
        if k == 'q':
            self.plt.close()
            return
            
        if k == 'd':
            self.delete_mode = not self.delete_mode
            print(f"Delete Mode: {self.delete_mode}")
            self._update_status()
            self.plt.render()
            return

        if k == 'r':
            # Restore all deleted CPs
            count = len(self.ignored_cp_indices)
            if count > 0:
                self.ignored_cp_indices.clear()
                print(f"Restored {count} deleted CPs.")
                # Refresh analysis or value
                if self.mode == 'VIEW':
                    self._run_analysis()
                else:
                    self.plt.render()
            else:
                print("No deleted CPs to restore.")
            return

        # TOGGLES for CP Types
        toggles = {'z': 1, 'x': 2, 'c': 3, 'v': 4}
        if k in toggles:
            ctype = toggles[k]
            if ctype in self.active_cp_types:
                self.active_cp_types.remove(ctype)
            else:
                self.active_cp_types.add(ctype)
            self._update_status()
            if self.mode == 'VIEW':
                print(f"Refreshing analysis with types: {self.active_cp_types}")
                self._run_analysis()
            else:
                self.plt.render()
            return
            
        if k == 'h':
            print('\n--- HELP ---')
            print('1-9: Switch to Fragment ID')
            print('k: Killer move - Select all unassigned')
            print('n: Neighbors - Add immediate neighbors')
            print('m: Clear - Deselect current fragment')
            print('z, x, c, v: Toggle CP types')
            print('d: Delete CP Mode (Visual only)')
            print('r: Restore All Deleted CPs')
            print('f: Toggle Analysis (Edit <-> View)')
            print('q: Quit and use current results')
            return

        # --- EDIT MODE CONTROLS ---
        if self.mode == 'EDIT':
            if k.isdigit() and k != '0':
                fid = int(k)
                self.current_frag_id = fid
                print(f'Switched to Fragment {fid}')
                self._update_status()

            elif k == 'k':
                assigned = set()
                for atoms in self.fragments.values():
                    assigned.update(atoms)
                count = 0
                for i in range(self.n_atoms):
                    if i not in assigned:
                        self.fragments[self.current_frag_id].add(i)
                        count += 1
                print(f'Added {count} remaining atoms to Fragment {self.current_frag_id}')
                self._update_atom_colors()
                self._update_status()
                self.plt.render()

            elif k == 'n':
                current_atoms = list(self.fragments[self.current_frag_id])
                if current_atoms:
                    rows = self.connectivity[current_atoms]
                    neighbors_indices = np.where(rows.sum(axis=0) > 0)[0]
                    count = 0
                    for idx in neighbors_indices:
                        if idx in self.fragments[self.current_frag_id]: continue
                        for fid, atoms in self.fragments.items():
                            if idx in atoms:
                                atoms.remove(idx)
                                break
                        self.fragments[self.current_frag_id].add(idx)
                        count += 1
                    print(f'Added {count} neighbor atoms to Fragment {self.current_frag_id}')
                    self._update_atom_colors()
                    self._update_status()
                    self.plt.render()

            elif k == 'm':
                count = len(self.fragments[self.current_frag_id])
                if count > 0:
                    self.fragments[self.current_frag_id].clear()
                    print(f'Cleared {count} atoms from Fragment {self.current_frag_id}')
                    self._update_atom_colors()
                    self._update_status()
                    self.plt.render()
                
            elif k == 'f':
                print('Running CP Analysis...')
                self._run_analysis()
                
        # --- VIEW MODE CONTROLS ---
        elif self.mode == 'VIEW':
            if k == 'f':
                print('Returning to Edit Mode...')
                self._clear_analysis_visuals()
                self.mode = 'EDIT'
                self._update_status()
                self.plt.render()

    def _run_analysis(self):
        self._clear_analysis_visuals()
        if len(self.fragments) < 2:
            print('Need at least 2 fragments defined to analyze interfaces.')
            return

        frag_indices_1based = {}
        for fid, atoms in self.fragments.items():
            if not atoms: continue
            name = f'Fragmento{fid}'
            frag_indices_1based[name] = sorted([i + 1 for i in atoms])
            
        idx_nums = np.array([i + 1 for i in range(self.n_atoms)])
        
        try:
            pair_to_cps, pair_to_vmd = self.analysis_callback(
                self.positions, 
                idx_nums, 
                self.raw_cp_data, 
                frag_indices_1based,
                self.active_cp_types 
            )
            self._visualize_results(pair_to_cps)
            self.mode = 'VIEW'
            self._update_status()
            
        except Exception as e:
            print(f'Analysis failed: {e}')
            import traceback
            traceback.print_exc()

    def _visualize_results(self, pair_to_cps):
        # Reset visible paths
        self.visible_path_indices.clear()

        cp_indices_to_show = set()
        for cps in pair_to_cps.values():
            for cp in cps:
                cp_indices_to_show.add(int(cp))
                
        colors = {1: 'silver', 2: 'lime', 3: 'purple', 4: 'gold'}
        visible_path_anchor_coords = []
        
        count = 0
        for (idx, type_cp, coord) in self.raw_cp_data:
            if idx in self.ignored_cp_indices: continue # Skip ignored CPs

            if idx in cp_indices_to_show:
                c = colors.get(type_cp, 'white')
                act = Sphere(pos=coord, r=0.2, c=c).lighting('glossy')
                act.cp_idx = idx # Tag for deletion
                self.cp_actors.append(act)
                count += 1
                
                if type_cp != 1:
                    visible_path_anchor_coords.append(np.array(coord))
                
        path_count = 0
        if self.path_data and visible_path_anchor_coords:
            cutoff = 0.05 
            for pid, coords in self.path_data.items():
                if len(coords) == 0: continue
                show_path = False
                for cp_pos in visible_path_anchor_coords:
                     dists = np.sqrt(np.sum((coords - cp_pos)**2, axis=1))
                     if np.any(dists < cutoff):
                         show_path = True
                         break
                
                if show_path:
                    t = Tube(coords, r=0.03, c='purple').alpha(0.6)
                    self.path_actors.append(t)
                    self.visible_path_indices.add(pid) # Capture ID
                    path_count += 1
                
        self.plt.add(self.cp_actors)
        self.plt.add(self.path_actors)
        print(f'Analysis complete. Showing {count} Interaction CPs and {path_count} Paths.')
        self.plt.render()

    def _clear_analysis_visuals(self):
        self.plt.remove(self.cp_actors)
        self.plt.remove(self.path_actors)
        self.cp_actors = []
        self.path_actors = []

    def start(self):
        self.plt.show(resetcam=True, interactive=False)
        
        if self.plt.interactor:
            style = self.plt.interactor.GetInteractorStyle()
            if style:
                style.RemoveObservers("CharEvent")
        
        if self.plt.interactor and not self.plt.interactor.GetInitialized():
            self.plt.interactor.Initialize()
        if self.plt.interactor:
            self.plt.interactor.Start()
        
        frag_indices_1based = {}
        for fid, atoms in self.fragments.items():
            if not atoms: continue
            name = f'Fragmento{fid}'
            frag_indices_1based[name] = sorted([i + 1 for i in atoms])
            
        # Returning 4 items: fragments, active types, ignored indices, visible paths
        return frag_indices_1based, self.active_cp_types, self.ignored_cp_indices, self.visible_path_indices

# Wrapper function to be called from IME
def run_interactive_session(xyz_path, cp_data, path_data, analysis_callback):
    molecule = read(xyz_path) 
    session = InteractiveSession(molecule, cp_data, path_data, analysis_callback)
    return session.start()

