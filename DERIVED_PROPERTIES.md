# Derived Fragment Properties

This document details the **Derived Local Descriptors** calculated by `DFT-ChemDescriptors`. These descriptors are "derived" because they are not just raw values for single atoms, but rather **aggregated properties** that characterize an entire functional group or molecular fragment.

## Concept
When a fragment is selected (e.g., a hydroxyl group, an aromatic ring), the software computes properties for each constituent atom and bond. To provide a holistic view of the fragment's electronic state (which is crucial for QSAR/QSPR), these values are aggregated.

## Calculated Descriptors

The following descriptors are generated for the fragment and saved in `derived_local_descriptors.csv`:

### 1. Aggregated Atomic Charges & Reactivity Indices
For every charge method selected (e.g., Hirshfeld, CM5, VDZ), the script calculates the **sum** of the properties over all atoms in the fragment.

*   **Sum of Charges (`sum_frag_Charge_Method`)**: Represents the total net charge of the fragment. High positive values indicate an electron-deficient fragment (electrophilic), while negative values indicate an electron-rich fragment (nucleophilic).
*   **Sum of Fukui Indices (`sum_frag_f+_Method`, `sum_frag_f-_Method`)**:
    *   $\sum f^-$: Total nucleophilic susceptibility.
    *   $\sum f^+$: Total electrophilic susceptibility.
*   **Sum of Dual Descriptor (`sum_frag_CDD_Method`)**: Indicates the overall amphiphilic character.

### 2. Aggregated QTAIM Topological Properties
Quantum Theory of Atoms in Molecules (QTAIM) defines critical points (CPs) where the gradient of electron density is zero.

#### Atomic Critical Points (ACPs)
Properties at the nuclear position, summed for all atoms in the fragment:
*   **Electron Density ($\rho$)**: Measure of electronic crowding at the nucleus.
*   **Laplacian ($\nabla^2\rho$)**: Indicates charge concentration or depletion.
*   **Energy Densities**: $H$ (Energy), $V$ (Potential), $G$ (Kinetic).

#### Bond Critical Points (BCPs)
Properties at the saddle point of electron density between bonded atoms **within** the fragment:
*   **Sum of Density ($\rho$) at BCPs**: Correlates with the total bond strength within the fragment.
*   **Energy Densities ($H, V, G$)**: Characterize the nature of interactions (covalent vs. non-covalent).

### 3. Kinetic/Potential Energy Ratio ($-G/V$)
For each state (Neutral, Anion, Cation), the ratio of Lagrangian Kinetic Energy ($G$) to Potential Energy Density ($V$) is calculated as $-G/V$ (or equivalently $G/|V|$):

1.  **Atoms**: For every atom in the selected fragment.
2.  **Bonds**: For every bond where **both** atoms belong to the fragment.

**Aggregated Descriptors**:
*   **Sum of Atomic $-G/V$**: Sum of ratios for all fragment atoms.
*   **Sum of Bond $-G/V$**: Sum of ratios for all fragment bonds.

**Chemical Utility**:
The $-G/V$ ratio is a sensitive indicator of bond nature:
*   **$-G/V < 0.5$**: Indicates **shared-shell (covalent)** interactions. (Potential energy dominates, $H < 0$).
*   **$0.5 < -G/V < 1$**: Indicates **intermediate** interactions.
*   **$-G/V > 1$**: Indicates **closed-shell (non-covalent)** interactions (e.g., ionic, van der Waals). (Kinetic energy dominates).

### 4. Fukui Kernel Descriptors (Bond-Level)

Based on the work of [Franco-Pérez et al. (2020)](https://doi.org/10.1007/s00214-020-2557-4), these descriptors capture bond-level reactivity by computing products of Fukui functions between bonded atom pairs **within** the fragment.


For a bond between atoms $A$ and $B$:

| Descriptor | Formula | Description |
|---|---|---|
| `fukui_kernel_plus` | $f^+_A \cdot f^+_B$ | Product of electrophilic Fukui functions |
| `fukui_kernel_minus` | $f^-_A \cdot f^-_B$ | Product of nucleophilic Fukui functions |
| `fukui_kernel_avg` | $\frac{1}{2}(f^+_A f^+_B + f^-_A f^-_B)$ | Average of the electrophilic and nucleophilic kernels |
| `dual_kernel_simple` | $f^+_A f^+_B - f^-_A f^-_B$ | Simple dual kernel (difference of kernels) |
| `dual_kernel_tau` | $f^+_A f^+_B - f^-_A f^-_B - \frac{1}{2}\Delta f_A \Delta f_B$ | Dual kernel $\tau$ variant |
| `dual_kernel_plus` | $f^+_A f^+_B - f^-_A f^-_B + \frac{1}{2}\Delta f_A \Delta f_B$ | Dual kernel plus variant |

Where $\Delta f_i = f^+_i - f^-_i$ is the Dual Descriptor of atom $i$.

**Naming convention**: `{descriptor_type}_{atom1}_{atom2}_{method}`, with the lower-indexed atom always listed first to ensure canonical ordering.

## Application in QSAR
These derived descriptors allow you to regress biological activity or physical properties against the precise electronic state of a specific pharmacophore or functional group, rather than the entire molecule.

---

## Substituent Site Descriptors *(New in v2.0)*

When **substituent site analysis** is enabled, the script goes beyond the fragment itself and characterizes the **substituent groups** attached to each atom of the common fragment. This produces a rich set of statistical descriptors saved in `substituent_site_descriptors.csv`.

### Concept

For each atom in the common fragment, the script identifies all substituent branches (atoms **not** in the fragment that are connected to it). These branches are then analyzed from two complementary perspectives:

1. **Proximal (from the fragment outward)**: How does the substituent look from the perspective of the fragment atom?
2. **Distal (from the tips inward)**: How does the substituent look from its most distant extremities?

### Naming Convention

All substituent descriptors follow the format:

```
R_{atomID}({symbol})_{block}_{property}_{method_or_state}_{stat}
```

**Examples**:
- `R_12(C)_general_qN_Hirshfeld_mean` — Mean neutral charge (Hirshfeld) over all substituent atoms at site C12
- `R_5(C)_L2_fplus_Hirshfeld_sum` — Sum of $f^+$ for proximal layer L2 at site C5
- `R_12(C)_D1_ACP_Density_of_all_electrons_N_mean` — Mean electron density ACP at the distal tips (D1) of site C12
- `R_12(C)_BCP_anchor_Orbital_Overlap_Distance_D(r)_N_mean` — Mean D(r) at the anchor bond CPs

### Descriptor Blocks

#### General Block
Statistical aggregation over **all** substituent atoms at a site:
- `n_branches`: Number of distinct branches
- `n_atoms`: Total substituent atom count
- `n_heavy_atoms`: Non-hydrogen atoms
- `n_heteroatoms`: Non-C, non-H atoms
- Charge/Fukui statistics: sum, mean, max, min, std for each charge method
- ACP statistics: mean, max, min, std for each QTAIM property
- Internal BCP statistics: mean, max, min, std for bonds within the branches

#### Proximal Layers (L1, L2, L3)
Cumulative topological layers expanding **outward** from the fragment atom via BFS:
- **L1**: Substituent atoms at topological distance 1 (directly bonded to the fragment atom)
- **L2**: L1 $\cup$ atoms at distance $\le 2$
- **L3**: L2 $\cup$ atoms at distance $\le 3$

Same statistics as the General block, but restricted to atoms in each layer.

#### Distal Layers (D1, D2, D3)
Cumulative topological layers expanding **inward** from the most distant substituent atoms:

1. **Determine D_max**: The maximum topological distance from the fragment atom to any substituent atom
2. **Identify the Distal Region**: All atoms at distance $D_{max}$ (the "tips")
3. **Reverse Multi-Source BFS**: Simultaneously propagate from all tip atoms back through the substituent subgraph
4. **Cumulative layers**:
   - **D1**: The distal region itself (distance 0 from tips)
   - **D2**: D1 $\cup$ atoms at distance $\le 1$ from tips
   - **D3**: D2 $\cup$ atoms at distance $\le 2$ from tips

This approach elegantly handles cyclic substituents, bridge-bonded systems, and asymmetric branching without special-case logic.

#### Anchor BCP Block
Bond Critical Point properties at the bond(s) connecting the fragment atom to each substituent root atom:
- `BCP_anchor_count`: Number of anchor bonds
- Statistics (mean, max, min, std) for each QTAIM property across anchor BCPs
- Optionally includes $D(r)$ (Orbital Overlap Distance) at the anchor BCP position

#### Internal BCP Block
Bond Critical Point statistics for bonds **within** the substituent branches (both in General and per-layer blocks):
- `BCP_internal_count`: Number of internal bonds
- Statistics for each QTAIM property

### Dynamic Computation
- **Atomic CPs for substituents**: If ACPs for substituent atoms are not found in the base `_cps_atomic.txt` file, the script dynamically launches Multiwfn to compute them, saving results to `_cps_sub_atomic.txt`
- **Bond CPs**: Anchor and internal BCPs are computed on-demand and cached as `_cps_sub_anchor_{a1}-{a2}_bond.txt` and `_cps_sub_{a1}-{a2}_bond.txt`
- **Skip existing**: If output files already exist, calculations are skipped for efficiency

### Parallelization
All substituent computations are parallelized across molecules using `ThreadPoolExecutor` with a safe thread count of `os.cpu_count() // 2` to prevent system overload.
