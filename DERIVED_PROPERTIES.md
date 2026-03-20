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
