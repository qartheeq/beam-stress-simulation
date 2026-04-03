"""
simulation/results.py — Simulation output data structure.

In a real CalculiX run, results are stored in .frd (FRD) binary files,
which are then parsed by post-processors like ParaView or CalculiX GraphiX.

Our SimulationResults dataclass is the in-memory equivalent of a parsed
.frd file — containing all the field variables (displacement, stress) at
every node, ready for visualization.

Fields follow Abaqus/CalculiX naming conventions:
    U1, U2, U3   → displacements in X, Y, Z
    S11          → normal stress in X (axial)
    S12          → shear stress in XY plane
    S_MISES      → von Mises equivalent stress
"""

import numpy as np
from dataclasses import dataclass


@dataclass
class SimulationResults:
    """
    All output fields from the FEA solver, indexed by node.

    Array shape: (N_nodes,) — one value per node along the beam.
    Scalar fields: summary values extracted from the arrays.
    """

    # ── Displacement fields (per node) ──────────────────────────────────────
    # In FEA we solve for these: K · u = F, where u is this displacement vector.

    displacement_x: np.ndarray      # U1: axial displacement [m]
                                     #   For pure bending with no axial load → ≈ 0
    displacement_y: np.ndarray      # U2: transverse deflection [m]
                                     #   This is the "sagging" of the beam
    displacement_z: np.ndarray      # U3: out-of-plane [m] → 0 for 2D loading
    total_displacement: np.ndarray  # |U| = sqrt(U1²+U2²+U3²) [m]

    # ── Stress fields (per node) ─────────────────────────────────────────────
    # Computed from displacements via the constitutive (material) law: σ = E·ε

    bending_stress: np.ndarray      # σ_bending = M(x)·c/I [Pa]
                                     #   Max at fixed end, 0 at free end (tip)
    shear_stress: np.ndarray        # τ = V·Q/(I·b) [Pa]
                                     #   Constant along beam for point-load
    von_mises_stress: np.ndarray    # σ_vm = sqrt(σ² + 3τ²) [Pa]
                                     #   Industry standard scalar stress measure
    axial_stress: np.ndarray        # σ_axial = F_axial/A [Pa] → 0 here

    # ── Scalar summary values ─────────────────────────────────────────────────
    max_deflection: float           # δ_max = F·L³/(3·E·I) [m]  (at free end)
    max_bending_stress: float       # σ_max = F·L·c/I [Pa]       (at fixed end)
    max_von_mises: float            # [Pa]
    reaction_force_y: float         # R_y at fixed end [N]  = F (equilibrium)
    reaction_moment_z: float        # M_z at fixed end [N·m] = F·L

    # ── Solver metadata ────────────────────────────────────────────────────────
    converged: bool
    solver_iterations: int          # Number of iterations (1 for linear solver)
    residual_norm: float            # ||K·u - F|| — should be near machine epsilon
    solve_time_seconds: float
