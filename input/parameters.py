"""
parameters.py — Beam input parameters as a typed dataclass.

In a real CAE tool (ANSYS, SolidWorks Simulation), this data would come
from a GUI form or a project file (.json/.xml). Here we use a Python
dataclass so everything is explicit, typed, and easy to validate.

All units are SI:  meters [m], Newtons [N], Pascals [Pa], kg/m³.
"""

from dataclasses import dataclass, field
from typing import Literal


@dataclass
class BeamParameters:
    """
    Complete definition of a cantilever beam simulation.

    A cantilever beam is fixed at one end (x=0) and free at the other (x=L).
    A point load F is applied at the free end in the -Y direction.

    Physical setup:
        ─────────────────────────────────►  x
        |  (fixed wall)     ↕ F (downward load at tip)
        ████████████████████○
        x=0                 x=L

    Attributes:
        length:          Beam length L [m]. Default 1.0 m.
        width:           Cross-section width b [m]. Default 0.05 m (5 cm).
        height:          Cross-section height h [m]. Default 0.10 m (10 cm).
        elastic_modulus: Young's modulus E [Pa]. Steel ≈ 200 GPa.
        load_force:      Point load at free end F [N]. Positive = downward.
        num_elements:    Number of finite elements to divide the beam into.
                         More elements → finer mesh → more accurate solution.
        cross_section:   Shape of the cross-section.
        material_name:   Human-readable label (does not affect physics).
        density:         Material density ρ [kg/m³]. Steel ≈ 7850.
        poisson_ratio:   ν (nu). Affects lateral strain. 0.3 for steel.
    """

    # Geometry
    length: float = 1.0
    width: float = 0.05
    height: float = 0.10

    # Material
    elastic_modulus: float = 200e9   # 200 GPa — structural steel
    density: float = 7850.0          # kg/m³
    poisson_ratio: float = 0.3

    # Loading
    load_force: float = 1000.0       # 1 kN

    # Mesh resolution
    num_elements: int = 20

    # Cross-section shape
    cross_section: Literal["rectangular", "circular", "I-beam"] = "rectangular"

    # Label only — does not change calculations
    material_name: str = "steel"
