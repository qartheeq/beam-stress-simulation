"""
beam_geometry.py — Parametric beam geometry generation.

In a real pipeline this module would call FreeCAD's Python API:

    import FreeCAD, Part
    doc = FreeCAD.newDocument("BeamModel")
    box = doc.addObject("Part::Box", "Beam")
    box.Length = params.length * 1000   # FreeCAD uses mm internally
    box.Width  = params.width  * 1000
    box.Height = params.height * 1000
    doc.recompute()
    Part.export([box], "beam.step")

We mock that API here and instead compute the same geometric properties
analytically using numpy. The rest of the pipeline doesn't know the difference.

In production CAE systems, geometry is stored in STEP or IGES format,
which is what meshing tools (GMSH, Netgen) consume as input.
"""

import numpy as np
from dataclasses import dataclass
from typing import Tuple

from input.parameters import BeamParameters
from .shapes import CrossSection, compute_cross_section


@dataclass
class BeamGeometry:
    """
    Complete geometric description of the beam.

    Equivalent to what FreeCAD's Part workbench would give you after
    creating a solid box and computing its properties.

    Attributes:
        params:        Original input parameters (stored for reference).
        cross_section: Computed section properties (I, A, c, S).
        axis_points:   (N, 3) array of centerline sample points [m].
                       In FreeCAD terms, these are points along the beam's
                       "sweep path" — useful for visualization and meshing.
        bounding_box:  (length, width, height) tuple [m].
        volume:        Total beam volume V = A × L [m³].
        surface_area:  Total beam surface area [m²].
        mass:          m = ρ × V [kg].
    """
    params: BeamParameters
    cross_section: CrossSection
    axis_points: np.ndarray          # shape (N_pts, 3)
    bounding_box: Tuple[float, float, float]
    volume: float
    surface_area: float
    mass: float


def generate_beam_geometry(params: BeamParameters) -> BeamGeometry:
    """
    Generate beam geometry — mocking the FreeCAD Part workbench.

    This function's structure mirrors a real FreeCAD script:
      1. Create document
      2. Define cross-section shape
      3. Sweep along beam axis (or use makeBox for rectangular)
      4. Compute solid properties (volume, surface area, CoM)
      5. Export to STEP

    Steps 1, 3, and 5 are mocked. Steps 2 and 4 use real math.

    Args:
        params: Validated BeamParameters.

    Returns:
        BeamGeometry with all properties populated.
    """
    # ── MOCK: FreeCAD document creation ──────────────────────────────────────
    # [REAL CODE WOULD BE]:
    #   import FreeCAD, Part
    #   doc = FreeCAD.newDocument("BeamModel")
    #   box = doc.addObject("Part::Box", "Beam")
    #   box.Length, box.Width, box.Height = L*1000, b*1000, h*1000
    #   doc.recompute()
    # ─────────────────────────────────────────────────────────────────────────
    _freecad_mock_create_document("BeamModel")

    # Step 1: Compute cross-section properties (REAL math)
    cross_section = compute_cross_section(
        cross_section_type=params.cross_section,
        width=params.width,
        height=params.height,
    )

    # Step 2: Generate centerline axis points along X from 0 → L (REAL math)
    # These 100 points represent the beam's "spine" — used by meshing and viz
    n_pts = max(100, params.num_elements * 5)
    x_coords = np.linspace(0.0, params.length, n_pts)
    y_coords = np.zeros(n_pts)   # beam lies along X axis, centered at Y=0
    z_coords = np.zeros(n_pts)
    axis_points = np.column_stack([x_coords, y_coords, z_coords])

    # Step 3: Compute solid properties (REAL math)
    L, b, h = params.length, params.width, params.height
    volume = cross_section.area * L

    # Surface area of a rectangular prism: 2(LW + LH + WH)
    # For non-rectangular sections, use perimeter × length + 2 × end faces
    if params.cross_section == "rectangular":
        surface_area = 2.0 * (L * b + L * h + b * h)
    else:
        # Perimeter × length + 2 × end area (approximate for circular/I-beam)
        import math
        if params.cross_section == "circular":
            perimeter = math.pi * b           # circumference
        else:
            # I-beam: approximate perimeter of thin-walled shape
            tf = h / 8.0
            tw = b / 5.0
            hw = h - 2.0 * tf
            perimeter = 2.0 * (b + tf) + 2.0 * (hw + tw)
        surface_area = perimeter * L + 2.0 * cross_section.area

    mass = params.density * volume

    # ── MOCK: FreeCAD export to STEP ─────────────────────────────────────────
    # [REAL CODE WOULD BE]:
    #   import ImportGui
    #   ImportGui.export([doc.getObject("Beam")], "outputs/beam.step")
    # ─────────────────────────────────────────────────────────────────────────
    _freecad_mock_export_step("outputs/beam.step")

    return BeamGeometry(
        params=params,
        cross_section=cross_section,
        axis_points=axis_points,
        bounding_box=(L, b, h),
        volume=volume,
        surface_area=surface_area,
        mass=mass,
    )


def get_geometry_report(geometry: BeamGeometry) -> dict:
    """
    Return a structured dict summary suitable for logging.

    These are the values an engineer would verify before proceeding
    to meshing: "Does the geometry look right?"
    """
    cs = geometry.cross_section
    p = geometry.params
    return {
        "cross_section_type": p.cross_section,
        "length_m":           round(p.length, 4),
        "width_m":            round(p.width, 4),
        "height_m":           round(p.height, 4),
        "area_m2":            f"{cs.area:.6e}",
        "moment_of_inertia_m4": f"{cs.moment_of_inertia:.6e}",
        "section_modulus_m3": f"{cs.section_modulus:.6e}",
        "c_m":                f"{cs.c:.6e}",
        "volume_m3":          f"{geometry.volume:.6e}",
        "surface_area_m2":    f"{geometry.surface_area:.4f}",
        "mass_kg":            f"{geometry.mass:.4f}",
    }


# ── Internal mock helpers ─────────────────────────────────────────────────────
# These functions represent the FreeCAD API calls that would happen in
# a real pipeline. They are no-ops here but show WHERE real code goes.

def _freecad_mock_create_document(name: str) -> None:
    """
    MOCK — In production:
        FreeCAD.newDocument(name)
    """
    pass  # No-op: FreeCAD not installed in this environment


def _freecad_mock_export_step(path: str) -> None:
    """
    MOCK — In production:
        Part.export([solid], path)
    Then GMSH would read this .step file to create the mesh.
    """
    pass  # No-op: FreeCAD not installed in this environment
