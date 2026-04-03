"""
shapes.py — Cross-section geometric property calculator.

In structural mechanics, the shape of a beam's cross-section determines
how it resists bending. The key property is the Second Moment of Area (I),
also called the Area Moment of Inertia.

Think of I as "how far is material distributed from the neutral axis?"
An I-beam puts most material far from the center → high I → stiff beam.

                     Neutral axis (bending about Z)
                     ─────────────────────────
         ┌───────┐        ▲
         │       │        │ c = h/2 (for rectangle)
         │       │        ▼
         └───────┘
         b (width)

Bending stress formula:  σ = M·c / I
  • M  = bending moment at a cross-section [N·m]
  • c  = distance from neutral axis to extreme fiber [m]
  • I  = second moment of area [m⁴]

Larger I → smaller σ for the same moment → less stress.
"""

import math
from dataclasses import dataclass


@dataclass
class CrossSection:
    """
    Computed geometric properties of a beam cross-section.

    All values are derived analytically from width and height.
    These are the inputs to the FEA solver — analogous to what
    FreeCAD's Section Properties tool would output.
    """
    area: float               # A  [m²]  — resists axial load
    moment_of_inertia: float  # I  [m⁴]  — resists bending (about Z axis)
    centroid_y: float         # ȳ  [m]   — distance from bottom to neutral axis
    c: float                  # c  [m]   — distance from neutral axis to extreme fiber
    section_modulus: float    # S = I/c [m³] — convenience: σ_max = M/S


def compute_cross_section(
    cross_section_type: str,
    width: float,
    height: float,
) -> CrossSection:
    """
    Compute cross-section properties for a given shape.

    Args:
        cross_section_type: "rectangular", "circular", or "I-beam"
        width:  b dimension [m] — also used as diameter for circular
        height: h dimension [m]

    Returns:
        CrossSection with analytically computed properties.

    Raises:
        ValueError: For unknown section types.

    Notes on formulas:
        Rectangular:
            A = b·h
            I = b·h³/12     (bending about horizontal neutral axis)
            c = h/2

        Circular (width = diameter d):
            A = π·d²/4
            I = π·d⁴/64
            c = d/2

        I-beam (simplified thin-flange approximation):
            Flanges: b × (h/6) each, at top and bottom
            Web:     (b/6) × (2h/3), centered
            I = 2·[b·tf³/12 + b·tf·(h/2 - tf/2)²] + tw·hw³/12
            where tf = flange thickness, hw = web height, tw = web thickness
    """
    ctype = cross_section_type.lower()

    if ctype == "rectangular":
        # Standard rectangular section — most common in textbook examples
        A = width * height
        I = (width * height**3) / 12.0
        centroid_y = height / 2.0
        c = height / 2.0

    elif ctype == "circular":
        # Circular section — width is treated as diameter
        d = width
        A = math.pi * d**2 / 4.0
        I = math.pi * d**4 / 64.0
        centroid_y = d / 2.0
        c = d / 2.0

    elif ctype == "i-beam":
        # Simplified I-beam: flanges are (b × h/8), web is (b/5 × 3h/4)
        # These proportions approximate a standard wide-flange section
        b = width
        h = height
        tf = h / 8.0          # flange thickness
        tw = b / 5.0          # web thickness
        hw = h - 2.0 * tf     # web height (clear between flanges)

        # Flange areas (top + bottom, symmetric → neutral axis at h/2)
        A_flange = b * tf
        A_web = tw * hw
        A = 2.0 * A_flange + A_web

        # Parallel axis theorem: I_total = I_web + 2*(I_flange + A_flange*d²)
        # d = distance from centroid of flange to neutral axis
        d_flange = (hw / 2.0) + (tf / 2.0)
        I_web = (tw * hw**3) / 12.0
        I_flange = (b * tf**3) / 12.0
        I = I_web + 2.0 * (I_flange + A_flange * d_flange**2)

        centroid_y = h / 2.0
        c = h / 2.0

    else:
        raise ValueError(
            f"Unknown cross_section type: '{cross_section_type}'. "
            f"Choices: 'rectangular', 'circular', 'I-beam'"
        )

    section_modulus = I / c
    return CrossSection(
        area=A,
        moment_of_inertia=I,
        centroid_y=centroid_y,
        c=c,
        section_modulus=section_modulus,
    )
