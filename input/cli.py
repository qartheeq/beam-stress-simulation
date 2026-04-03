"""
cli.py — Command-line interface for the Mini CAE Pipeline.

This module parses terminal arguments and maps them to BeamParameters.
In a production CAE tool you'd have a full GUI — this is the minimal
equivalent for scripting and automation.

Usage example:
    python main.py --length 2.0 --force 5000 --elements 40
"""

import argparse
import sys
from typing import List, Optional
from .parameters import BeamParameters


def parse_args(argv: Optional[List[str]] = None) -> BeamParameters:
    """
    Parse command-line arguments into a BeamParameters instance.

    If no arguments are given, all defaults from BeamParameters are used,
    so `python main.py` runs the pipeline with a sensible steel beam.

    Args:
        argv: Argument list (defaults to sys.argv[1:] when None).

    Returns:
        BeamParameters populated from CLI flags.
    """
    parser = argparse.ArgumentParser(
        prog="mini-cae",
        description=(
            "Mini CAE Pipeline — Cantilever Beam Stress Simulator\n"
            "Simulates stress and deflection for a rectangular cantilever beam."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # ── Geometry ──────────────────────────────────────────────────────────────
    parser.add_argument(
        "--length", type=float, default=1.0,
        metavar="L",
        help="Beam length [m]",
    )
    parser.add_argument(
        "--width", type=float, default=0.05,
        metavar="B",
        help="Cross-section width [m]",
    )
    parser.add_argument(
        "--height", type=float, default=0.10,
        metavar="H",
        help="Cross-section height [m]",
    )

    # ── Material ──────────────────────────────────────────────────────────────
    parser.add_argument(
        "--elastic-modulus", type=float, default=200e9,
        metavar="E",
        help="Young's modulus [Pa]  (steel=200e9, aluminium=70e9)",
    )
    parser.add_argument(
        "--density", type=float, default=7850.0,
        metavar="RHO",
        help="Material density [kg/m³]",
    )
    parser.add_argument(
        "--poisson", type=float, default=0.3,
        metavar="NU",
        help="Poisson's ratio (dimensionless)",
    )
    parser.add_argument(
        "--material", type=str, default="steel",
        metavar="NAME",
        help="Material label (informational only)",
    )

    # ── Loading ───────────────────────────────────────────────────────────────
    parser.add_argument(
        "--force", type=float, default=1000.0,
        metavar="F",
        help="Point load at free end [N]  (positive = downward)",
    )

    # ── Mesh ──────────────────────────────────────────────────────────────────
    parser.add_argument(
        "--elements", type=int, default=20,
        metavar="N",
        help="Number of beam elements (finer mesh = more accurate)",
    )

    # ── Cross-section shape ───────────────────────────────────────────────────
    parser.add_argument(
        "--section", type=str, default="rectangular",
        choices=["rectangular", "circular", "I-beam"],
        help="Cross-section shape",
    )

    args = parser.parse_args(argv)

    # Map parsed namespace → BeamParameters dataclass
    return BeamParameters(
        length=args.length,
        width=args.width,
        height=args.height,
        elastic_modulus=args.elastic_modulus,
        density=args.density,
        poisson_ratio=args.poisson,
        load_force=args.force,
        num_elements=args.elements,
        cross_section=args.section,
        material_name=args.material,
    )


def validate_parameters(params: BeamParameters) -> None:
    """
    Sanity-check physical parameters before the pipeline starts.

    CAE tools typically validate inputs at this stage to give the user
    clear error messages before wasting time generating geometry/mesh.

    Raises:
        ValueError: With a descriptive message for the first failing check.
    """
    checks = [
        (params.length > 0,          f"length must be > 0, got {params.length}"),
        (params.width > 0,           f"width must be > 0, got {params.width}"),
        (params.height > 0,          f"height must be > 0, got {params.height}"),
        (params.elastic_modulus > 0, f"elastic_modulus must be > 0, got {params.elastic_modulus}"),
        (params.density > 0,         f"density must be > 0, got {params.density}"),
        (0 < params.poisson_ratio < 0.5,
                                     f"poisson_ratio must be in (0, 0.5), got {params.poisson_ratio}"),
        (params.load_force != 0,     "load_force must be non-zero"),
        (params.num_elements >= 2,   f"num_elements must be >= 2, got {params.num_elements}"),
        (params.width <= params.length,
                                     "width should not exceed length (unlikely geometry)"),
    ]

    for condition, message in checks:
        if not condition:
            raise ValueError(f"Invalid parameter: {message}")
