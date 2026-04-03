# Geometry module: builds the 3D beam model programmatically.
# In a real pipeline this would call FreeCAD's Part workbench API.
# Here we compute the exact same geometric properties analytically.
from .shapes import CrossSection, compute_cross_section
from .beam_geometry import BeamGeometry, generate_beam_geometry, get_geometry_report

__all__ = [
    "CrossSection", "compute_cross_section",
    "BeamGeometry", "generate_beam_geometry", "get_geometry_report",
]
