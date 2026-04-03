# Mesh module: divides the geometry into finite elements.
# This is the bridge between continuous geometry and discrete FEA math.
from .beam_mesh import BeamMesh, MeshNode, MeshElement, generate_mesh, get_mesh_report

__all__ = ["BeamMesh", "MeshNode", "MeshElement", "generate_mesh", "get_mesh_report"]
