"""
mesh/beam_mesh.py — Finite element mesh generation for a 1D beam.

What is meshing?
────────────────
A continuous beam has infinite degrees of freedom — you can't solve that
on a computer. Meshing divides it into a finite number of small pieces
("elements"), each described by a simple polynomial approximation.

For a 1D Euler-Bernoulli beam:
  • Each element spans between two NODES.
  • Each node has 2 degrees of freedom (DOF): transverse displacement v
    and rotation θ.
  • The full beam is assembled by connecting elements at shared nodes.

In real CAE pipelines, meshing is done by tools like GMSH, Netgen, or
HyperMesh, which read a STEP file and output a mesh file (.msh, .inp).
We generate the mesh directly in Python using numpy.

Mesh structure for a 20-element beam:
   Node: 0────1────2────3 ... ────20
         |elem0|elem1|elem2|...     |
         x=0                      x=L

Node 0: fixed end (boundary condition applied here)
Node N: free end  (load applied here)

CalculiX input format reference (for when you connect real CalculiX):
   *NODE
   1, 0.0, 0.0, 0.0
   2, 0.05, 0.0, 0.0
   ...
   *ELEMENT, TYPE=B31, ELSET=BEAM
   1, 1, 2
   2, 2, 3
   ...
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Tuple

from geometry.beam_geometry import BeamGeometry


@dataclass
class MeshNode:
    """
    A single point in the FEA mesh.

    In 3D space the beam lies along the X axis, so Y and Z are 0.
    In a full 3D mesh (SOLID elements), nodes would have non-zero Y, Z.
    For our 1D beam formulation, only X matters physically.
    """
    node_id: int
    x: float    # [m] — position along beam axis
    y: float    # [m] — always 0.0 for this 1D beam
    z: float    # [m] — always 0.0 for this 1D beam


@dataclass
class MeshElement:
    """
    A single beam element connecting two adjacent nodes.

    In CalculiX notation, this is element type B31 (3D 2-node beam).
    Each element stores its own length and centroid for convenience.
    """
    element_id: int
    node_ids: Tuple[int, int]   # (start_node, end_node)
    length: float               # l_e [m] — element length
    centroid: np.ndarray        # shape (3,) — midpoint of element [m]


@dataclass
class BeamMesh:
    """
    Complete 1D FEA mesh for the beam.

    Key arrays used by the solver:
        node_coordinates: shape (N_nodes, 3) — [x, y, z] for each node
        connectivity:     shape (N_elem, 2)  — [node_i, node_j] per element
        element_lengths:  shape (N_elem,)    — length of each element
    """
    nodes: List[MeshNode]
    elements: List[MeshElement]
    num_nodes: int
    num_elements: int
    node_coordinates: np.ndarray   # shape (N_nodes, 3)
    connectivity: np.ndarray       # shape (N_elem, 2)
    element_lengths: np.ndarray    # shape (N_elem,)


def generate_mesh(geometry: BeamGeometry) -> BeamMesh:
    """
    Generate a uniform 1D beam mesh.

    Nodes are placed at equal spacing along the X axis:
        x_i = i * (L / N_elements)   for i = 0, 1, ..., N_elements

    Elements connect adjacent node pairs:
        element_i connects node_i to node_{i+1}

    In a real pipeline, this step would:
      1. Pass the STEP file to GMSH: gmsh.model.add("beam"); gmsh.merge("beam.step")
      2. Set mesh size: gmsh.option.setNumber("Mesh.CharacteristicLengthMax", h)
      3. Generate: gmsh.model.mesh.generate(3)
      4. Write .inp file for CalculiX: gmsh.write("beam.inp")

    Args:
        geometry: BeamGeometry containing beam length and num_elements.

    Returns:
        BeamMesh with populated nodes, elements, and numpy arrays.
    """
    # ── MOCK: GMSH mesh generation ────────────────────────────────────────────
    # [REAL CODE WOULD BE]:
    #   import gmsh
    #   gmsh.initialize()
    #   gmsh.model.add("Beam")
    #   gmsh.merge("outputs/beam.step")
    #   gmsh.model.mesh.generate(1)   # 1D mesh along beam axis
    #   gmsh.write("outputs/beam.inp")
    #   gmsh.finalize()
    # ─────────────────────────────────────────────────────────────────────────

    L = geometry.params.length
    N = geometry.params.num_elements  # number of elements
    N_nodes = N + 1                    # number of nodes = elements + 1

    # Element length (uniform mesh)
    le = L / N

    # ── Build nodes ────────────────────────────────────────────────────────
    nodes: List[MeshNode] = []
    node_coords = np.zeros((N_nodes, 3))

    for i in range(N_nodes):
        x = i * le
        nodes.append(MeshNode(node_id=i, x=x, y=0.0, z=0.0))
        node_coords[i] = [x, 0.0, 0.0]

    # ── Build elements ──────────────────────────────────────────────────────
    elements: List[MeshElement] = []
    connectivity = np.zeros((N, 2), dtype=int)
    element_lengths = np.full(N, le)

    for i in range(N):
        n1, n2 = i, i + 1
        centroid = np.array([(nodes[n1].x + nodes[n2].x) / 2.0, 0.0, 0.0])
        elements.append(MeshElement(
            element_id=i,
            node_ids=(n1, n2),
            length=le,
            centroid=centroid,
        ))
        connectivity[i] = [n1, n2]

    return BeamMesh(
        nodes=nodes,
        elements=elements,
        num_nodes=N_nodes,
        num_elements=N,
        node_coordinates=node_coords,
        connectivity=connectivity,
        element_lengths=element_lengths,
    )


def get_mesh_report(mesh: BeamMesh) -> dict:
    """
    Return a loggable summary of mesh statistics.

    Engineers use this to verify mesh quality before running the solver.
    Key metric: mesh density (elements per meter). For a beam under
    bending, 10–50 elements is typically sufficient for <1% error.
    """
    le = mesh.element_lengths[0]  # uniform mesh
    L_total = le * mesh.num_elements
    return {
        "num_nodes":                    mesh.num_nodes,
        "num_elements":                 mesh.num_elements,
        "element_length_m":             f"{le:.6f}",
        "total_length_m":               f"{L_total:.4f}",
        "mesh_density_elem_per_m":      f"{mesh.num_elements / L_total:.2f}",
        "node_id_range":                f"[0, {mesh.num_nodes - 1}]",
        "dof_per_node":                 2,   # v (displacement) + θ (rotation)
        "total_dof":                    2 * mesh.num_nodes,
    }
