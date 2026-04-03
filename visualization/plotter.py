"""
visualization/plotter.py — Result visualization for the CAE pipeline.

In production CAE workflows, post-processing is done in ParaView:
    - Reads .frd or .vtk files output by the solver
    - Renders 3D surface color maps, deformation animations
    - Provides interactive stress probing, section cuts

We use matplotlib to produce equivalent 2D plots that show the same
physical information. The plots are saved as PNG files.

Two primary outputs:
    1. Stress heatmap: color-coded beam showing stress distribution
    2. Deformation plot: original vs. deflected shape overlay

Note on ParaView equivalence:
    ParaView "Warp by Vector" filter → our plot_deformation()
    ParaView "Color by Field" (von Mises) → our plot_stress_heatmap()
"""

from pathlib import Path
from typing import Optional, Dict
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.patches import FancyArrowPatch

from mesh.beam_mesh import BeamMesh
from simulation.results import SimulationResults
from input.parameters import BeamParameters


# Field name → SimulationResults attribute mapping
AVAILABLE_FIELDS = {
    "von_mises_stress": "von_mises_stress",
    "bending_stress":   "bending_stress",
    "shear_stress":     "shear_stress",
    "axial_stress":     "axial_stress",
    "displacement_y":   "displacement_y",
}


def plot_stress_heatmap(
    mesh: BeamMesh,
    results: SimulationResults,
    params: BeamParameters,
    output_path: Path,
    field: str = "von_mises_stress",
    colormap: str = "hot",
    dpi: int = 150,
) -> Path:
    """
    Generate a 2D stress heatmap of the beam viewed from the side.

    The beam is drawn as a rectangle extruded to its cross-section height.
    Color represents stress magnitude — red/white = high stress, black = low.
    This is the classic "contour plot" view you see in ANSYS or Abaqus.

    Visualization approach:
        - X axis: beam length (0 → L)
        - Y axis: cross-section height (-h/2 → +h/2)
        - Color: stress field value at each node
        - The stress distribution is shown using pcolormesh with bilinear
          interpolation across both the length and height of the beam

    Physical insight on bending stress distribution:
        - At the top fiber (y = +h/2): compression (negative in convention)
        - At neutral axis (y = 0):    zero bending stress
        - At bottom fiber (y = -h/2): tension (positive)
        - At fixed end (x = 0):       maximum stress
        - At free end (x = L):        zero bending stress

    Args:
        mesh:        BeamMesh with node x-coordinates.
        results:     SimulationResults with stress arrays.
        params:      BeamParameters for labeling.
        output_path: Where to save the PNG.
        field:       Stress field to visualize.
        colormap:    matplotlib colormap name.
        dpi:         Output resolution.

    Returns:
        Path to the saved PNG.
    """
    if field not in AVAILABLE_FIELDS:
        raise ValueError(
            f"Unknown field '{field}'. Available: {list(AVAILABLE_FIELDS.keys())}"
        )

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Get the stress values at each node (1D array along beam length)
    stress_1d = getattr(results, AVAILABLE_FIELDS[field])  # shape (N_nodes,)
    x_nodes = mesh.node_coordinates[:, 0]                  # shape (N_nodes,)

    # Create a 2D stress grid: extrude along the beam height
    # For bending stress, the actual distribution across the height is:
    #     σ(x, y) = M(x) · y / I    (varies linearly with y)
    # This gives the full physical picture — high at top/bottom fibers.
    h = params.height
    n_height = 20  # Number of points across the cross-section height

    y_levels = np.linspace(-h / 2.0, h / 2.0, n_height)   # shape (n_height,)

    # Build 2D stress grid: stress[height_idx, x_idx]
    if field == "bending_stress" or field == "von_mises_stress":
        # Bending stress varies linearly with y: σ(x,y) = σ_extreme(x) · (y/c)
        # σ_extreme(x) is the stress at the extreme fiber stored in results
        c = params.height / 2.0
        # stress_2d[i, j] = stress at height y_levels[i], node x_nodes[j]
        stress_2d = np.outer(y_levels / c, stress_1d)  # shape (n_height, N_nodes)
        # Take absolute value for von Mises (always positive)
        if field == "von_mises_stress":
            stress_2d = np.abs(stress_2d)
    else:
        # For shear stress: parabolic distribution, max at centroid
        # τ(y) = (3/2)·τ_avg·[1 - (2y/h)²]  for rectangular section
        profile = 1.5 * (1.0 - (2.0 * y_levels / h)**2)  # shape (n_height,)
        stress_2d = np.outer(profile, stress_1d)

    # ── Plot ──────────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(12, 4), dpi=dpi)
    fig.patch.set_facecolor("#1a1a2e")
    ax.set_facecolor("#1a1a2e")

    # Mesh grid for pcolormesh
    X_grid, Y_grid = np.meshgrid(x_nodes, y_levels)

    vmax = np.max(np.abs(stress_2d))
    vmin = 0.0

    pcm = ax.pcolormesh(
        X_grid, Y_grid, stress_2d,
        cmap=colormap,
        vmin=vmin, vmax=vmax,
        shading="gouraud",   # Smooth interpolation between nodes
    )

    # Colorbar
    cbar = fig.colorbar(pcm, ax=ax, pad=0.02, shrink=0.85)
    cbar.set_label(f"{field.replace('_', ' ').title()} [Pa]", color="white", fontsize=11)
    cbar.ax.yaxis.set_tick_params(color="white")
    plt.setp(cbar.ax.yaxis.get_ticklabels(), color="white")

    # Draw beam outline
    rect_x = [0, params.length, params.length, 0, 0]
    rect_y = [-h/2, -h/2, h/2, h/2, -h/2]
    ax.plot(rect_x, rect_y, "w-", linewidth=1.5, alpha=0.7)

    # Fixed wall indicator
    wall_y = np.linspace(-h/2, h/2, 8)
    for wy in wall_y:
        ax.annotate("", xy=(-0.02 * params.length, wy),
                    xytext=(0, wy),
                    arrowprops=dict(arrowstyle="-", color="cyan", lw=1.5))
    ax.axvline(x=0, color="cyan", linewidth=3, alpha=0.8, label="Fixed wall")

    # Force arrow at tip
    ax.annotate("",
        xy=(params.length, -h/2 - h*0.3),
        xytext=(params.length, -h/2 - h*0.05),
        arrowprops=dict(arrowstyle="->", color="yellow", lw=2.5))
    ax.text(params.length, -h/2 - h*0.45,
            f"F={params.load_force/1000:.1f} kN",
            color="yellow", ha="center", fontsize=9)

    # Labels and formatting
    ax.set_xlabel("Beam Length x [m]", color="white", fontsize=11)
    ax.set_ylabel("Cross-section y [m]", color="white", fontsize=11)
    ax.set_title(
        f"Cantilever Beam — {field.replace('_', ' ').title()}\n"
        f"L={params.length}m  b={params.width*100:.1f}cm  h={params.height*100:.1f}cm  "
        f"E={params.elastic_modulus/1e9:.0f}GPa  F={params.load_force/1000:.1f}kN",
        color="white", fontsize=12, pad=10,
    )
    ax.tick_params(colors="white")
    for spine in ax.spines.values():
        spine.set_edgecolor("gray")

    # Stress summary annotation
    max_mpa = results.max_von_mises / 1e6
    ax.text(0.02, 0.95,
            f"σ_vm,max = {max_mpa:.2f} MPa\nδ_max = {results.max_deflection*1000:.2f} mm",
            transform=ax.transAxes,
            color="lightgreen", fontsize=9, verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="black", alpha=0.6))

    plt.tight_layout()
    plt.savefig(output_path, dpi=dpi, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)

    return output_path


def plot_deformation(
    mesh: BeamMesh,
    results: SimulationResults,
    params: BeamParameters,
    output_path: Path,
    scale_factor: Optional[float] = None,
    dpi: int = 150,
) -> Path:
    """
    Generate a deformation plot showing original vs. deflected beam shape.

    This is equivalent to ParaView's "Warp by Vector" visualization — the
    deformed shape is shown with exaggerated displacement for visibility.

    Why exaggerate? In reality, a 1m steel beam under 1 kN deflects only
    ~1mm — invisible at plot scale. We scale it up visually while labeling
    the actual displacement value.

    The plot shows:
        - Dashed grey line: original (undeformed) beam centerline
        - Colored line:     deformed centerline (colored by von Mises stress)
        - Filled band:      beam body with stress color fill
        - Annotations:      max deflection, scale factor, reaction forces

    Args:
        mesh:         BeamMesh for original node positions.
        results:      SimulationResults with displacements and stress.
        params:       BeamParameters for annotations.
        output_path:  Where to save the PNG.
        scale_factor: Visual exaggeration (auto-computed if None).
        dpi:          Output resolution.

    Returns:
        Path to the saved PNG.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    x_orig = mesh.node_coordinates[:, 0]   # shape (N_nodes,)
    v_disp = results.displacement_y          # shape (N_nodes,) — downward positive

    # Auto-compute scale factor so max visual deflection = 10% of beam length
    if scale_factor is None:
        max_disp = np.max(np.abs(v_disp))
        if max_disp > 0:
            scale_factor = 0.10 * params.length / max_disp
        else:
            scale_factor = 1.0

    # Deformed Y coordinates (negative = downward in beam convention)
    y_deformed = -v_disp * scale_factor     # flip sign for downward visual

    h = params.height
    vm_stress = results.von_mises_stress    # shape (N_nodes,)
    vm_norm = vm_stress / (np.max(vm_stress) + 1e-10)   # Normalize 0→1

    # ── Plot ──────────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(2, 1, figsize=(13, 7), dpi=dpi,
                              gridspec_kw={"height_ratios": [3, 1]})
    fig.patch.set_facecolor("#0d1117")

    # ── Top panel: Deformation view ───────────────────────────────────────────
    ax = axes[0]
    ax.set_facecolor("#0d1117")

    # Draw original beam as grey rectangle outline
    beam_top_orig    = np.full_like(x_orig,  h / 2.0)
    beam_bottom_orig = np.full_like(x_orig, -h / 2.0)
    ax.fill_between(x_orig, beam_bottom_orig, beam_top_orig,
                    alpha=0.15, color="gray", label="Original shape")
    ax.plot(x_orig, beam_top_orig,    "--", color="gray", linewidth=1.0, alpha=0.5)
    ax.plot(x_orig, beam_bottom_orig, "--", color="gray", linewidth=1.0, alpha=0.5)

    # Deformed beam body: centerline shifted by scaled displacement
    beam_top_def    = y_deformed + h / 2.0
    beam_bottom_def = y_deformed - h / 2.0

    # Color the deformed beam by von Mises stress using a colormap
    cmap = plt.cm.plasma
    for i in range(len(x_orig) - 1):
        color = cmap(vm_norm[i])
        ax.fill_between(
            x_orig[i:i+2],
            beam_bottom_def[i:i+2],
            beam_top_def[i:i+2],
            color=color, alpha=0.85,
        )

    # Deformed centerline
    ax.plot(x_orig, y_deformed, color="cyan", linewidth=2.5,
            label="Deformed centerline", zorder=5)
    ax.plot([0, params.length], [0, 0], "w--", linewidth=1.0,
            alpha=0.4, label="Original centerline")

    # Fixed wall
    ax.axvline(x=0, color="cyan", linewidth=4, alpha=0.9)
    ax.fill_betweenx([-h, h], [-0.025 * params.length, 0],
                     color="cyan", alpha=0.2)
    ax.text(-0.03 * params.length, 0, "FIXED\nWALL",
            color="cyan", ha="center", va="center", fontsize=8, alpha=0.9)

    # Load arrow at tip
    tip_x = params.length
    tip_y_def = y_deformed[-1]
    arrow_len = abs(beam_top_def[-1] - beam_bottom_def[-1]) * 0.8
    ax.annotate("",
        xy=(tip_x, tip_y_def - arrow_len),
        xytext=(tip_x, tip_y_def + arrow_len * 0.3),
        arrowprops=dict(arrowstyle="->", color="yellow",
                        lw=2.5, mutation_scale=20))
    ax.text(tip_x + 0.02 * params.length, tip_y_def,
            f"F = {params.load_force/1000:.1f} kN",
            color="yellow", va="center", fontsize=9)

    # Dimension arrow for max deflection
    ax.annotate("",
        xy=(tip_x * 0.95, tip_y_def),
        xytext=(tip_x * 0.95, 0),
        arrowprops=dict(arrowstyle="<->", color="lightgreen", lw=1.5))
    ax.text(tip_x * 0.90, tip_y_def / 2,
            f"δ = {results.max_deflection*1000:.2f} mm\n(×{scale_factor:.0f} scale)",
            color="lightgreen", ha="right", va="center", fontsize=9)

    # Colorbar for stress
    sm = plt.cm.ScalarMappable(cmap=cmap,
                                norm=mcolors.Normalize(0, results.max_von_mises / 1e6))
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, pad=0.02, shrink=0.85, label="Von Mises [MPa]")
    cbar.ax.yaxis.label.set_color("white")
    cbar.ax.yaxis.set_tick_params(color="white")
    plt.setp(cbar.ax.yaxis.get_ticklabels(), color="white")

    ax.set_xlabel("Beam Length x [m]", color="white", fontsize=11)
    ax.set_ylabel("Deflection y [m] (scaled)", color="white", fontsize=11)
    ax.set_title(
        f"Beam Deformation  (Scale factor ×{scale_factor:.0f})  |  "
        f"L={params.length}m  E={params.elastic_modulus/1e9:.0f}GPa  "
        f"F={params.load_force/1000:.1f}kN",
        color="white", fontsize=11, pad=8,
    )
    ax.tick_params(colors="white")
    ax.legend(loc="upper left", framealpha=0.3,
              facecolor="black", labelcolor="white", fontsize=9)
    for spine in ax.spines.values():
        spine.set_edgecolor("gray")

    # ── Bottom panel: Stress distribution along length ────────────────────────
    ax2 = axes[1]
    ax2.set_facecolor("#0d1117")

    ax2.fill_between(x_orig, 0, results.bending_stress / 1e6,
                     alpha=0.7, color="#ff6b6b", label="Bending stress")
    ax2.fill_between(x_orig, 0, results.shear_stress / 1e6,
                     alpha=0.5, color="#4ecdc4", label="Shear stress")
    ax2.plot(x_orig, results.von_mises_stress / 1e6,
             color="yellow", linewidth=2.0, label="Von Mises", zorder=5)

    ax2.set_xlabel("Beam Length x [m]", color="white", fontsize=10)
    ax2.set_ylabel("Stress [MPa]", color="white", fontsize=10)
    ax2.set_title("Stress Distribution Along Beam Length", color="white", fontsize=10)
    ax2.tick_params(colors="white")
    ax2.legend(loc="upper right", framealpha=0.3,
               facecolor="black", labelcolor="white", fontsize=8)
    ax2.grid(True, alpha=0.2, color="gray")
    for spine in ax2.spines.values():
        spine.set_edgecolor("gray")

    plt.tight_layout(pad=1.5)
    plt.savefig(output_path, dpi=dpi, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)

    return output_path


def generate_all_plots(
    mesh: BeamMesh,
    results: SimulationResults,
    params: BeamParameters,
    output_dir: Path,
) -> dict:
    """
    Generate all standard plots and return a dict of {name: Path}.

    Creates output_dir if it doesn't exist. In a production pipeline,
    this would also export a VTK file for ParaView rendering.

    Args:
        mesh:       BeamMesh.
        results:    SimulationResults.
        params:     BeamParameters.
        output_dir: Directory where PNGs will be saved.

    Returns:
        Dict with keys "stress_heatmap" and "deformation", values are saved Paths.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    heatmap_path = plot_stress_heatmap(
        mesh=mesh,
        results=results,
        params=params,
        output_path=output_dir / "stress_heatmap.png",
        field="von_mises_stress",
        colormap="hot",
    )

    deform_path = plot_deformation(
        mesh=mesh,
        results=results,
        params=params,
        output_path=output_dir / "deformation_plot.png",
    )

    return {
        "stress_heatmap": heatmap_path,
        "deformation":    deform_path,
    }
