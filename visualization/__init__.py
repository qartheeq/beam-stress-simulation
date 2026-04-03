# Visualization module: converts simulation results to plots.
# In production CAE, this is handled by ParaView (VTK-based).
# We use matplotlib as a lightweight equivalent.
from .plotter import plot_stress_heatmap, plot_deformation, generate_all_plots

__all__ = ["plot_stress_heatmap", "plot_deformation", "generate_all_plots"]
