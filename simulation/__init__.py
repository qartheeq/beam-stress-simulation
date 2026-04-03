# Simulation module: finite element analysis solver.
# Mocks the CalculiX CCX solver API while using real Euler-Bernoulli math.
from .results import SimulationResults
from .solver import run_simulation, get_simulation_report

__all__ = ["SimulationResults", "run_simulation", "get_simulation_report"]
