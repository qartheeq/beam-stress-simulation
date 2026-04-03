# Beam Stress Simulation

A small Python CAE-style pipeline for simulating a cantilever beam under a point load.

The project walks through the same stages you would expect in an industrial simulation workflow:

1. Parse input parameters from the command line
2. Generate beam geometry
3. Build a finite element mesh
4. Solve the beam response
5. Export stress and deformation plots

It is intended as a learning and prototyping project, with lightweight Python modules standing in for tools such as FreeCAD, GMSH, CalculiX, and ParaView.

## Features

- CLI-driven beam simulation
- Support for rectangular, circular, and I-beam cross-sections
- Material and loading parameters in SI units
- Finite-element-style beam solve using NumPy
- Stress heatmap and deformation plot output as PNG files
- Clear logging for each pipeline stage

## Requirements

- Python 3.10 or newer
- `numpy`
- `matplotlib`

Install dependencies with:

```bash
pip install -r requirements.txt
```

## Quick Start

Run the simulation with default parameters:

```bash
python main.py
```

Run a custom case:

```bash
python main.py --length 2.0 --force 5000 --elements 40
```

Show all available options:

```bash
python main.py --help
```

## Command-Line Options

You can customize the beam with these flags:

- `--length` beam length in meters
- `--width` cross-section width in meters
- `--height` cross-section height in meters
- `--elastic-modulus` Young's modulus in Pascals
- `--density` material density in kg/m^3
- `--poisson` Poisson's ratio
- `--material` material label for display only
- `--force` tip load in Newtons
- `--elements` number of beam elements
- `--section` cross-section shape: `rectangular`, `circular`, or `I-beam`

Example:

```bash
python main.py --length 1.5 --width 0.04 --height 0.08 --force 2500 --elements 30 --section rectangular
```

## Output

Results are written to the `outputs/` directory:

- `outputs/stress_heatmap.png`
- `outputs/deformation_plot.png`

The console log also reports:

- geometry properties
- mesh statistics
- solver convergence
- maximum deflection
- maximum bending stress
- maximum von Mises stress
- reaction force and moment

## Project Structure

```text
main.py                 Pipeline orchestrator
input/                  CLI parsing and simulation parameters
geometry/               Beam geometry generation
mesh/                   Mesh generation
simulation/             FEA solver and result data
visualization/          Plot generation
outputs/                Generated PNG figures
```

## Notes

- All calculations use SI units.
- The pipeline is designed as a clean educational approximation of an industrial CAE workflow.
- Generated files in `outputs/` are ignored by git.

## License

Add a license here if you want to publish the project publicly.
