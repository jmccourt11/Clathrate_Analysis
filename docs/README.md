# Clathrate Analysis Package

A comprehensive Python package for analyzing clathrate structures and particle assemblies, with a focus on triangular bipyramids and their truncated variants.

## Features

### 🧪 Core Analysis
- **Particle Visualization**: 3D visualization of triangular bipyramids and assemblies
- **Cavity Detection**: Automatic detection and analysis of clathrate cavities
- **Volume Analysis**: Precise volume calculations for particles and cavities
- **Structure Analysis**: Comprehensive structural analysis with detailed statistics

### 🔧 Advanced Features
- **Truncation Analysis**: Study the effects of particle truncation on cavity properties
- **FFT Analysis**: 3D FFT and 2D projection analysis for structure factor determination
- **Tomogram Generation**: Create and analyze 3D tomograms from particle assemblies
- **Unit Cell Duplication**: Build larger assemblies from unit cells

### 🖥️ User Interface
- **Streamlit GUI**: Interactive web-based interface for easy analysis
- **Command Line Tools**: Script-based analysis for automation
- **Jupyter Integration**: Notebook-friendly functions for research workflows

## Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Quick Install
```bash
# Clone the repository
git clone https://github.com/yourusername/clathrate-analysis.git
cd clathrate-analysis

# Install the package
pip install -e .

# Install GUI dependencies (optional)
pip install -e .[gui]
```

### Development Install
```bash
# Install with development dependencies
pip install -e .[dev]
```

## Quick Start

### Using the GUI
```bash
# Launch the interactive GUI
python main.py gui
# or
streamlit run src/clathrate_gui.py
```

### Using Python Scripts
```python
from src.clathrate_analysis import parse_particles_and_shape, plot_clathrate_cavities

# Load your particle data
filename = 'path/to/your/particle_data.pos'
particles, shape_vertices, shape_color, simulation_data = parse_particles_and_shape(filename)

# Analyze cavities
cavity_centers, cavity_volumes, cavity_radii = plot_clathrate_cavities(
    particles=particles,
    shape_vertices=shape_vertices,
    shape_color=shape_color,
    show_particles=True,
    show_cavities=True,
    show_spheres=True
)
```

### Running Examples
```bash
# Run the comprehensive example
python main.py example

# Run the truncated cavity analysis example
python examples/truncated_cavity_example.py
```

## Package Structure

```
clathrate-analysis/
├── src/                          # Main package source
│   ├── __init__.py              # Package initialization
│   ├── clathrate_analysis.py    # Core analysis functions
│   ├── truncation_analysis.py   # Truncation-specific analysis
│   ├── tomogram_utils.py        # Tomogram handling utilities
│   └── clathrate_gui.py         # Streamlit GUI
├── tests/                       # Test suite
│   ├── __init__.py
│   ├── test_suite.py           # Comprehensive test suite
│   ├── test_cavity_detection.py
│   ├── test_cavity_centers.py
│   └── debug_cavity_detection.py
├── examples/                    # Example scripts
│   ├── __init__.py
│   ├── example_usage.py        # Comprehensive usage example
│   └── truncated_cavity_example.py
├── docs/                       # Documentation
│   └── README.md
├── main.py                     # Main entry point
├── setup.py                    # Package setup
└── requirements.txt            # Dependencies
```

## Core Functions

### Data Parsing
- `parse_particles_and_shape()`: Parse particle data and shape definitions from files

### Visualization
- `plot_particles()`: Visualize particle assemblies with optional unit cell duplication
- `plot_clathrate_cavities()`: Visualize particles with detected cavities
- `plot_cavity_spheres()`: Show cavity spheres at optimal centers

### Analysis
- `detect_clathrate_cavities()`: Detect cavities in particle assemblies
- `analyze_clathrate_structure()`: Comprehensive structural analysis
- `voxelize_particles()`: Convert particles to 3D voxel grids

### FFT Analysis
- `plot_fft_magnitude()`: 3D FFT magnitude visualization
- `create_2d_projections()`: Create 2D projections of 3D data
- `plot_projections_with_ffts()`: Compare projections with their FFTs

### Truncation Analysis
- `get_truncated_bipyramid_geometry()`: Generate truncated particle geometries
- `analyze_truncation_effects()`: Study truncation effects on properties
- `plot_truncation_cavity_comparison()`: Compare cavities across truncation levels

## File Format

The package expects particle data files with the following format:
```
//date: [date]
#[data] [column headers]
[simulation values]
[translation data]
[zoom factor]
[box matrix]
shape [shape definition]
[particle data - position and quaternion]
```

## Examples

### Basic Cavity Analysis
```python
from src.clathrate_analysis import *

# Load data
particles, shape_vertices, shape_color, simulation_data = parse_particles_and_shape('data.pos')

# Detect cavities
cavity_centers, cavity_volumes, cavity_radii = plot_clathrate_cavities(
    particles=particles,
    shape_vertices=shape_vertices,
    shape_color=shape_color,
    show_spheres=True
)

# Analyze structure
analysis = analyze_clathrate_structure(
    particles=particles,
    shape_vertices=shape_vertices,
    shape_color=shape_color
)
```

### Truncation Analysis
```python
from src.truncation_analysis import *

# Compare different truncation levels
plot_truncation_cavity_comparison(
    particles=particles,
    shape_vertices=shape_vertices,
    shape_color=shape_color,
    truncation_factors=[0.0, 0.2, 0.4, 0.6, 0.8]
)
```

### FFT Analysis
```python
# Create voxel grid
voxel_grid, edges = voxelize_particles(particles, grid_size=128)

# Plot 3D FFT
plot_fft_magnitude(voxel_grid, edges, log_scale=True)

# Create 2D projections
projections = create_2d_projections(voxel_grid)
plot_projections_with_ffts(projections)
```

## Testing

Run the comprehensive test suite:
```bash
python main.py test
# or
python tests/test_suite.py
```

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## Development

### Code Style
The project uses:
- **Black** for code formatting
- **Flake8** for linting
- **Pytest** for testing

### Running Tests
```bash
# Install development dependencies
pip install -e .[dev]

# Run tests
pytest tests/

# Run with coverage
pytest --cov=src tests/
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Citation

If you use this package in your research, please cite:

```bibtex
@software{clathrate_analysis,
  title={Clathrate Analysis Package},
  author={Your Name},
  year={2024},
  url={https://github.com/yourusername/clathrate-analysis}
}
```

## Support

- **Issues**: Report bugs and request features on [GitHub Issues](https://github.com/yourusername/clathrate-analysis/issues)
- **Documentation**: See the `docs/` directory for detailed documentation
- **Examples**: Check the `examples/` directory for usage examples

## Acknowledgments

- Built with [Plotly](https://plotly.com/) for interactive 3D visualization
- Uses [Streamlit](https://streamlit.io/) for the web interface
- Leverages [SciPy](https://scipy.org/) for scientific computing
- Powered by [NumPy](https://numpy.org/) for numerical operations 