# 3D Tomogram Creation and Visualization Scripts

This directory contains three scripts that demonstrate how to create and plot 3D tomograms from particle data using the clathrate analysis modules.

## Prerequisites

Make sure you have the required Python packages installed:

```bash
pip install numpy plotly scipy tifffile matplotlib pandas streamlit
```

## Scripts Overview

### 1. `simple_tomogram_plot.py` - Basic Tomogram Creation

**Purpose**: Demonstrates the basic workflow to create and visualize a 3D tomogram from particle data.

**Usage**:
```bash
# Using your own particle file
python simple_tomogram_plot.py your_particles.pos

# Using synthetic data (no arguments)
python simple_tomogram_plot.py
```

**What it does**:
- Loads particle data from a .pos file or creates synthetic data
- Creates a basic tomogram (TIFF file) from particle positions
- Shows 4 different visualization methods:
  1. Isosurface rendering
  2. Volume rendering  
  3. Orthogonal slices (3D)
  4. 2D slice views

**Output**: `simple_tomogram.tif`

### 2. `spherical_tomogram_example.py` - Fast Spherical Particles

**Purpose**: Demonstrates creating tomograms using simple spherical particles for faster computation.

**Usage**:
```bash
# Using your own particle file (positions only, shapes ignored)
python spherical_tomogram_example.py your_particles.pos

# Using synthetic data
python spherical_tomogram_example.py
```

**What it does**:
- Creates tomograms using spherical approximations of particles
- Much faster than complex geometries
- Shows three examples:
  1. Basic spherical particles
  2. Spherical particles with spherical cavity objects
  3. Large assembly using unit cell duplication
- Provides performance and memory advantages

**Output**: 
- `spherical_particles_tomogram.tif`
- `spherical_particles_with_cavities.tif`
- `large_spherical_assembly.tif`

### 3. `quick_spherical_tomogram.py` - Command Line Tool

**Purpose**: Simple command-line tool for quickly creating spherical particle tomograms.

**Usage**:
```bash
# Basic usage
python quick_spherical_tomogram.py

# With custom parameters
python quick_spherical_tomogram.py --radius 0.2 --grid-size 128

# Create synthetic particles
python quick_spherical_tomogram.py --synthetic 100

# Load from file
python quick_spherical_tomogram.py your_file.pos --output my_tomogram.tif
```

**What it does**:
- Command-line interface for quick tomogram creation
- Customizable sphere radius and grid size
- Performance statistics and timing
- Optional visualization

**Output**: Customizable filename (default: `quick_spherical.tif`)

### 4. `spherical_vs_complex_comparison.py` - Performance Comparison

**Purpose**: Compares spherical vs complex geometry tomogram creation performance.

**Usage**:
```bash
python spherical_vs_complex_comparison.py
```

**What it does**:
- Benchmarks different voxelization methods
- Compares computation time and memory usage
- Analyzes volume differences between methods
- Provides recommendations for different use cases

**Output**: 
- `comparison_spherical.tif`
- `comparison_complex.tif`
- Performance analysis

### 5. `advanced_tomogram_with_cavities.py` - Cavity Detection and Objects

**Purpose**: Demonstrates advanced features including cavity detection and placing objects in cavities.

**Usage**:
```bash
# Using your own particle file
python advanced_tomogram_with_cavities.py your_particles.pos

# Using synthetic clathrate-like data
python advanced_tomogram_with_cavities.py
```

**What it does**:
- Detects cavities in the particle assembly using advanced algorithms
- Creates tomograms with objects placed in detected cavities:
  - Cubic objects in cavities
  - Bipyramid objects in cavities
- Creates a comparison tomogram showing particles vs cavity objects
- Provides detailed analysis of voxel distribution

**Output**: 
- `advanced_tomogram_cubes.tif`
- `advanced_tomogram_bipyramids.tif`
- `comparison_tomogram.tif`

### 3. `create_tomogram_script.py` - Comprehensive Examples

**Purpose**: Comprehensive script showing all available tomogram creation and visualization features.

**Usage**:
```bash
python create_tomogram_script.py
```

**What it does**:
- **Example 1**: Basic particle tomogram with multiple visualization methods
- **Example 2**: Tomograms with cavity objects (cubes and bipyramids)
- **Example 3**: Truncated particle tomograms
- **Example 4**: Comparative visualization with different parameters

**Output**: Multiple TIFF files demonstrating different approaches

## Understanding the Output

### TIFF Files
All scripts generate TIFF files that can be:
- Opened in ImageJ/FIJI for detailed analysis
- Loaded into other tomography software
- Used for further computational analysis

### Voxel Values
The voxel grids use different values to represent different materials:
- `0.0`: Empty space
- `1.0`: Original particles  
- `2.0`: Cavity objects (when present)

### Visualization Types

1. **Isosurface**: Shows surfaces at a specific threshold value
   - Good for seeing particle shapes and boundaries
   - Adjustable threshold parameter

2. **Volume Rendering**: Shows internal structure with opacity
   - Good for seeing density variations
   - Adjustable opacity parameter

3. **Orthogonal Slices**: Shows 2D cross-sections in 3D space
   - Good for understanding internal structure
   - Shows XY, XZ, and YZ slices

4. **2D Slices**: Traditional 2D slice views
   - Good for detailed analysis
   - Can show middle slice or all slices

## Customization

### Cavity Detection Parameters

You can adjust cavity detection by modifying these parameters in the scripts:

```python
cavities = detect_simple_cavities(
    particles=particles,
    shape_vertices=shape_vertices,
    grid_size=128,           # Higher = more detail, slower
    padding=0.15,            # Boundary padding
    min_radius=0.08,         # Minimum cavity size
    min_separation=0.2,      # Minimum distance between cavities
    boundary_margin=0.15,    # Distance from edges
    min_surrounding_particles=4,           # Particles around cavity
    max_empty_neighbors_fraction=0.5,      # Empty space tolerance
    debug=True               # Show detailed output
)
```

### Tomogram Parameters

```python
voxel_grid, edges = voxelize_particles(
    particles=particles,
    grid_size=64,            # 64x64x64 voxels (increase for higher resolution)
    padding=0.1,             # 10% padding around structure
    shape_vertices=shape_vertices
)
```

### Visualization Parameters

```python
plot_3d_tomogram(
    filename, 
    plot_type='isosurface',  # 'isosurface', 'volume', 'slices'
    threshold=0.5,           # For isosurface (0-1)
    opacity=0.3,             # For volume rendering (0-1)
    colorscale='Viridis'     # Color scheme
)
```

## Troubleshooting

### No Cavities Detected
If cavity detection fails:
1. Reduce `min_radius` (try 0.05 or smaller)
2. Reduce `min_separation` (try 0.1)
3. Increase `max_empty_neighbors_fraction` (try 0.7)
4. Reduce `min_surrounding_particles` (try 2)

### Memory Issues
If you run out of memory:
1. Reduce `grid_size` (try 64 or 32)
2. Reduce the number of particles
3. Use smaller `padding`

### Slow Performance
To speed up processing:
1. Use smaller `grid_size`
2. Reduce number of particles
3. Set `debug=False` in cavity detection

## File Formats

### Input (.pos files)
The scripts expect .pos files in the format used by the clathrate analysis modules. These should contain:
- Particle positions (x, y, z)
- Particle orientations (quaternions)
- Shape definitions
- Simulation metadata

### Output (.tif files)  
Multi-page TIFF files compatible with:
- ImageJ/FIJI
- Python (tifffile, scikit-image)
- MATLAB
- Commercial tomography software

## Spherical vs Complex Geometry

### When to Use Spherical Particles

**Advantages:**
- **10-50x faster** computation
- **Lower memory usage** (no complex hulls)
- **Simpler implementation** 
- **Orientation-independent**
- **Perfect for prototyping** and testing
- **Good for large assemblies** where speed matters

**Use Cases:**
- Rapid prototyping and testing
- Large-scale simulations (>1000 particles)
- Parameter sweeping and optimization
- Initial cavity detection studies
- Educational demonstrations

### When to Use Complex Geometry

**Advantages:**
- **Realistic particle shapes**
- **Captures orientation effects**
- **Better scientific accuracy**
- **Material-specific modeling**

**Use Cases:**
- Final production simulations
- Precise cavity analysis
- Anisotropic material studies
- Publication-quality results

### Performance Guidelines

| Particles | Spherical Time | Complex Time | Recommendation |
|-----------|---------------|--------------|----------------|
| < 50      | < 1 sec       | < 5 sec      | Either method |
| 50-200    | < 5 sec       | 30-60 sec    | Start with spherical |
| 200-1000  | 10-30 sec     | 5-30 min     | Use spherical first |
| > 1000    | 1-5 min       | > 1 hour     | Spherical recommended |

## Integration with Existing Tools

These scripts work seamlessly with the existing modules:
- `clathrate_analysis.py`: Core analysis functions
- `tomogram_utils.py`: Tomogram creation and visualization
- `truncation_analysis.py`: Truncated particle analysis
- `clathrate_gui.py`: Streamlit GUI interface

## Examples with Real Data

If you have a real .pos file, try:

```bash
# Basic workflow
python simple_tomogram_plot.py path/to/your/structure.pos

# Advanced cavity analysis  
python advanced_tomogram_with_cavities.py path/to/your/structure.pos
```

The scripts will automatically parse your file and generate appropriate tomograms and visualizations.

