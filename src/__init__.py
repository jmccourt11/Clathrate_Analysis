"""
Clathrate Analysis Package

A comprehensive package for analyzing clathrate structures and particle assemblies.
"""

from .clathrate_analysis import (
    parse_particles_and_shape,
    get_bipyramid_geometry,
    get_truncated_bipyramid_geometry,
    duplicate_unit_cell,
    plot_particles,
    voxelize_particles,
    detect_clathrate_cavities,
    plot_clathrate_cavities,
    analyze_clathrate_structure,
    plot_cavity_analysis,
    plot_cavity_spheres,
    create_2d_projections,
    plot_projections_with_ffts,
    plot_fft_magnitude
)

from .truncation_analysis import (
    analyze_truncation_effects,
    plot_truncation_comparison,
    plot_particles_with_truncation,
    plot_truncation_cavity_comparison
)

from .tomogram_utils import (
    create_tomogram_from_particles,
    plot_3d_tomogram,
    save_voxel_grid_as_tiff,
    load_tomogram_tiff
)

__version__ = "1.0.0"
__author__ = "Clathrate Analysis Team"

__all__ = [
    # Core analysis functions
    'parse_particles_and_shape',
    'get_bipyramid_geometry',
    'get_truncated_bipyramid_geometry',
    'duplicate_unit_cell',
    'plot_particles',
    'voxelize_particles',
    'detect_clathrate_cavities',
    'plot_clathrate_cavities',
    'analyze_clathrate_structure',
    'plot_cavity_analysis',
    'plot_cavity_spheres',
    
    # FFT and projection functions
    'create_2d_projections',
    'plot_projections_with_ffts',
    'plot_fft_magnitude',
    
    # Truncation analysis
    'analyze_truncation_effects',
    'plot_truncation_comparison',
    'plot_particles_with_truncation',
    'plot_truncation_cavity_comparison',
    
    # Tomogram utilities
    'create_tomogram_from_particles',
    'plot_3d_tomogram',
    'save_voxel_grid_as_tiff',
    'load_tomogram_tiff'
] 