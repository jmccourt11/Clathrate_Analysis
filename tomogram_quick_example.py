#!/usr/bin/env python3
"""
Quick Tomogram Example

This is a minimal example showing the key steps to create and plot a 3D tomogram.
"""

# Key imports needed (assuming you're in the right directory)
from src.clathrate_analysis import parse_particles_and_shape, voxelize_particles
from src.tomogram_utils import save_voxel_grid_as_tiff, plot_3d_tomogram

def quick_tomogram_example(particle_file):
    """
    Minimal example of creating and plotting a tomogram
    """
    
    # Step 1: Load particle data
    particles, shape_vertices, shape_color, simulation_data = parse_particles_and_shape(particle_file)
    print(f"Loaded {len(particles)} particles")
    
    # Step 2: Create voxel grid
    voxel_grid, edges = voxelize_particles(
        particles=particles,
        grid_size=64,                    # 64x64x64 grid
        padding=0.1,                     # 10% padding
        shape_vertices=shape_vertices
    )
    print(f"Created voxel grid: {voxel_grid.shape}")
    
    # Step 3: Save as TIFF
    tomogram_filename = "quick_tomogram.tif"
    save_voxel_grid_as_tiff(voxel_grid, tomogram_filename, pixel_size=1.0)
    print(f"Saved tomogram: {tomogram_filename}")
    
    # Step 4: Visualize
    print("Creating 3D visualization...")
    plot_3d_tomogram(tomogram_filename, plot_type='isosurface', threshold=0.5)
    
    return tomogram_filename

# Example usage:
# tomogram_file = quick_tomogram_example("your_structure.pos")

