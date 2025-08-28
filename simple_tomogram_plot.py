#!/usr/bin/env python3
"""
Simple 3D Tomogram Plotting Script

This script demonstrates the basic workflow to create and plot a 3D tomogram.

Usage:
    python simple_tomogram_plot.py [particle_file.pos]

If no file is provided, synthetic data will be generated.
"""

import numpy as np
import os
import sys

# Add src directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from clathrate_analysis import (
    parse_particles_and_shape, 
    get_bipyramid_geometry,
    voxelize_particles
)
from tomogram_utils import (
    save_voxel_grid_as_tiff,
    plot_3d_tomogram,
    plot_3d_isosurface,
    plot_3d_volume,
    plot_tomogram_slices_2d
)

def create_simple_tomogram(particles, shape_vertices, output_filename="simple_tomogram.tif"):
    """
    Create a simple tomogram from particles
    """
    print(f"Creating tomogram from {len(particles)} particles...")
    
    # Voxelize the particles
    voxel_grid, edges = voxelize_particles(
        particles=particles,
        grid_size=64,  # Smaller for faster processing
        padding=0.1,
        shape_vertices=shape_vertices
    )
    
    print(f"Voxel grid shape: {voxel_grid.shape}")
    print(f"Filled voxels: {np.sum(voxel_grid > 0)}")
    
    # Save as TIFF
    save_voxel_grid_as_tiff(voxel_grid, output_filename, pixel_size=1.0)
    
    return output_filename, voxel_grid

def plot_tomogram_examples(tomogram_filename):
    """
    Show different ways to visualize the tomogram
    """
    print(f"\nVisualizing tomogram: {tomogram_filename}")
    
    # 1. Isosurface plot
    print("1. Creating isosurface plot...")
    plot_3d_tomogram(tomogram_filename, plot_type='isosurface', threshold=0.5)
    
    # 2. Volume rendering
    print("2. Creating volume rendering...")
    plot_3d_tomogram(tomogram_filename, plot_type='volume', opacity=0.3)
    
    # 3. Orthogonal slices
    print("3. Creating orthogonal slices...")
    plot_3d_tomogram(tomogram_filename, plot_type='slices')
    
    # 4. 2D slice view
    print("4. Creating 2D slice views...")
    plot_tomogram_slices_2d(tomogram_filename, slice_type='middle')

def create_synthetic_data():
    """
    Create simple synthetic particle data for demonstration
    """
    print("Creating synthetic particle data...")
    
    # Create particles in a simple arrangement
    particles = []
    positions = [
        (0, 0, 0), (2, 0, 0), (0, 2, 0), (0, 0, 2),
        (2, 2, 0), (2, 0, 2), (0, 2, 2), (2, 2, 2),
        (1, 1, 1)  # Center particle
    ]
    
    for pos in positions:
        # Simple orientation (no rotation)
        quat = (1, 0, 0, 0)
        particles.append((pos, quat))
    
    # Get default bipyramid geometry
    vertices, _, _ = get_bipyramid_geometry()
    
    return particles, vertices

def main():
    """
    Main function
    """
    print("="*50)
    print("SIMPLE 3D TOMOGRAM PLOTTING")
    print("="*50)
    
    # Check if a file was provided as command line argument
    if len(sys.argv) > 1:
        particle_file = sys.argv[1]
        print(f"Loading particles from: {particle_file}")
        
        try:
            particles, shape_vertices, shape_color, simulation_data = parse_particles_and_shape(particle_file)
            print(f"Loaded {len(particles)} particles from file")
        except Exception as e:
            print(f"Error loading file: {e}")
            print("Using synthetic data instead...")
            particles, shape_vertices = create_synthetic_data()
    else:
        print("No particle file provided. Using synthetic data...")
        particles, shape_vertices = create_synthetic_data()
    
    # Create the tomogram
    tomogram_filename, voxel_grid = create_simple_tomogram(particles, shape_vertices)
    
    # Plot the tomogram in different ways
    plot_tomogram_examples(tomogram_filename)
    
    print(f"\nTomogram saved as: {tomogram_filename}")
    print("Done!")

if __name__ == "__main__":
    main()

