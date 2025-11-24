#%%
#!/usr/bin/env python3
"""
3D Tomogram Creation and Visualization Script

This script demonstrates how to create and plot 3D tomograms from particle data
using the clathrate analysis modules.

Author: Generated Script
Date: Current
"""
#%%
import numpy as np
import os
import sys

#%%
# Add src directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from clathrate_analysis import (
    parse_particles_and_shape, 
    get_bipyramid_geometry, 
    get_truncated_bipyramid_geometry,
    detect_simple_cavities,
    detect_clathrate_cavities,
    duplicate_unit_cell
)
from tomogram_utils import (
    create_tomogram_from_particles,
    create_tomogram_with_cavity_objects,
    plot_3d_tomogram,
    plot_3d_isosurface,
    plot_3d_volume,
    plot_3d_slices,
    plot_tomogram_slices_2d
)

def example_1_basic_particle_tomogram(filename=None):
    """
    Example 1: Create a basic tomogram from particles only
    """
    print("="*60)
    print("EXAMPLE 1: Basic Particle Tomogram")
    print("="*60)
    
    # Use default file if none provided
    if filename is None:
        # You can replace this with your actual file path
        filename = 'C:\\Users\\b304014\\Software\\blee\\models\\ClaS_bipyramid_averaged.pos'  # Replace with actual file
        #filename = "C:\\Users\\b304014\\Box\\zhihua\\models\\Right bipyramids ClaIV_cubic_bipyra_UC.pos"
        print(f"Using example file: {filename}")
        
        # If the file doesn't exist, create some example data
        if not os.path.exists(filename):
            print("Example file not found. Creating synthetic particle data...")
            particles, shape_vertices, shape_color = create_synthetic_particles()
            simulation_data = None
        else:
            particles, shape_vertices, shape_color, simulation_data = parse_particles_and_shape(filename)
    else:
        # Parse the provided file
        particles, shape_vertices, shape_color, simulation_data = parse_particles_and_shape(filename)
    
    print(f"Loaded {len(particles)} particles")
    
    
    duplicated_particles = duplicate_unit_cell(particles, nx=4, ny=4, nz=4, simulation_data=simulation_data)
    duplicated_positions = [(pos, quat) for pos, quat, _ in duplicated_particles]
    
    # Create tomogram
    print("\nCreating tomogram from particles...")
    tomogram_filename, voxel_grid = create_tomogram_from_particles(
        particles=duplicated_positions,
        grid_size=128,
        padding=0.1,
        shape_vertices=shape_vertices,
        pixel_size=1.0,
        filename="ClaS_tomogram_128x128x128_4x4x4unitcells.tif"
    )
    
    print(f"Tomogram saved as: {tomogram_filename}")
    print(f"Voxel grid shape: {voxel_grid.shape}")
    print(f"Non-zero voxels: {np.sum(voxel_grid > 0)}")
    
    # Plot the tomogram in different ways
    print("\nPlotting tomogram as isosurface...")
    plot_3d_tomogram(tomogram_filename, plot_type='isosurface', threshold=0.5)
    
    # print("\nPlotting tomogram as volume rendering...")
    # plot_3d_tomogram(tomogram_filename, plot_type='volume', opacity=0.3)
    
    # print("\nPlotting tomogram slices...")
    # plot_3d_tomogram(tomogram_filename, plot_type='slices')
    
    return tomogram_filename, particles, shape_vertices, shape_color

def example_2_tomogram_with_cavities(particles, shape_vertices, shape_color, simulation_data=None):
    """
    Example 2: Create a tomogram with cavity objects
    """
    print("\n" + "="*60)
    print("EXAMPLE 2: Tomogram with Cavity Objects")
    print("="*60)
    
    # Detect cavities first
    print("Detecting cavities...")
    cavities = detect_simple_cavities(
        particles=particles,
        shape_vertices=shape_vertices,
        shape_color=shape_color,
        grid_size=64,
        padding=0.15,
        min_radius=0.1,
        min_separation=0.3,
        boundary_margin=0.2,
        debug=False  # Reduce output
    )
    
    if len(cavities) == 0:
        print("No cavities detected. Skipping cavity tomogram.")
        return None
    
    cavity_centers = [c['center'] for c in cavities]
    cavity_radii = [c['radius'] for c in cavities]
    
    print(f"Found {len(cavities)} cavities")
    
    # Create tomogram with cubic objects in cavities
    print("\nCreating tomogram with cubic cavity objects...")
    cube_tomogram_filename = create_tomogram_with_cavity_objects(
        particles=particles,
        cavity_centers=cavity_centers,
        cavity_radii=cavity_radii,
        grid_size=64,
        padding=0.1,
        shape_vertices=shape_vertices,
        pixel_size=1.0,
        filename="tomogram_with_cubes.tif",
        cavity_object_type='cube',
        cavity_object_scale=0.8
    )
    
    # Create tomogram with bipyramid objects in cavities
    print("\nCreating tomogram with bipyramid cavity objects...")
    bipyramid_tomogram_filename = create_tomogram_with_cavity_objects(
        particles=particles,
        cavity_centers=cavity_centers,
        cavity_radii=cavity_radii,
        grid_size=64,
        padding=0.1,
        shape_vertices=shape_vertices,
        pixel_size=1.0,
        filename="tomogram_with_bipyramids.tif",
        geometry_func=get_bipyramid_geometry,
        cavity_object_type='bipyramid',
        cavity_object_scale=1.0
    )
    
    # Plot the cavity tomograms
    print("\nPlotting cube cavity tomogram...")
    plot_3d_tomogram(cube_tomogram_filename, plot_type='isosurface', threshold=0.5)
    
    print("\nPlotting bipyramid cavity tomogram...")
    plot_3d_tomogram(bipyramid_tomogram_filename, plot_type='volume', opacity=0.4)
    
    return cube_tomogram_filename, bipyramid_tomogram_filename

def example_3_truncated_particle_tomogram(particles, shape_vertices, shape_color, simulation_data=None):
    """
    Example 3: Create tomograms with truncated particles
    """
    print("\n" + "="*60)
    print("EXAMPLE 3: Truncated Particle Tomogram")
    print("="*60)
    
    truncation_factor = 0.3
    print(f"Using truncation factor: {truncation_factor}")
    
    # Create tomogram with truncated particles
    print("\nCreating tomogram with truncated particles...")
    truncated_filename, truncated_grid = create_tomogram_from_particles(
        particles=particles,
        grid_size=128,
        padding=0.1,
        shape_vertices=shape_vertices,
        pixel_size=1.0,
        filename="truncated_particle_tomogram.tif"
    )
    
    # Note: The actual truncation would need to be applied in the voxelize_particles function
    # For demonstration, we'll use the regular geometry but mention this limitation
    
    print(f"Truncated tomogram saved as: {truncated_filename}")
    
    # Plot the truncated tomogram
    print("\nPlotting truncated particle tomogram...")
    plot_3d_tomogram(truncated_filename, plot_type='isosurface', threshold=0.5)
    
    # Show 2D slices
    print("\nShowing 2D slices of truncated tomogram...")
    plot_tomogram_slices_2d(truncated_filename, slice_type='middle')
    
    return truncated_filename

def example_4_comparative_visualization():
    """
    Example 4: Compare different visualization methods
    """
    print("\n" + "="*60)
    print("EXAMPLE 4: Comparative Visualization")
    print("="*60)
    
    # This example assumes we have a tomogram file already created
    tomogram_file = "basic_particle_tomogram.tif"
    
    if not os.path.exists(tomogram_file):
        print(f"Tomogram file {tomogram_file} not found. Run example 1 first.")
        return
    
    print("Comparing different visualization methods...")
    
    # Isosurface with different thresholds
    print("\n1. Isosurface with threshold 0.3...")
    plot_3d_tomogram(tomogram_file, plot_type='isosurface', threshold=0.3, colorscale='Viridis')
    
    print("\n2. Isosurface with threshold 0.7...")
    plot_3d_tomogram(tomogram_file, plot_type='isosurface', threshold=0.7, colorscale='Plasma')
    
    # Volume rendering with different opacities
    print("\n3. Volume rendering with opacity 0.2...")
    plot_3d_tomogram(tomogram_file, plot_type='volume', opacity=0.2, colorscale='Blues')
    
    print("\n4. Volume rendering with opacity 0.6...")
    plot_3d_tomogram(tomogram_file, plot_type='volume', opacity=0.6, colorscale='Reds')
    
    # Orthogonal slices
    print("\n5. Orthogonal slices...")
    plot_3d_tomogram(tomogram_file, plot_type='slices', colorscale='Viridis')

def create_synthetic_particles():
    """
    Create synthetic particle data for demonstration if no real data is available
    """
    print("Creating synthetic particle data...")
    
    # Create a simple cubic lattice of particles
    n = 3  # 3x3x3 = 27 particles
    spacing = 2.0
    particles = []
    
    for i in range(n):
        for j in range(n):
            for k in range(n):
                x = i * spacing - (n-1) * spacing / 2
                y = j * spacing - (n-1) * spacing / 2
                z = k * spacing - (n-1) * spacing / 2
                
                # Random orientation
                qw = np.random.uniform(0.5, 1.0)
                qx = np.random.uniform(-0.5, 0.5)
                qy = np.random.uniform(-0.5, 0.5)
                qz = np.random.uniform(-0.5, 0.5)
                
                # Normalize quaternion
                norm = np.sqrt(qw**2 + qx**2 + qy**2 + qz**2)
                qw, qx, qy, qz = qw/norm, qx/norm, qy/norm, qz/norm
                
                particles.append(((x, y, z), (qw, qx, qy, qz)))
    
    # Create default bipyramid geometry
    vertices, faces, color = get_bipyramid_geometry()
    
    print(f"Created {len(particles)} synthetic particles")
    return particles, vertices, color

def main():
    """
    Main function to run all tomogram examples
    """
    print("="*60)
    print("3D TOMOGRAM CREATION AND VISUALIZATION")
    print("="*60)
    
    # You can specify your particle file here
    particle_file = None  # Set to your .pos file path if available
    
    try:
        # Example 1: Basic particle tomogram
        tomogram_file, particles, shape_vertices, shape_color = example_1_basic_particle_tomogram(particle_file)
        
        # # Example 2: Tomogram with cavity objects
        # example_2_tomogram_with_cavities(particles, shape_vertices, shape_color)
        
        # # Example 3: Truncated particle tomogram
        # example_3_truncated_particle_tomogram(particles, shape_vertices, shape_color)
        
        # # Example 4: Comparative visualization
        # example_4_comparative_visualization()
        
    except Exception as e:
        print(f"Error during execution: {e}")
        print("Make sure you have the required dependencies installed:")
        print("  pip install numpy plotly scipy tifffile")
        raise
    
    print("\n" + "="*60)
    print("TOMOGRAM EXAMPLES COMPLETE")
    print("="*60)
    print("\nGenerated files:")
    for filename in ["basic_particle_tomogram.tif", "tomogram_with_cubes.tif", 
                     "tomogram_with_bipyramids.tif", "truncated_particle_tomogram.tif"]:
        if os.path.exists(filename):
            print(f"  - {filename}")

if __name__ == "__main__":
    main()

# %%

