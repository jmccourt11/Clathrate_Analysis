#!/usr/bin/env python3
"""
Advanced Tomogram with Cavity Objects

This script demonstrates how to create tomograms with cavity detection and 
place objects (cubes or bipyramids) in the detected cavities.

Usage:
    python advanced_tomogram_with_cavities.py [particle_file.pos]
"""

import numpy as np
import os
import sys

# Add src directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from clathrate_analysis import (
    parse_particles_and_shape, 
    get_bipyramid_geometry,
    detect_simple_cavities,
    plot_cavity_objects
)
from tomogram_utils import (
    create_tomogram_with_cavity_objects,
    plot_3d_tomogram,
    save_voxel_grid_as_tiff
)

def detect_and_analyze_cavities(particles, shape_vertices):
    """
    Detect cavities in the particle assembly
    """
    print("Detecting cavities...")
    
    cavities = detect_simple_cavities(
        particles=particles,
        shape_vertices=shape_vertices,
        grid_size=128,
        padding=0.15,
        min_radius=0.08,      # Smaller minimum radius
        min_separation=0.2,   # Closer cavities allowed
        boundary_margin=0.15,
        min_surrounding_particles=4,  # Relaxed requirement
        max_empty_neighbors_fraction=0.5,
        debug=True
    )
    
    if len(cavities) == 0:
        print("No cavities detected with current parameters.")
        print("Try adjusting the detection parameters:")
        print("  - Reduce min_radius")
        print("  - Reduce min_separation")
        print("  - Increase max_empty_neighbors_fraction")
        return [], []
    
    cavity_centers = [c['center'] for c in cavities]
    cavity_radii = [c['radius'] for c in cavities]
    
    print(f"\nDetected {len(cavities)} cavities:")
    for i, cavity in enumerate(cavities):
        print(f"  Cavity {i+1}: center={cavity['center']}, radius={cavity['radius']:.4f}")
    
    return cavity_centers, cavity_radii

def create_cavity_tomograms(particles, shape_vertices, cavity_centers, cavity_radii):
    """
    Create tomograms with different types of cavity objects
    """
    if len(cavity_centers) == 0:
        print("No cavities to create tomograms with.")
        return []
    
    tomogram_files = []
    
    # 1. Tomogram with cubic objects in cavities
    print("\n" + "="*50)
    print("Creating tomogram with CUBIC cavity objects...")
    print("="*50)
    
    cube_filename = create_tomogram_with_cavity_objects(
        particles=particles,
        cavity_centers=cavity_centers,
        cavity_radii=cavity_radii,
        grid_size=128,
        padding=0.1,
        shape_vertices=shape_vertices,
        filename="advanced_tomogram_cubes.tif",
        cavity_object_type='cube',
        cavity_object_scale=0.8  # 80% of cavity radius
    )
    tomogram_files.append(cube_filename)
    
    # 2. Tomogram with bipyramid objects in cavities
    print("\n" + "="*50)
    print("Creating tomogram with BIPYRAMID cavity objects...")
    print("="*50)
    
    bipyramid_filename = create_tomogram_with_cavity_objects(
        particles=particles,
        cavity_centers=cavity_centers,
        cavity_radii=cavity_radii,
        grid_size=128,
        padding=0.1,
        shape_vertices=shape_vertices,
        filename="advanced_tomogram_bipyramids.tif",
        geometry_func=get_bipyramid_geometry,
        cavity_object_type='bipyramid',
        cavity_object_scale=1.2  # 120% of cavity radius
    )
    tomogram_files.append(bipyramid_filename)
    
    return tomogram_files

def visualize_cavity_tomograms(tomogram_files):
    """
    Visualize the created tomograms
    """
    for i, filename in enumerate(tomogram_files):
        print(f"\n" + "="*50)
        print(f"VISUALIZING: {filename}")
        print("="*50)
        
        # Isosurface visualization
        print("Creating isosurface plot...")
        plot_3d_tomogram(filename, plot_type='isosurface', threshold=0.5, colorscale='Viridis')
        
        # Volume rendering
        print("Creating volume rendering...")
        plot_3d_tomogram(filename, plot_type='volume', opacity=0.4, colorscale='Plasma')

def create_comparison_tomogram(particles, shape_vertices, cavity_centers, cavity_radii):
    """
    Create a direct voxel grid comparison showing particles and cavity objects
    """
    if len(cavity_centers) == 0:
        print("No cavities for comparison tomogram.")
        return
    
    print("\n" + "="*50)
    print("Creating COMPARISON voxel grid...")
    print("="*50)
    
    # Use the plot_cavity_objects function to get a voxel grid
    voxel_grid, edges = plot_cavity_objects(
        particles=particles,
        cavity_centers=cavity_centers,
        cavity_radii=cavity_radii,
        shape_vertices=shape_vertices,
        show_particles=True,
        cavity_object_type='cube',
        cavity_object_scale=0.9
    )
    
    # Save this special voxel grid
    comparison_filename = "comparison_tomogram.tif"
    save_voxel_grid_as_tiff(voxel_grid, comparison_filename, pixel_size=1.0)
    
    print(f"Comparison tomogram saved as: {comparison_filename}")
    
    # Analyze the voxel grid
    particles_voxels = np.sum(voxel_grid == 1.0)
    cavity_objects_voxels = np.sum(voxel_grid == 2.0)
    empty_voxels = np.sum(voxel_grid == 0.0)
    total_voxels = voxel_grid.size
    
    print(f"\nVoxel Grid Analysis:")
    print(f"  Total voxels: {total_voxels}")
    print(f"  Particle voxels: {particles_voxels} ({particles_voxels/total_voxels*100:.1f}%)")
    print(f"  Cavity object voxels: {cavity_objects_voxels} ({cavity_objects_voxels/total_voxels*100:.1f}%)")
    print(f"  Empty voxels: {empty_voxels} ({empty_voxels/total_voxels*100:.1f}%)")
    
    # Visualize the comparison
    plot_3d_tomogram(comparison_filename, plot_type='isosurface', threshold=0.5)
    
    return comparison_filename

def create_synthetic_clathrate_data():
    """
    Create a more realistic synthetic clathrate-like structure
    """
    print("Creating synthetic clathrate-like data...")
    
    particles = []
    
    # Create a cage-like structure with a central cavity
    # Outer shell particles
    radius = 3.0
    n_particles = 20
    
    for i in range(n_particles):
        theta = 2 * np.pi * i / n_particles
        phi = np.pi * (0.3 + 0.4 * (i % 3))  # Vary height
        
        x = radius * np.sin(phi) * np.cos(theta)
        y = radius * np.sin(phi) * np.sin(theta)
        z = radius * np.cos(phi)
        
        # Random orientation
        qw = np.random.uniform(0.8, 1.0)
        qx = np.random.uniform(-0.3, 0.3)
        qy = np.random.uniform(-0.3, 0.3)
        qz = np.random.uniform(-0.3, 0.3)
        
        # Normalize quaternion
        norm = np.sqrt(qw**2 + qx**2 + qy**2 + qz**2)
        quat = (qw/norm, qx/norm, qy/norm, qz/norm)
        
        particles.append(((x, y, z), quat))
    
    # Add some inner particles
    for i in range(8):
        x = np.random.uniform(-1.5, 1.5)
        y = np.random.uniform(-1.5, 1.5)
        z = np.random.uniform(-1.5, 1.5)
        
        quat = (1, 0, 0, 0)  # No rotation
        particles.append(((x, y, z), quat))
    
    # Get default bipyramid geometry
    vertices, _, _ = get_bipyramid_geometry()
    
    print(f"Created {len(particles)} synthetic particles in clathrate-like arrangement")
    return particles, vertices

def main():
    """
    Main function
    """
    print("="*60)
    print("ADVANCED TOMOGRAM WITH CAVITY DETECTION")
    print("="*60)
    
    # Load or create particle data
    if len(sys.argv) > 1:
        particle_file = sys.argv[1]
        print(f"Loading particles from: {particle_file}")
        
        try:
            particles, shape_vertices, shape_color, simulation_data = parse_particles_and_shape(particle_file)
            print(f"Loaded {len(particles)} particles from file")
        except Exception as e:
            print(f"Error loading file: {e}")
            print("Using synthetic clathrate data instead...")
            particles, shape_vertices = create_synthetic_clathrate_data()
    else:
        print("No particle file provided. Using synthetic clathrate data...")
        particles, shape_vertices = create_synthetic_clathrate_data()
    
    # Detect cavities
    cavity_centers, cavity_radii = detect_and_analyze_cavities(particles, shape_vertices)
    
    if len(cavity_centers) > 0:
        # Create tomograms with cavity objects
        tomogram_files = create_cavity_tomograms(particles, shape_vertices, cavity_centers, cavity_radii)
        
        # Visualize the tomograms
        visualize_cavity_tomograms(tomogram_files)
        
        # Create comparison tomogram
        comparison_file = create_comparison_tomogram(particles, shape_vertices, cavity_centers, cavity_radii)
        
        print(f"\n" + "="*60)
        print("SUMMARY")
        print("="*60)
        print("Created tomogram files:")
        for filename in tomogram_files + [comparison_file]:
            if filename and os.path.exists(filename):
                print(f"  - {filename}")
    else:
        print("\nNo cavities detected. Consider:")
        print("1. Using a different particle file with more complex structure")
        print("2. Adjusting cavity detection parameters")
        print("3. Using the synthetic clathrate data (run without arguments)")

if __name__ == "__main__":
    main()

