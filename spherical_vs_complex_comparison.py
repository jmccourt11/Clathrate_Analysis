#!/usr/bin/env python3
"""
Spherical vs Complex Geometry Tomogram Comparison

This script compares the performance and results of creating tomograms
with simple spherical particles vs complex bipyramid geometries.

Author: Generated Script
Date: Current
"""

import numpy as np
import os
import sys
import time

# Add src directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from clathrate_analysis import (
    parse_particles_and_shape, 
    get_bipyramid_geometry,
    voxelize_particles
)
from tomogram_utils import save_voxel_grid_as_tiff, plot_3d_tomogram

def create_spherical_voxel_grid_simple(particles, grid_size=64, padding=0.1, sphere_radius=0.2):
    """
    Simple spherical voxelization for comparison
    """
    positions = np.array([pos for pos, _ in particles])
    
    # Calculate bounding box
    min_corner = positions.min(axis=0)
    max_corner = positions.max(axis=0)
    box_size = max_corner - min_corner
    
    # Add padding
    min_corner = min_corner - padding * box_size
    max_corner = max_corner + padding * box_size
    
    # Create grid
    edges = [np.linspace(min_corner[d], max_corner[d], grid_size + 1) for d in range(3)]
    centers = [0.5 * (edges[d][:-1] + edges[d][1:]) for d in range(3)]
    voxel_grid = np.zeros((grid_size, grid_size, grid_size), dtype=np.float32)
    
    # Create coordinate grids
    z_grid, y_grid, x_grid = np.meshgrid(
        np.arange(grid_size), 
        np.arange(grid_size), 
        np.arange(grid_size), 
        indexing='ij'
    )
    
    x_coords = centers[0][x_grid]
    y_coords = centers[1][y_grid]
    z_coords = centers[2][z_grid]
    
    # Fill spheres
    for pos, _ in particles:
        distances = np.sqrt(
            (x_coords - pos[0])**2 + 
            (y_coords - pos[1])**2 + 
            (z_coords - pos[2])**2
        )
        voxel_grid[distances <= sphere_radius] = 1.0
    
    return voxel_grid, edges

def benchmark_voxelization_methods(particles, shape_vertices=None):
    """
    Compare performance of different voxelization methods
    """
    print("="*60)
    print("PERFORMANCE COMPARISON")
    print("="*60)
    
    grid_sizes = [32, 64, 128]
    results = {}
    
    for grid_size in grid_sizes:
        print(f"\nTesting grid size: {grid_size}³ = {grid_size**3:,} voxels")
        print("-" * 40)
        
        # Test spherical voxelization
        print("Testing spherical voxelization...")
        start_time = time.time()
        sphere_grid, sphere_edges = create_spherical_voxel_grid_simple(
            particles, grid_size=grid_size, sphere_radius=0.15
        )
        sphere_time = time.time() - start_time
        sphere_filled = np.sum(sphere_grid > 0)
        
        print(f"  Time: {sphere_time:.2f} seconds")
        print(f"  Filled voxels: {sphere_filled:,}")
        print(f"  Fill rate: {sphere_filled/sphere_time:.0f} voxels/sec")
        
        # Test complex geometry voxelization (if shape_vertices available)
        if shape_vertices is not None:
            print("Testing complex geometry voxelization...")
            start_time = time.time()
            complex_grid, complex_edges = voxelize_particles(
                particles, grid_size=grid_size, padding=0.1, 
                shape_vertices=shape_vertices
            )
            complex_time = time.time() - start_time
            complex_filled = np.sum(complex_grid > 0)
            
            print(f"  Time: {complex_time:.2f} seconds")
            print(f"  Filled voxels: {complex_filled:,}")
            print(f"  Fill rate: {complex_filled/complex_time:.0f} voxels/sec")
            
            speedup = complex_time / sphere_time
            print(f"  Spherical is {speedup:.1f}x faster")
        else:
            complex_time = None
            complex_filled = None
            speedup = None
            print("  Complex geometry: No shape vertices available")
        
        results[grid_size] = {
            'sphere_time': sphere_time,
            'sphere_filled': sphere_filled,
            'complex_time': complex_time,
            'complex_filled': complex_filled,
            'speedup': speedup
        }
    
    return results

def compare_visual_results(particles, shape_vertices=None):
    """
    Create side-by-side tomograms for visual comparison
    """
    print("\n" + "="*60)
    print("VISUAL COMPARISON")
    print("="*60)
    
    grid_size = 64  # Moderate resolution for comparison
    
    # Create spherical tomogram
    print("Creating spherical particle tomogram...")
    sphere_grid, sphere_edges = create_spherical_voxel_grid_simple(
        particles, grid_size=grid_size, sphere_radius=0.15
    )
    sphere_filename = "comparison_spherical.tif"
    save_voxel_grid_as_tiff(sphere_grid, sphere_filename, pixel_size=1.0)
    
    # Create complex geometry tomogram (if possible)
    if shape_vertices is not None:
        print("Creating complex geometry tomogram...")
        complex_grid, complex_edges = voxelize_particles(
            particles, grid_size=grid_size, padding=0.1, 
            shape_vertices=shape_vertices
        )
        complex_filename = "comparison_complex.tif"
        save_voxel_grid_as_tiff(complex_grid, complex_filename, pixel_size=1.0)
    else:
        print("No shape vertices available for complex geometry")
        complex_filename = None
    
    # Visualize both
    print("\nVisualizing spherical tomogram...")
    plot_3d_tomogram(sphere_filename, plot_type='isosurface', threshold=0.5, colorscale='Blues')
    
    if complex_filename:
        print("Visualizing complex geometry tomogram...")
        plot_3d_tomogram(complex_filename, plot_type='isosurface', threshold=0.5, colorscale='Reds')
    
    return sphere_filename, complex_filename

def analyze_volume_differences(particles, shape_vertices=None):
    """
    Analyze volume and density differences between methods
    """
    print("\n" + "="*60)
    print("VOLUME ANALYSIS")
    print("="*60)
    
    grid_size = 128  # High resolution for accurate volume calculation
    
    # Spherical analysis
    sphere_radii = [0.1, 0.15, 0.2, 0.25]
    print("Spherical particles volume analysis:")
    print("Radius | Filled Voxels | Volume Fraction")
    print("-" * 40)
    
    sphere_results = {}
    for radius in sphere_radii:
        sphere_grid, _ = create_spherical_voxel_grid_simple(
            particles, grid_size=grid_size, sphere_radius=radius
        )
        filled = np.sum(sphere_grid > 0)
        fraction = filled / sphere_grid.size
        print(f"{radius:6.2f} | {filled:11,} | {fraction:13.4f}")
        sphere_results[radius] = {'filled': filled, 'fraction': fraction}
    
    # Complex geometry analysis (if available)
    if shape_vertices is not None:
        print("\nComplex geometry volume:")
        complex_grid, _ = voxelize_particles(
            particles, grid_size=grid_size, padding=0.1, 
            shape_vertices=shape_vertices
        )
        complex_filled = np.sum(complex_grid > 0)
        complex_fraction = complex_filled / complex_grid.size
        print(f"Filled Voxels: {complex_filled:,}")
        print(f"Volume Fraction: {complex_fraction:.4f}")
        
        # Find closest spherical equivalent
        best_match_radius = None
        best_match_diff = float('inf')
        for radius, data in sphere_results.items():
            diff = abs(data['fraction'] - complex_fraction)
            if diff < best_match_diff:
                best_match_diff = diff
                best_match_radius = radius
        
        print(f"\nClosest spherical equivalent: radius = {best_match_radius:.2f}")
        print(f"Volume fraction difference: {best_match_diff:.4f}")
    else:
        print("\nNo complex geometry available for comparison")

def create_test_particles(n_particles=50):
    """
    Create test particles for comparison
    """
    particles = []
    
    # Create particles in a rough sphere arrangement
    for i in range(n_particles):
        # Random position within a sphere
        theta = np.random.uniform(0, 2*np.pi)
        phi = np.random.uniform(0, np.pi)
        r = np.random.uniform(0.5, 3.0)
        
        x = r * np.sin(phi) * np.cos(theta)
        y = r * np.sin(phi) * np.sin(theta)
        z = r * np.cos(phi)
        
        # Random orientation
        qw = np.random.uniform(0.5, 1.0)
        qx = np.random.uniform(-0.5, 0.5)
        qy = np.random.uniform(-0.5, 0.5)
        qz = np.random.uniform(-0.5, 0.5)
        
        # Normalize quaternion
        norm = np.sqrt(qw**2 + qx**2 + qy**2 + qz**2)
        quat = (qw/norm, qx/norm, qy/norm, qz/norm)
        
        particles.append(((x, y, z), quat))
    
    return particles

def main():
    """
    Main comparison function
    """
    print("="*60)
    print("SPHERICAL vs COMPLEX GEOMETRY COMPARISON")
    print("="*60)
    
    # Try to load real particle data
    particle_file = None  # Set to your .pos file path if available
    
    if particle_file and os.path.exists(particle_file):
        print(f"Loading particles from: {particle_file}")
        particles, shape_vertices, shape_color, simulation_data = parse_particles_and_shape(particle_file)
        print(f"Loaded {len(particles)} particles")
    else:
        print("Creating test particles for comparison...")
        particles = create_test_particles(n_particles=100)
        # Get default bipyramid geometry for comparison
        shape_vertices, _, _ = get_bipyramid_geometry()
        print(f"Created {len(particles)} test particles")
    
    try:
        # Performance benchmark
        benchmark_results = benchmark_voxelization_methods(particles, shape_vertices)
        
        # Visual comparison
        sphere_file, complex_file = compare_visual_results(particles, shape_vertices)
        
        # Volume analysis
        analyze_volume_differences(particles, shape_vertices)
        
        print("\n" + "="*60)
        print("COMPARISON SUMMARY")
        print("="*60)
        
        print("\nSPHERICAL PARTICLES:")
        print("Advantages:")
        print("  ✓ Much faster computation")
        print("  ✓ Lower memory usage")
        print("  ✓ Simpler implementation")
        print("  ✓ Orientation-independent")
        print("  ✓ Good for prototyping")
        
        print("\nDisadvantages:")
        print("  ✗ Less realistic geometry")
        print("  ✗ Cannot capture anisotropic effects")
        print("  ✗ May not represent real particle shapes")
        
        print("\nCOMPLEX GEOMETRY:")
        print("Advantages:")
        print("  ✓ Realistic particle shapes")
        print("  ✓ Captures orientation effects")
        print("  ✓ Better for scientific accuracy")
        print("  ✓ Can model specific materials")
        
        print("\nDisadvantages:")
        print("  ✗ Slower computation")
        print("  ✗ Higher memory usage")
        print("  ✗ More complex implementation")
        print("  ✗ Requires shape definitions")
        
        print("\nRECOMMENDATIONS:")
        print("• Use spherical for rapid prototyping and testing")
        print("• Use complex geometry for final simulations")
        print("• Consider hybrid approaches for large systems")
        print("• Validate spherical approximations against complex geometry")
        
        print(f"\nGenerated comparison files:")
        if sphere_file and os.path.exists(sphere_file):
            print(f"  - {sphere_file}")
        if complex_file and os.path.exists(complex_file):
            print(f"  - {complex_file}")
        
    except Exception as e:
        print(f"Error during comparison: {e}")
        print("Make sure you have the required dependencies installed:")
        print("  pip install numpy plotly scipy tifffile")
        raise

if __name__ == "__main__":
    main()
