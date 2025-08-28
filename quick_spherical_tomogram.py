#!/usr/bin/env python3
"""
Quick Spherical Tomogram Creator

A minimal script for quickly creating tomograms with spherical particles.
Perfect for testing, prototyping, or when you need fast results.

Usage:
    python quick_spherical_tomogram.py [particle_file.pos] [--radius 0.15] [--grid-size 64]
"""

import numpy as np
import os
import sys
import argparse

# Add src directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from clathrate_analysis import parse_particles_and_shape
from tomogram_utils import save_voxel_grid_as_tiff, plot_3d_tomogram

def quick_spherical_voxelization(particles, grid_size=64, sphere_radius=0.15, padding=0.1):
    """
    Fast spherical particle voxelization
    """
    positions = np.array([pos for pos, _ in particles])
    
    # Bounding box
    min_corner = positions.min(axis=0) - padding * (positions.max(axis=0) - positions.min(axis=0))
    max_corner = positions.max(axis=0) + padding * (positions.max(axis=0) - positions.min(axis=0))
    
    # Grid setup
    x_edges = np.linspace(min_corner[0], max_corner[0], grid_size + 1)
    y_edges = np.linspace(min_corner[1], max_corner[1], grid_size + 1)
    z_edges = np.linspace(min_corner[2], max_corner[2], grid_size + 1)
    
    x_centers = 0.5 * (x_edges[:-1] + x_edges[1:])
    y_centers = 0.5 * (y_edges[:-1] + y_edges[1:])
    z_centers = 0.5 * (z_edges[:-1] + z_edges[1:])
    
    # Create coordinate grids
    Z, Y, X = np.meshgrid(z_centers, y_centers, x_centers, indexing='ij')
    
    # Initialize voxel grid
    voxel_grid = np.zeros((grid_size, grid_size, grid_size), dtype=np.float32)
    
    # Fill spheres
    for pos, _ in particles:
        distances = np.sqrt((X - pos[0])**2 + (Y - pos[1])**2 + (Z - pos[2])**2)
        voxel_grid[distances <= sphere_radius] = 1.0
    
    return voxel_grid

def create_synthetic_spheres(n=27):
    """
    Create synthetic spherical particles in a cubic arrangement
    """
    particles = []
    side = int(np.ceil(n**(1/3)))
    spacing = 1.0
    
    count = 0
    for i in range(side):
        for j in range(side):
            for k in range(side):
                if count >= n:
                    break
                
                x = i * spacing - (side-1) * spacing / 2
                y = j * spacing - (side-1) * spacing / 2
                z = k * spacing - (side-1) * spacing / 2
                
                particles.append(((x, y, z), (1, 0, 0, 0)))
                count += 1
            if count >= n:
                break
        if count >= n:
            break
    
    return particles

def main():
    """
    Main function with command line interface
    """
    parser = argparse.ArgumentParser(description='Create spherical particle tomograms quickly')
    parser.add_argument('particle_file', nargs='?', help='Particle data file (.pos)')
    parser.add_argument('--radius', '-r', type=float, default=0.15, help='Sphere radius (default: 0.15)')
    parser.add_argument('--grid-size', '-g', type=int, default=64, help='Grid size (default: 64)')
    parser.add_argument('--output', '-o', default='quick_spherical.tif', help='Output filename (default: quick_spherical.tif)')
    parser.add_argument('--no-plot', action='store_true', help='Skip visualization')
    parser.add_argument('--synthetic', '-s', type=int, help='Create N synthetic particles instead of loading file')
    
    args = parser.parse_args()
    
    print("="*50)
    print("QUICK SPHERICAL TOMOGRAM CREATOR")
    print("="*50)
    
    # Load or create particles
    if args.synthetic:
        print(f"Creating {args.synthetic} synthetic particles...")
        particles = create_synthetic_spheres(args.synthetic)
    elif args.particle_file and os.path.exists(args.particle_file):
        print(f"Loading particles from: {args.particle_file}")
        particles, _, _, _ = parse_particles_and_shape(args.particle_file)
    else:
        print("Creating default synthetic particles...")
        particles = create_synthetic_spheres(27)
    
    print(f"Using {len(particles)} particles")
    print(f"Sphere radius: {args.radius}")
    print(f"Grid size: {args.grid_size}³ = {args.grid_size**3:,} voxels")
    
    # Create voxel grid
    print("Voxelizing particles...")
    import time
    start_time = time.time()
    
    voxel_grid = quick_spherical_voxelization(
        particles, 
        grid_size=args.grid_size, 
        sphere_radius=args.radius
    )
    
    elapsed = time.time() - start_time
    filled_voxels = np.sum(voxel_grid > 0)
    
    print(f"Voxelization completed in {elapsed:.2f} seconds")
    print(f"Filled {filled_voxels:,} voxels ({filled_voxels/voxel_grid.size:.1%})")
    
    # Save tomogram
    print(f"Saving tomogram: {args.output}")
    save_voxel_grid_as_tiff(voxel_grid, args.output, pixel_size=1.0)
    
    # Visualize (unless disabled)
    if not args.no_plot:
        print("Creating visualization...")
        plot_3d_tomogram(args.output, plot_type='isosurface', threshold=0.5)
    
    print(f"\nDone! Tomogram saved as: {args.output}")
    
    # Print usage statistics
    memory_mb = voxel_grid.nbytes / (1024**2)
    print(f"\nStatistics:")
    print(f"  Processing time: {elapsed:.2f} seconds")
    print(f"  Memory usage: {memory_mb:.1f} MB")
    print(f"  Voxelization rate: {filled_voxels/elapsed:.0f} voxels/sec")

if __name__ == "__main__":
    main()
